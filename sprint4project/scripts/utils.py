import os
import random
from functools import partial

import numpy as np

import timm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import AutoModel, AutoTokenizer

from scripts.dataset import MultimodalDataset, collate_fn, get_transforms
import joblib

from tqdm import tqdm

class MultimodalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        self.image_model = timm.create_model(config.IMAGE_MODEL_NAME,
                                             pretrained=True,
                                             num_classes=0)

        self.mass_proj = nn.Sequential(nn.Linear(1, config.MASS_PROJECTION_DIM), 
                                       nn.LayerNorm(config.MASS_PROJECTION_DIM), 
                                       nn.ReLU(), 
                                       nn.Linear(config.MASS_PROJECTION_DIM, config.MASS_PROJECTION_DIM), 
                                       nn.ReLU())
        
        concat_dim = self.text_model.config.hidden_size + self.image_model.num_features + config.MASS_PROJECTION_DIM

        self.regressor = nn.Sequential(nn.Linear(concat_dim, concat_dim // 2),
                                       nn.LayerNorm(concat_dim // 2),
                                       nn.ReLU(),
                                       nn.Dropout(0.3),
                                       nn.Linear(concat_dim // 2, 1))

    def forward(self, image, input_ids, attention_mask, mass):
        token_embeddings = self.text_model(input_ids, attention_mask).last_hidden_state
        
        ## mean pooling по токенам
        mask = attention_mask.unsqueeze(2).expand_as(token_embeddings)
        masked_out = token_embeddings * mask
        summed = masked_out.sum(dim=1)
        lengths = attention_mask.sum(dim=1).unsqueeze(1)
        
        text_emb = summed / lengths
        image_emb = self.image_model(image)
        mass_emb = self.mass_proj(mass) 

        fused_emb = torch.cat([text_emb, image_emb, mass_emb], dim = 1)

        output = self.regressor(fused_emb)

        return output

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def set_requires_grad(module, unfreeze_pattern="", verbose=False):
    if len(unfreeze_pattern) == 0:
        for _, param in module.named_parameters():
            param.requires_grad = False
        return

    pattern = unfreeze_pattern.split("|")

    for name, param in module.named_parameters():
        if any([name.startswith(p) for p in pattern]):
            param.requires_grad = True
            if verbose:
                print(f"Разморожен слой: {name}")
        else:
            param.requires_grad = False


def train(config, device):
    seed_everything(config.SEED)

    # Инициализация модели
    model = MultimodalModel(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
    
    ## Замораживаю модели полностью (разморожу после 10 эпохи)
    set_requires_grad(model.text_model, unfreeze_pattern='', verbose=True)
    set_requires_grad(model.image_model, unfreeze_pattern='', verbose=True)

    # Оптимизатор с разными LR
    optimizer = AdamW([{'params': model.text_model.parameters(),
                        'lr': config.TEXT_LR}, 
                        {'params': model.image_model.parameters(),
                         'lr': config.IMAGE_LR},
                        {'params': model.mass_proj.parameters(), 
                         'lr': config.MASS_PROJECTOR_LR},
                        {'params': model.regressor.parameters(), 
                         'lr': config.REGRESSOR_LR}])

    criterion = nn.MSELoss()

    # Загрузка данных
    train_transforms = get_transforms(config,  ds_type="train")
    val_transforms = get_transforms(config, ds_type="val")
    cals_scaler = joblib.load('data/cal_scaler.joblib')
    masses_scaler = joblib.load('data/mass_scaler.joblib')

    train_dataset = MultimodalDataset(config, train_transforms, cals_scaler, masses_scaler, ds_type="train")
    val_dataset = MultimodalDataset(config, val_transforms, cals_scaler, masses_scaler, ds_type="val")

    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE,
                              shuffle=True,
                              collate_fn=partial(collate_fn,
                                                 tokenizer=tokenizer))
    val_loader = DataLoader(val_dataset,
                            batch_size=config.BATCH_SIZE,
                            shuffle=False,
                            collate_fn=partial(collate_fn,
                                               tokenizer=tokenizer))
    
    ## Обучение
    print("training started")
    best_mae = 1e12
    patience = 6
    for epoch in range(config.EPOCHS):
        ## Размораживаем слои после 10ой эпохи у уменьшаем batch_size
        if epoch == 11: 
            set_requires_grad(model.text_model, 
                              unfreeze_pattern=config.TEXT_MODEL_UNFREEZE, 
                              verbose=True)
            set_requires_grad(model.image_model, 
                              unfreeze_pattern=config.IMAGE_MODEL_UNFREEZE, 
                              verbose=True)
            train_loader = DataLoader(train_dataset, batch_size=config.FINE_TUNE_BATCH_SIZE, 
                                      shuffle=True,
                                      collate_fn=partial(collate_fn,
                                                         tokenizer=tokenizer))
            val_loader = DataLoader(val_dataset, batch_size=config.FINE_TUNE_BATCH_SIZE, 
                                    shuffle=False,
                                    collate_fn=partial(collate_fn,
                                                       tokenizer=tokenizer))
            
        model.train()
        
        total_loss = 0.0

        for batch in tqdm(train_loader):
            inputs = {'image': batch['images'].to(device),
                      'input_ids': batch['input_ids'].to(device),
                      'attention_mask': batch['attention_mask'].to(device),
                      'mass': batch['masses'].to(device)}
            
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            preds = model(**inputs)
            loss = criterion(preds, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Валидация
        _, val_mae = validate(model, val_loader, cals_scaler, device)

        print(f"Epoch {epoch}/{config.EPOCHS-1} | avg_loss: {total_loss/len(train_loader):.4f} | val_mae: {val_mae:.4f}")

        if val_mae < best_mae:
            best_mae = val_mae
            best_weights = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break 

    torch.save(best_weights, config.SAVE_PATH)

def validate(model, val_loader, cals_scaler, device):
    model.eval()
    criterion = nn.L1Loss(reduction='sum')
    
    total_loss = 0.0
    total_samples = 0
    all_preds = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = {'image': batch['images'].to(device),
                      'input_ids': batch['input_ids'].to(device),
                      'attention_mask': batch['attention_mask'].to(device),
                      'mass': batch['masses'].to(device)}
            
            labels = cals_scaler.inverse_transform(batch['labels'].cpu().numpy())

            preds = model(**inputs)
            preds = cals_scaler.inverse_transform(preds.cpu().numpy())
            all_preds.extend(preds.flatten())
            
            loss = np.sum(np.abs(labels.flatten() - preds.flatten()))
            
            total_loss += loss
            total_samples += len(labels)
       
    mae = total_loss / total_samples
    
    return all_preds, mae
