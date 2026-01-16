import torch
from torch.utils.data import Dataset

from PIL import Image
import timm
import numpy as np
import pandas as pd

from transformers import AutoTokenizer

import albumentations as A


class MultimodalDataset(Dataset):

    def __init__(self, config, transforms, cals_scaler, masses_scaler, ds_type="train"):
        df = pd.read_csv(config.DISHES_DF_PATH)
        if ds_type == "train":
            self.df = df[df['split']=='train'].reset_index(drop=True)
        else:
            self.df = df[df['split']=='test'].reset_index(drop=True)

        self.df_ingrs = pd.read_csv(config.INGRIDIENTS_DF_PATH, index_col='id')
        self.transforms = transforms
        self.calories = cals_scaler.transform(self.df['total_calories'].values.reshape(-1,1)).flatten()
        self.masses = masses_scaler.transform(self.df['total_mass'].values.reshape(-1,1)).flatten()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.loc[idx, 'ingredients']
        text = [self.df_ingrs.loc[j].item() for j in [int(i.lstrip('ingr_0')) for i in text.split(';')]]
        text = ', '.join(text)

        dish_id = self.df.loc[idx, 'dish_id']
        image = Image.open(f'data/images/{dish_id}/rgb.png').convert('RGB')
        image = self.transforms(image=np.array(image))['image']
        
        mass = self.masses[idx]
        label = self.calories[idx]

        return {'text': text, 'image': image, 'mass': mass, 'label': label}


def collate_fn(batch, tokenizer):
    texts = [item["text"] for item in batch]
    images = torch.stack([item["image"] for item in batch])
    masses = torch.tensor([item["mass"] for item in batch]).unsqueeze(1).float()
    labels = torch.tensor([item["label"] for item in batch]).unsqueeze(1).float()

    tokenized_input = tokenizer(texts,
                                padding='longest',
                                truncation=True, 
                                return_tensors="pt")

    return {
        'images': images,
        'input_ids': tokenized_input["input_ids"],
        'attention_mask': tokenized_input["attention_mask"], 
        'masses': masses, 
        'labels': labels,
        }


def get_transforms(config, ds_type="train"):
    cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)

    if ds_type == "train":
        transforms = A.Compose(
            [
                A.Resize(cfg.input_size[1], cfg.input_size[2], p=1.0), 
                A.Affine(scale=(0.9, 1.1),
                         rotate=(-10, 10),
                         translate_percent=(-0.05, 0.05),
                         shear=(-20, 20),
                         fill=255.,
                         p=0.7),
                A.ColorJitter(brightness=0.3,
                              contrast=0.3,
                              saturation=0.3,
                              hue=0.3,
                              p=0.7),
                A.OneOf([#A.Downscale(scale_min=0.9, scale_max=1.0, p=1.0),
                         A.GaussianBlur(blur_limit=(3, 3), p=1.0),
                         A.ImageCompression(quality_lower=60, quality_upper=80, p=1.0),
                         ], p=0.7),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                A.ToTensorV2(p=1.0)
            ],
            seed=42,
        )
    else:
        transforms = A.Compose(
            [
                A.Resize(cfg.input_size[1], cfg.input_size[2], p=1.0), 
                A.Normalize(mean=cfg.mean, std=cfg.std),
                A.ToTensorV2(p=1.0)
            ],
            seed=42,
        )

    return transforms