import torch 

from tqdm import tqdm

def train_epoch(model, loader, device, tokenizer):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0) # paddingи совпадают в ids и labels
    
    model = model.to(device)
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader):
        ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        lengths = batch['lengths'].to(device)

        optimizer.zero_grad()
        logits, _ = model(ids, lengths)
        loss = criterion(logits.view(-1, tokenizer.vocab_size), labels.view(-1)) # paddingи совпадают в ids и labels
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
    
    mean_loss = total_loss / len(loader)
    
    return mean_loss