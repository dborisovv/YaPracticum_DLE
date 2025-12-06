from torch.utils.data import Dataset

from torch.nn.utils.rnn import pad_sequence

import torch

class NextTokenDataset(Dataset):
    def __init__(self, texts, tokenizer, val_ds = False):

        self.samples = []
        self.labels = []
        
        min_text_length = 10 if val_ds else 5
        for line in texts:
            length = len(line.split())
            
            if length < min_text_length: 
                continue
                
            token_ids = tokenizer.encode(line, add_special_tokens=True)
            token_ids = token_ids[1:] # Убираем [CLS]
      
            if val_ds:
                x = torch.tensor(token_ids[:int(0.75*length)], dtype = torch.long) # 3/4 исходного текста
                y = torch.tensor(token_ids[int(0.75*length):], dtype = torch.long)
            else: 
                x = torch.tensor(token_ids[:-1], dtype = torch.long)
                y = torch.tensor(token_ids[1:], dtype = torch.long)
            
            self.samples.append(x)    
            self.labels.append(y)

           
    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        return {'text': self.samples[idx], 'label': self.labels[idx]}
    

    
def collate_fn(batch):
    texts = [item['text'] for item in batch]
    labels = [item['label'] for item in batch]
    lengths = torch.tensor([len(text) for text in texts], dtype=torch.long)

    padded_texts = pad_sequence(texts, batch_first = True, padding_value = 0)
    padded_labels = pad_sequence(labels, batch_first = True, padding_value = 0)

    masks = (padded_texts != 0).long()
   
    return {
        'input_ids': padded_texts, 
        'labels': padded_labels,
        'masks': masks,            
        'lengths': lengths
           }