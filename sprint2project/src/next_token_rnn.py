import torch

import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class NextTokenRNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers, pad_idx=0):
        super().__init__()


        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim, vocab_size)


        self.init_weights()


    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)


    def forward(self, input_ids, lengths):
        embedded = self.embedding(input_ids)
        
        packed = pack_padded_sequence(
            embedded,
            lengths.to('cpu'),
            batch_first=True,
            enforce_sorted=False)
        
        rnn_out, _ = self.rnn(packed)
        
        output, output_lengths = pad_packed_sequence(rnn_out, 
                                                     batch_first=True)
        
        output_normed = self.norm(output)
        
        drpt = self.dropout(output_normed)
        
        logits = self.fc(drpt)

        return logits, output_lengths
    
    def generate(self, start_sequences, lengths, eos_token_id, device, max_sentence_lenght = 50):
        with torch.no_grad():
            current_batch = start_sequences.clone().to(device)
            current_lengths = lengths.clone().to(device)
            can_add_tokens = torch.ones(current_batch.size(0), dtype=torch.bool).to(device)
            # "продлеваем" батч по горизонтальной оси
            padding = torch.zeros((current_batch.size(0), max_sentence_lenght - current_batch.size(1)), dtype=torch.long).to(device)
            current_batch = torch.cat([current_batch, padding], dim=1) 
            
            generated = []
            
            while can_add_tokens.any(): 
                
                logits, real_lengths = self(current_batch, current_lengths)
               
                # Логиты для последних реальных токенов
                batch_idx = torch.arange(logits.size(0))
                time_idx = real_lengths - 1                           
                last_logits = logits[batch_idx, time_idx, :] 
        
                next_tokens = torch.argmax(last_logits, dim=-1)

                current_batch[can_add_tokens, current_lengths[can_add_tokens]] = next_tokens[can_add_tokens]
                
                current_lengths[can_add_tokens] += 1
                
                can_add_tokens *= (current_lengths < max_sentence_lenght)
                next_tokens *= can_add_tokens
                can_add_tokens *= ~(next_tokens == eos_token_id)
                
                generated.append(next_tokens)
                
        return torch.stack(generated, dim = 1).to(device)