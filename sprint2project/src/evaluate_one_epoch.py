from tqdm import tqdm

import evaluate

def evaluate_epoch(model, loader, device, tokenizer): 
    
    model.eval()
    total_rough1 = 0
    total_rough2 = 0
    init_sentences, trues_sentences, predicted_sentences = [], [], []
    
    for batch in tqdm(loader):
        start_sequences = batch['input_ids'].to(device)
        lengths = batch['lengths'].to(device)
        labels = batch['labels'].to(device)

        decoded_sentences = tokenizer.batch_decode(start_sequences.tolist(), skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels.tolist(), skip_special_tokens=True)

        preds = model.generate(start_sequences, lengths, tokenizer.sep_token_id, device)
        decoded_preds = tokenizer.batch_decode(preds.tolist(), skip_special_tokens=True)
        
        not_empty_strings = [len(string) != 0 for string in decoded_preds]
        
        decoded_preds = [item for item, flag in zip(decoded_preds, not_empty_strings) if flag]
        decoded_sentences = [item for item, flag in zip(decoded_sentences, not_empty_strings) if flag]
        decoded_labels = [item for item, flag in zip(decoded_labels, not_empty_strings) if flag]
        
        rouge = evaluate.load("rouge")
        results = rouge.compute(predictions=decoded_preds,
                                references=decoded_labels,
                                use_stemmer=True)
        
        rough1_, rough2_ = results['rouge1'], results['rouge2']

        total_rough1 += rough1_
        total_rough2 += rough2_

        init_sentences.append(decoded_sentences)
        trues_sentences.append(decoded_labels)
        predicted_sentences.append(decoded_preds)

    mean_rough1 = total_rough1 / len(loader)
    mean_rough2 = total_rough2 / len(loader)
    
    init_sentences = [item for sublist in init_sentences for item in sublist]
    trues_sentences = [item for sublist in trues_sentences for item in sublist]
    predicted_sentences = [item for sublist in predicted_sentences for item in sublist]
    
    return mean_rough1, mean_rough2, init_sentences, trues_sentences, predicted_sentences