from transformers import AutoModelForCausalLM

from transformers import pipeline

import evaluate

def eval_transformer(starts, trues, model_name): 
    
    gpt_tokenizer = AutoTokenizer.from_pretrained(model_name)
    gpt_model = AutoModelForCausalLM.from_pretrained(model_name)

    generator = pipeline(
        task="text-generation",
        model=gpt_model,
        tokenizer=gpt_tokenizer,
        device=-1)

    out = generator(
                    starts,
                    #max_length=50, ## Установил как в generate() у LSTM, но...
                    truncation=True, 
                    max_new_tokens = 15, ## Перебивает max_length, default=256.
                    num_return_sequences=1,
                    do_sample=True,
                    top_p=0.95,
                    temperature=0.8
                    )
        

    gpt_preds = [out[i][0]["generated_text"] for i in range(len(out))]
    gpt_preds = [gpt_pred[len(start):] for start,gpt_pred in zip(starts, gpt_preds)]
    gpt_preds = [pred.replace('\n', '') for pred in gpt_preds]

    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=gpt_preds,
                            references=trues,
                            use_stemmer=True)
            
    rough1_gpt, rough2_gpt = results['rouge1'], results['rouge2']

    return rough1_gpt, rough2_gpt, gpt_preds