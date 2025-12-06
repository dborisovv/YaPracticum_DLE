import re

def clean_text(text): 
    text = text.lower()
    text = re.sub(r'@\S+|https?://\S+|;\S+|:\S+|#\S+', " ", text)  #эмодзи, ссылки, упоминания
    text = re.sub(r"[^a-z0-9'\-\s]+", " ", text)
    text = re.sub(r'(?<!\w)-|-(?!\w)', " ", text) # оставялем тире между словами
    text = re.sub(r"\s+", " ", text).strip() 
    return text