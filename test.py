import os
# check if the file exitst

path = "./models/tm/3hvfq75j/target_tokenizer.json"
if not os.path.exists(path):
    raise FileNotFoundError(f"File not found: {path}")
