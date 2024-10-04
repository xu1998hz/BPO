from transformers import AutoTokenizer
from datasets import load_dataset
from typing import TypeVar, Iterable, List
import json

T = TypeVar('T')

def prompt_batchify(data: Iterable[T], batch_size: int) -> Iterable[List[T]]:
    assert batch_size > 0
    batch, batch_subreddit, batch_content = [], [], []
    for item in data:
        # Yield next batch
        if len(batch) == batch_size:
            yield batch, batch_subreddit, batch_content
            batch, batch_subreddit, batch_content = [], [], []    
        batch.append(f"SUBREDDIT: {item['subreddit']}\nPOST: {item['content']}\nPlease summarize the post by given subreddit: ")
        batch_subreddit.append(item['subreddit'])
        batch_content.append(item['content'])
    # Yield last un-filled batch
    if len(batch) != 0:
        yield batch, batch_subreddit, batch_content

data = load_dataset('/share/edc/home/wendaxu/tldr-17', trust_remote_code=True)
sampling_prompt_dataset=[data['train'][i] for i in range(10000, 20000, 1)]
tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b')

count = 0
cur_dict = {"sub": [], "cont": [], "prompt": []}
for prompts, batch_sub, batch_content in prompt_batchify(sampling_prompt_dataset, 1):
    if count == 16:
        break
    cur_len = len(tokenizer.tokenize(prompts[0]))
    if cur_len>2048:
       cur_dict["sub"]+=batch_sub
       cur_dict["cont"]+=batch_content
       cur_dict["prompt"]+=prompts
       count+=1

with open("oom.json", 'w') as f:
    json.dump(cur_dict, f)
    print("File is saved!")
