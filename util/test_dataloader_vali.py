from trl import SFTTrainer, trainer, DataCollatorForCompletionOnlyLM
import json
from transformers import AutoTokenizer

class config:
    train_size=10000
    num_epoch=50
    acc_steps=16
    lr=5e-5
    ds_config="config/ds_config_zero3.json"
    model_name="google/gemma-2b" 
    prefix="/share/edc/home/wendaxu"
    max_inp_length=512
    max_tar_length=128

def formatting_prompts_func(example):
    text = f"SUBREDDIT: {example['subreddit']}\nPOST: {example['content']}\nPlease summarize the post by given subreddit: "
    return text 

with open('sft_data/dev.jsonl') as f:
    eval_dataset = [json.loads(line) for line in f]

tokenizer = AutoTokenizer.from_pretrained(config.model_name)

eval_dataset = trainer.ConstantLengthDataset(
    tokenizer,
    eval_dataset,
    formatting_func=formatting_prompts_func,
    seq_length=config.da+config.max_tar_length,
)

for batch in eval_dataset:
    print(batch)
