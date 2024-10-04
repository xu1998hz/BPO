from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
from typing import TypeVar, Iterable, List
import click
import json
from tqdm import tqdm

T = TypeVar('T')

def prompt_batchify(data: Iterable[T], batch_size: int) -> Iterable[List[T]]:
    assert batch_size > 0

    batch = []
    for item in data:
        # Yield next batch
        if len(batch) == batch_size:
            yield batch
            batch = []
            
        batch.append(f"SUBREDDIT: {item['subreddit']}\nPOST: {item['content']}\nPlease summarize the post by given subreddit: ")

    # Yield last un-filled batch
    if len(batch) != 0:
        yield batch

@click.command()
@click.option('-iter_index', type=int)
@click.option('-start_index', type=int)
@click.option('-end_index', type=int)
@click.option('-mode', type=str)
@click.option('-num_ret', type=int)
@click.option('-ckpt', type=str)
@click.option('-rep_index', type=str)
@click.option('-sample_strategy', type=str)
def main(iter_index, start_index, end_index, mode, num_ret, ckpt, rep_index, sample_strategy):
    # start_index = list(range(0, 2001, 250))[iter_index]
    # end_index = list(range(0, 2001, 250))[iter_index+1]

    tokenizer = AutoTokenizer.from_pretrained(ckpt) 
    model = AutoModelForCausalLM.from_pretrained(ckpt, use_safetensors=True, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

    # collect 2000 query during the inference, each query generates 16 responses
    data = load_dataset('webis/tldr-17')
    eval_dataset=[data['train'][i] for i in range(10000+start_index, 10000+end_index, 1)]
    ret_dict = {}
    
    with tqdm(total=len(eval_dataset)) as pbar:
        for prompts in prompt_batchify(eval_dataset, 1):
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024, add_special_tokens=False).to(model.device) 
            if sample_strategy == "top-p":
                output = model.generate(inputs=inputs.input_ids, max_new_tokens=128, do_sample=True, temperature=1.0, top_k=50, top_p=0.95, num_return_sequences=num_ret)
            elif sample_strategy == "greedy":
                output = model.generate(inputs=inputs.input_ids, max_new_tokens=128, do_sample=False, temperature=0, top_k=50, top_p=0.95, num_return_sequences=num_ret)
            else:
                print("We don't support other strategy!")
                exit(1)
            responses = tokenizer.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            ret_dict[prompts[0]]=responses
            pbar.update(1)
    
    with open(f'{mode}_{iter_index}_{start_index}_{end_index}_{rep_index}.json', 'w') as f:
        json.dump(ret_dict, f)
        print("File is saved!")

if __name__ == "__main__":
    main()