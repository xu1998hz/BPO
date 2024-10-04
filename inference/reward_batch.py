from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import TypeVar, Iterable, List
import click
from datasets import load_dataset
import json
from tqdm import tqdm

T = TypeVar('T')

def prompt_batchify(data: Iterable[T], batch_size: int) -> Iterable[List[T]]:
    assert batch_size > 0

    batch = {"subreddit": [], "content": []}
    for item in data:
        # Yield next batch
        if len(batch) == batch_size:
            yield batch
            batch = []
            
        batch["subreddit"].append(item["subreddit"])
        batch["content"].append(item["content"])

    # Yield last un-filled batch
    if len(batch) != 0:
        yield batch

def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)
    
def logits_compute(tokenizer, model, subreddit_txts, post_txts, texts):
    inp_ls, label_ls, len_ls = [], [], []
    # batchify texts
    for subreddit_txt, post_txt, text in zip(subreddit_txts, post_txts, texts):
        prompt_len = len(tokenizer.tokenize(f"SUBREDDIT: {subreddit_txt}\nPOST: {post_txt}\nPlease summarize the post by given subreddit: "))
        temp_txt = f"SUBREDDIT: {subreddit_txt}\nPOST: {post_txt}\nPlease summarize the post by given subreddit: {text}"
        inp_ls += [temp_txt]
        label_ls += [[-100]*prompt_len+[1]*(len(tokenizer.tokenize(temp_txt))-prompt_len)]
        len_ls += [len(tokenizer.tokenize(temp_txt))]

    # pad to max for label list
    max_len = max(len_ls)
    pad_label_ls = []
    for ele in label_ls:
        if len(ele) < max_len:
            pad_label_ls+=[ele+[-100]*(max_len-len(ele))]
        else:
            pad_label_ls+=[ele]

    inputs = tokenizer(inp_ls, return_tensors="pt", padding=True, truncation=True, max_length=1024, add_special_tokens=False).to(model.device) 
    labels=torch.LongTensor(pad_label_ls)
    
    all_logits = model(input_ids=inputs["input_ids"], attention_mask=inputs['attention_mask']).logits.to('cpu')
    logits = _get_batch_logps(all_logits, labels, True)
    
    return logits

@click.command()
@click.option('-model_pi_name')
@click.option('-model_ref_name')
@click.option('-start_index', type=int)
@click.option('-end_index', type=int)
@click.option('-mode')
@click.option('-per_batch', type=int, default=3)
def main(model_pi_name, model_ref_name, start_index, end_index, mode, per_batch):
    data = load_dataset('webis/tldr-17')
    eval_dataset=[data['train'][i] for i in range(10000+start_index, 10000+end_index, 1)]
    cur_start_index, cur_end_index = 10000+start_index, 10000+end_index
    text_sample = json.load(open(f'{mode}_{cur_start_index}_{cur_end_index}_top-p.json'))
    
    logits_dict = {}

    tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b') 
    model_pi = AutoModelForCausalLM.from_pretrained(model_pi_name, use_safetensors=True, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model_pi.eval()

    with torch.no_grad():
        with tqdm(total=len(eval_dataset)) as pbar:
            for i in range(0, len(eval_dataset), per_batch):
                subreddit_txts, post_txts, texts = [], [], []
                for j in range(i, i+per_batch):
                    batch = eval_dataset[j]
                    prompt=f"SUBREDDIT: {batch['subreddit']}\nPOST: {batch['content']}\nPlease summarize the post by given subreddit: "
                    subreddit_txts.extend(batch['subreddit'] * 12)
                    post_txts.extend(batch['content'] * 12)
                    texts.extend(text_sample[prompt])
                    logits_dict[prompt] = {}
                logits = logits_compute(tokenizer, model_pi, subreddit_txts, post_txts, texts).tolist()
                for j in range(i, i+per_batch):
                    prompt=f"SUBREDDIT: {batch['subreddit']}\nPOST: {batch['content']}\nPlease summarize the post by given subreddit: "
                    logits_dict[prompt]['pi'] = logits[(j-i)*12:(j-i+1)*12]
                pbar.update(per_batch)
            # for batch in eval_dataset:
            #     prompt=f"SUBREDDIT: {batch['subreddit']}\nPOST: {batch['content']}\nPlease summarize the post by given subreddit: "
            #     logits_dict[prompt] = {}
            #     logits_dict[prompt]['pi'] = logits_compute(tokenizer, model_pi, batch['subreddit']*12, batch['content']*12, text_sample[prompt]).tolist()
            #     pbar.update(1)

    tokenizer = AutoTokenizer.from_pretrained(model_ref_name) 
    model_ref = AutoModelForCausalLM.from_pretrained(model_ref_name, use_safetensors=True, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model_ref.eval()

    with torch.no_grad():
        with tqdm(total=len(eval_dataset)) as pbar:
            for i in range(0, len(eval_dataset), per_batch):
                subreddit_txts, post_txts, texts = [], [], []
                for j in range(i, i+per_batch):
                    batch = eval_dataset[j]
                    prompt=f"SUBREDDIT: {batch['subreddit']}\nPOST: {batch['content']}\nPlease summarize the post by given subreddit: "
                    subreddit_txts.extend(batch['subreddit'] * 12)
                    post_txts.extend(batch['content'] * 12)
                    texts.extend(text_sample[prompt])
                    logits_dict[prompt] = {}
                logits = logits_compute(tokenizer, model_ref, subreddit_txts, post_txts, texts).tolist()
                for j in range(i, i+per_batch):
                    prompt=f"SUBREDDIT: {batch['subreddit']}\nPOST: {batch['content']}\nPlease summarize the post by given subreddit: "
                    logits_dict[prompt]['ref'] = logits[(j-i)*12:(j-i+1)*12]
                pbar.update(per_batch)
            # for batch in eval_dataset:
            #     prompt=f"SUBREDDIT: {batch['subreddit']}\nPOST: {batch['content']}\nPlease summarize the post by given subreddit: "
            #     logits_dict[prompt]['ref'] = logits_compute(tokenizer, model_ref, batch['subreddit']*12, batch['content']*12, text_sample[prompt]).tolist()
            #     pbar.update(1)
    
    prefix = model_pi_name.split('/')[-2]
    with open(f'{prefix}_{start_index}_{end_index}.json', 'w') as f:
        json.dump(logits_dict, f)
        print("File is saved!")

if __name__ == "__main__":
    main()
    