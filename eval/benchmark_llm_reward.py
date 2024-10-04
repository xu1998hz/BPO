import json
import click
import json
from typing import TypeVar, Iterable, List
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

T = TypeVar('T')

def batchify(data: Iterable[T], batch_size: int) -> Iterable[List[T]]:
    assert batch_size > 0
    batch = []
    for item in data:
        # Yield next batch
        if len(batch) == batch_size:
            yield batch
            batch = []
        batch.append(item)
    # Yield last un-filled batch
    if len(batch) != 0:
        yield batch

@click.command()
@click.option('-file_name', help='sample_10000_12000.txt')
@click.option('-data_split', type=str)
@click.option('-ref_addr', type=str, default=None)
@click.option('-start_index', type=int, default=0)
@click.option('-end_index', type=int, default=None)
@click.option('-task_type', type=str)
@click.option('-max_length', type=int, default=640)
@click.option('-batch_size', type=int, default=32)
@click.option('-special_tok', type=bool)
def main(file_name, data_split, ref_addr, start_index, end_index, task_type, max_length, batch_size, special_tok):
    if special_tok:
        data = open(file_name).readlines()
        data = ''.join(data).split('[SEP_WENDA]')
    else:
        data = open(file_name).readlines()
        data = [ele[:-1] for ele in data]
    with open(f'sft_{task_type}_data/{data_split}.jsonl') as f:
        eval_dataset = [json.loads(line) for line in f][start_index:end_index]

    if ref_addr:
        ref_txt_ls = open(ref_addr, 'r').readlines()
        ref_txt_ls = [line[:-1] for line in ref_txt_ls]

    reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
    rank_model, tokenizer = AutoModelForSequenceClassification.from_pretrained(reward_name).to('cuda'), AutoTokenizer.from_pretrained(reward_name)

    count = 0
    total = 0
    reward_diff = 0
    p_ls, better_ls, worse_ls = [],[],[]
    with torch.no_grad():
        if ref_addr:
            for response_ls, ref_ls, eval_data_batch in zip(batchify(data, batch_size), batchify(ref_txt_ls, batch_size), batchify(eval_dataset, batch_size)):
                # evaluate response against human written reference
                inputs = tokenizer([ele['prompt'] for ele in eval_data_batch], ref_ls, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to('cuda')
                ref_score_ls = rank_model(**inputs).logits.cpu().detach().tolist()

                inputs = tokenizer([ele['prompt'] for ele in eval_data_batch], response_ls, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to('cuda')
                summ_score_ls =  rank_model(**inputs).logits.cpu().detach().tolist()
                for ref_score, summ_score, p_ele, summ_ele, ref_ele in zip(ref_score_ls, summ_score_ls, eval_data_batch, response_ls, ref_ls):
                    reward_diff+=summ_score[0]-ref_score[0]
                    if summ_score[0] > ref_score[0]:
                        better_ls+=[summ_ele]
                        worse_ls+=[ref_ele]
                        count += 1
                    else:
                        better_ls+=[ref_ele]
                        worse_ls+=[summ_ele]
                    p_ls+=[p_ele]
                    total += 1
        else:
            for response_ls, eval_data_batch in zip(batchify(data, batch_size), batchify(eval_dataset, batch_size)):
                inputs = tokenizer([ele['prompt'] for ele in eval_data_batch], [ele['label'] for ele in eval_data_batch], return_tensors='pt', padding=True, truncation=True, max_length=max_length).to('cuda')
                ref_score_ls = rank_model(**inputs).logits.cpu().detach().tolist()
        
                inputs = tokenizer([ele['prompt'] for ele in eval_data_batch], response_ls, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to('cuda')
                summ_score_ls =  rank_model(**inputs).logits.cpu().detach().tolist()
                for ref_score, summ_score, p_ele, summ_ele, ref_ele in zip(ref_score_ls, summ_score_ls, eval_data_batch, response_ls, [ele['label'] for ele in eval_data_batch]):
                    reward_diff+=summ_score[0]-ref_score[0]
                    if summ_score[0] > ref_score[0]:
                        better_ls+=[summ_ele]
                        worse_ls+=[ref_ele]
                        count += 1
                    else:
                        better_ls+=[ref_ele]
                        worse_ls+=[summ_ele]
                    p_ls+=[p_ele]
                    total += 1
                    
        print(total)
        print("Win Rate: ", count/total)
        print("Reward Diff: ", reward_diff/total)

if __name__ == "__main__":
    main()