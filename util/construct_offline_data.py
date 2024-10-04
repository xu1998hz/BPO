from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import TypeVar, Iterable, Any, Dict, List, Optional, Tuple, Union
import json
import os
import click

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

reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
rank_model, tokenizer = AutoModelForSequenceClassification.from_pretrained(reward_name).to('cuda'), AutoTokenizer.from_pretrained(reward_name)

@click.command()
@click.option('-prefix')
@click.option('-data_type')
@click.option('-seed_1')
@click.option('-seed_2')
def main(seed_1, seed_2, prefix, data_type):
    lines_1 = open(f'outputs/sampling_sft_{seed_1}.txt', 'r').readlines()
    lines_1 = [ele[:-1] for ele in lines_1]
    lines_2 = open(f'outputs/sampling_sft_{seed_2}.txt', 'r').readlines()
    lines_2 = [ele[:-1] for ele in lines_2]

    with open(f'sft_data/sampling.jsonl') as f:
        prompts = [json.loads(line)['prompt'] for line in f][0:2000]

    for step, (prompt_batch, batch_1, batch_2) in enumerate(zip(batchify(prompts, 16), batchify(lines_1, 16), batchify(lines_2, 16))):
        prompt_ls, chosen_ls, reject_ls = list(prompt_batch), [], []
        inputs = tokenizer(prompt_batch, batch_1, return_tensors='pt', padding=True, truncation=True, max_length=640).to('cuda')
        score_1_ls = rank_model(**inputs).logits.cpu().detach().tolist()

        inputs = tokenizer(prompt_batch, batch_2, return_tensors='pt', padding=True, truncation=True, max_length=640).to('cuda')
        score_2_ls = rank_model(**inputs).logits.cpu().detach().tolist()

        for score_1, score_2, sample_1, sample_2 in zip(score_1_ls, score_2_ls, batch_1, batch_2):
            if score_1[0] > score_2[0]:
                chosen_ls+=[sample_1]
                reject_ls+=[sample_2]
            else:
                chosen_ls+=[sample_2]
                reject_ls+=[sample_1]

        selected_data = {'prompt': prompt_ls, 'chosen': chosen_ls, 'rejected': reject_ls}

        if not os.path.exists(f'{prefix}/{data_type}_train_data_{seed_1}'):
            os.mkdir(f'{prefix}/{data_type}_train_data_{seed_1}')
        if not os.path.exists(f'{prefix}/{data_type}_train_data_{seed_1}/epoch_0'):
            os.mkdir(f'{prefix}/{data_type}_train_data_{seed_1}/epoch_0')

        with open(f'{prefix}/{data_type}_train_data_{seed_1}/epoch_0/{step}_rank.json', 'w') as f:
            json.dump(selected_data, f)

if __name__ == "__main__":
    main()