from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from typing import TypeVar, Iterable, List
import click
from tqdm import tqdm
import json
import random
import numpy as np
import os

T = TypeVar('T')

def prompt_batchify(data: Iterable[T], batch_size: int) -> Iterable[List[T]]:
    assert batch_size > 0

    batch = []
    for item in data:
        # Yield next batch
        if len(batch) == batch_size:
            yield batch
            batch = []
        
        batch.append(f"{item['prompt']}")

    # Yield last un-filled batch
    if len(batch) != 0:
        yield batch

# epoch_0_rand_rand_44_may_10_5e-05_0.0_resume_False_500_replay_0_offline_False_shuffle_True_reward_on_policy_False_ret_2_sample_per_interval_merged_1_sft
# epoch_0_rand_rand_44_may_10_5e-05_0.0_resume_False_500_replay_0_offline_False_shuffle_True_reward_on_policy_False_ret_2_sample_sft_1_sft
# CUDA_VISIBLE_DEVICES=3 python3 inference/inference_adapter.py -ckpt /data/user_data/xixu/wendaxu/june_1_all/epoch_0_rand_rand_44_may_10_5e-05_5.5_resume_False_500_replay_0_offline_False_shuffle_True_reward_on_policy_False_ret_60_sample_per_interval_merged_1_merge_on_interval -prefix  -data_split test -ensemble True -decode_strategy greedy -num_lora 5 -task_type tldr
@click.command()
@click.option('-ckpt', help="sft_lora_256")
@click.option('-prefix', help="rand_rand")
@click.option('-data_split', help="either dev or test", type=str)
@click.option('-ensemble', type=bool, default=False)
@click.option('-decode_strategy', type=str)
@click.option('-num_lora', type=int)
@click.option('-start_index', type=int, default=0)
@click.option('-end_index', type=int, default=None)
@click.option('-task_type', type=str)
@click.option('-batch_size', type=int, help="32, 256, 64")
def main(ckpt, data_split, prefix, ensemble, decode_strategy, start_index, end_index, num_lora, task_type, batch_size):
    # load in corresponding testing set
    with open(f'sft_{task_type}_data/{data_split}.jsonl') as f:
        eval_dataset = [json.loads(line) for line in f][start_index:end_index]

    model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", torch_dtype=torch.bfloat16, device_map="auto")
    model.config.use_cache=False
    if ckpt == "sft_lora_256":
        peft_model_id = "xu1998hz/0_sft_lora_256"
        model = PeftModel.from_pretrained(model, peft_model_id, is_trainable=False, adapter_name=f"model_0")

        for i in range(1, num_lora):    
            model.load_adapter(f"xu1998hz/{i}_sft_lora_256", adapter_name=f"model_{i}")
        max_inp_length=512
    elif ckpt[:2] == "hh":
        print("Perform hh")
        peft_model_id = f"/data/user_data/xixu/wendaxu/sft_hh/sft_42_sft_lora_256_hh/checkpoint-{ckpt[2:]}"
        model = PeftModel.from_pretrained(model, peft_model_id, is_trainable=False, adapter_name=f"model_0")

        if ensemble:  
            for i, seed in enumerate([43, 44, 45, 46]):
                peft_model_id = f"/data/user_data/xixu/wendaxu/sft_hh/sft_{seed}_sft_lora_256_hh/checkpoint-{ckpt[2:]}" 
                model.load_adapter(peft_model_id, adapter_name=f"model_{i+1}")
        save_name = f'{data_split}_{prefix}_{task_type}_{ckpt[2:]}.txt'
        max_inp_length=64
    elif ckpt[:4] == "harm":
        print("Perform Harm")
        peft_model_id = f"/data/user_data/xixu/sft_harm/sft_42_sft_lora_256_harm/checkpoint-{ckpt[4:]}"
        model = PeftModel.from_pretrained(model, peft_model_id, is_trainable=False, adapter_name=f"model_0")

        if ensemble:  
            for i, seed in enumerate([43, 44, 45, 46]):
                peft_model_id = f"/data/user_data/xixu/sft_harm/sft_{seed}_sft_lora_256_harm/checkpoint-{ckpt[4:]}"
                model.load_adapter(peft_model_id, adapter_name=f"model_{i+1}")
        save_name = f'{data_split}_{prefix}_{task_type}_{ckpt[4:]}.txt'
        max_inp_length=256
    else:
        save_name = f'{data_split}_{prefix}_{task_type}.txt'
        peft_model_id = f"{ckpt}/model_0"
        model = PeftModel.from_pretrained(model, peft_model_id, is_trainable=False, adapter_name=f"model_0")
        
        if ensemble:  
            for i in range(1, num_lora):
                peft_model_id = f"{ckpt}/model_{i}"
                model.load_adapter(peft_model_id, adapter_name=f"model_{i}")
        max_inp_length=512

    if ensemble:  
        weighted_adapter_name = "dpo_inference"
        model.add_weighted_adapter(
            adapters=[f"model_{i}" for i in range(num_lora)],
            weights=[1/num_lora]*num_lora,
            adapter_name=weighted_adapter_name,
            combination_type="linear"
        )
        model.set_adapter(weighted_adapter_name)
    else:
        model.set_adapter("model_0")

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(f'google/gemma-2b') 
    save_file = open(save_name, 'w')

    # all_outs=[]
    with torch.no_grad():
        with tqdm(total=len(eval_dataset)) as pbar:
            for prompts in prompt_batchify(eval_dataset, batch_size): 
                inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_inp_length).to(model.device) 
                if decode_strategy == "greedy":
                    output = model.generate(inputs=inputs.input_ids, max_new_tokens=128, do_sample=False, temperature=0, num_return_sequences=1)
                elif decode_strategy == "ancestral":
                    output = model.generate(inputs=inputs.input_ids, max_new_tokens=128, do_sample=True, num_beams=1, num_return_sequences=1)  
                elif decode_strategy == "top-p":
                    output = model.generate(inputs=inputs.input_ids, max_new_tokens=128, do_sample=True, temperature=1.0, top_k=0, top_p=0.95, num_return_sequences=1)
                else:
                    print("Your decoding strategy is not supported!")
                    exit(1)  
                outputs = tokenizer.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
                for out in outputs:
                    save_file.write(out.replace('\n','')+'\n')
                    pbar.update(1)
            #     all_outs+=outputs
            #     for _ in outputs:
            #         pbar.update(1)
            # save_file.write('[SEP_WENDA]'.join(all_outs))
    save_file.close()
    print(f"Save at {save_name}")

if __name__ == "__main__":
    main() 