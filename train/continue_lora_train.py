from trl import DPOTrainer
import torch
import wandb
from datasets import Dataset
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import click
import json
from transformers import TrainerCallback
import os
import numpy as np
import copy
import transformers
import datasets 
from datasets import load_dataset
from typing import TypeVar, Iterable, List
import random
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os
import time
from peft import PeftModel
from openai import OpenAI
from transformers.trainer_callback import PrinterCallback
import math

class config:
    step_per_feedback=1
    vali_size=100
    num_epoch=1
    batch_size=1
    acc_steps=16
    lr=5e-5
    num_query=1
    ds_config="config/ds_config_zero3.json"
    model_ref_name="xu1998hz/sft"
    max_new_token=128
    max_prompt_length=512

datasets.disable_progress_bar()
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

class SaveEvalOutputsCallback(TrainerCallback):
    def __init__(self, eval_dataset, model, tokenizer, output_dir, num_samples=20):
        self.eval_dataset = eval_dataset
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.num_samples = num_samples

    def on_evaluate(self, args, state):
        # Randomly select num_samples indices from the evaluation dataset
        indices = np.random.choice(len(self.eval_dataset), self.num_samples, replace=False)

        self.model.eval()
        with torch.no_grad():
            with open(os.path.join(self.output_dir, f'epoch_{state.epoch}.txt'), "w") as f:
                for idx in indices:
                    example = self.eval_dataset[idx]
                    inputs = self.tokenizer(example['prompt'], return_tensors="pt", truncation=True, max_length=config.max_prompt_length).to(args.device)
                    outputs = self.model.generate(**inputs, max_new_tokens=config.max_new_token, num_return_sequences=1, do_sample=False)
                    predicted_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    f.write(predicted_text)
                    f.write('\n')
                    f.write('='*70)
                    f.write('\n')
                    f.flush()
                
        self.model.train()

def completions_with_backoff_openai(client, system_prompt, prompt_txt, model_type, n):
    response = client.chat.completions.create(
        model=model_type,  # "gpt-3.5-turbo", "gpt-4"
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": prompt_txt},
        ],
        temperature=1.0,
        max_tokens=1024,
        top_p=1,
        n=n,
    ).choices[0].message.content
    return response


def set_up_trainer(model, model_index, training_args, train_feedback_data, tokenizer):
    dpo_trainer = DPOTrainer(
        model=model,
        # ref_model=ref_model,
        model_adapter_name=f"model_{model_index}",
        ref_adapter_name="sft",
        args=training_args,
        beta=0.5,
        train_dataset=train_feedback_data,
        tokenizer=tokenizer,
        max_length=config.max_new_token+config.max_prompt_length,
        max_prompt_length=config.max_prompt_length,
        # callbacks=[SaveEvalOutputsCallback(raw_eval_dataset, model, tokenizer, output_dir=f'sample_output/{mode}', num_samples=20)]
    )
    # dpo_trainer.remove_callback(PrinterCallback)
    return dpo_trainer

def set_up_single_lora(model, sft_lora_addr, adpater_name):
    model = PeftModel.from_pretrained(
        model,
        sft_lora_addr,
        is_trainable=True,
        adapter_name=adpater_name,
    )
    return model

def set_up_merge_lora(model, sft_lora_addr_ls, weighted_adapter_name):
    model_names=[]
    for i, ele in enumerate(sft_lora_addr_ls):    
        model.load_adapter(ele, adapter_name=f"model_{i}")
        model_names+=[f"model_{i}"]
    
    # perform model averaging on lora weights
    model.add_weighted_adapter(
        adapters=model_names,
        weights=[1/len(model_names)]*len(model_names),
        adapter_name=weighted_adapter_name,
        combination_type="linear"
    )
    model.set_adapter(weighted_adapter_name)

    # set all sft or weighted_adapter_name parameters to be non-grad
    for name, param in model.named_parameters():
        if weighted_adapter_name in name:
            param.requires_grad = False
    return model

@click.command()
@click.option('-mode', type=str, default="rand_rand")
@click.option('-save_step', type=int, default=200)
@click.option('-seed', type=int, default=43)
@click.option('-prefix', type=str, default="/mnt/data6/")
def main(mode, prefix, save_step, seed):
    # define all training specific data
    config.max_step=125

    if not os.path.exists('save_weights'):
        os.mkdir('save_weights')

    # load in all model parameters
    model = AutoModelForCausalLM.from_pretrained('google/gemma-2b', torch_dtype=torch.bfloat16, device_map="cpu", attn_implementation="flash_attention_2").to("cuda")
    tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b')

    wandb.init(project='DPO_Master', name=f"{mode}_{seed}_debug_{save_step}_continue", config=\
    {
        "seed": seed,
        "save_step": save_step,
        "epoch": config.num_epoch,
        "train batch size": config.batch_size * config.acc_steps,
        "lr": config.lr,
        "setting": mode
    })

    # initialize trainer
    training_args = TrainingArguments(
        disable_tqdm=True,
        report_to="none", # don't report to wandb through trainer
        per_device_train_batch_size=config.batch_size,
        max_steps=config.step_per_feedback,
        gradient_accumulation_steps=config.acc_steps, # simulate larger batch sizes
        output_dir='outputs',
        save_strategy="no",
        evaluation_strategy="no", 
        logging_strategy="steps",
        logging_steps=1,
        weight_decay=0,
        warmup_ratio=0,
        seed=None, 
        data_seed=None,
        remove_unused_columns=False,
        bf16=True
        # deepspeed=config.ds_config,
    )

    # You must not use the caches of kv, otherwise you won't learn
    model.config.use_cache = False

    # define my own lr scheduler
    optimizer = transformers.AdamW(model.parameters(), lr=config.lr)
    lr_scheduler = transformers.get_scheduler("cosine", optimizer, num_warmup_steps=0, num_training_steps=config.max_step)

    # set up one adapter
    model=set_up_single_lora(model, f"/share/edc/home/wendaxu/greedy_search_0_sft_lora_256_all_linear_True_fixed_april_29/checkpoint-3057", f"model_0")
    sft_lora_addr_ls = [f"/share/edc/home/wendaxu/greedy_search_{i}_sft_lora_256_all_linear_True_fixed_april_29/checkpoint-3057" for i in range(8)]
    model=set_up_merge_lora(model, sft_lora_addr_ls, 'sft')

    print(model.print_trainable_parameters())

    with tqdm(total=config.max_step) as pbar:
        for step in range(config.max_step):                          
            selected_data=json.load(open(f'online_train_data_{seed}_{mode}/0_rank.json'))
            # selected_data['prompt']=[selected_data['prompt'][0]]
            # selected_data['chosen']=[selected_data['chosen'][0].replace(selected_data['prompt'][0], '')]
            # selected_data['rejected']=[selected_data['rejected'][0].replace(selected_data['prompt'][0], '')]

            print(selected_data)
            train_feedback_data = Dataset.from_dict(selected_data)
            
            
            print("finish annotation")
            # sanity check on the train feedback data
            temp_report_dict={}
            if len(train_feedback_data['prompt'])>0:
                # save weights if it reaches eval step
                # if (step % save_step == 0 or save_step == config.max_step-1):
                #     training_args.save_strategy="steps"
                #     training_args.save_steps=config.step_per_feedback
                #     training_args.output_dir=prefix+f"save_weights_{mode}_seed_{seed}_continue/{step}/model_0"
                # set the different seeds for different lora
                for i in range(8):
                    training_args.seed=seed+i
                    training_args.data_seed=training_args.seed
                    # replace training data with newly updated data
                    dpo_trainer = set_up_trainer(model, i, training_args, train_feedback_data, tokenizer)
                    dpo_trainer.lr_scheduler = copy.deepcopy(lr_scheduler) # copy.deepcopy(lr_scheduler)
                    dpo_trainer.model.set_adapter(f"model_{i}")
                    print("let us see: ", dpo_trainer.model.print_trainable_parameters())
                    # dpo_trainer.model.set_adapter(f"model_0")
                    train_result = dpo_trainer.train()  
                    temp_report_dict[f'model_{i}_loss']=train_result.training_loss 
                    torch.cuda.empty_cache()
                
            wandb.log(temp_report_dict)
            # update lr by ourself not by trainer to avoid sync issues on each lora
            lr_scheduler.step()
            pbar.update(1)

if __name__ == "__main__":
    main()  