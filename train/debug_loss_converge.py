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
    num_epoch=2
    batch_size=1
    acc_steps=16
    lr=1e-6
    num_query=1
    ds_config="config/ds_config_zero3.json"
    model_ref_name="xu1998hz/sft"
    max_new_token=128
    max_prompt_length=512
    beta=0.1

def call_gemini(prompt_txt):
    model = genai.GenerativeModel(model_name="gemini-pro")
    completion = model.generate_content(
        prompt_txt,
        generation_config={"temperature": 0.0, "max_output_tokens": 1024},
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }

    )
    try:
        return completion.text
    except:
        return "[BLOCKED]"

datasets.disable_progress_bar()
T = TypeVar('T')

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
    torch.cuda.empty_cache()
    per_token_logps = torch.gather(logits.log_softmax(-1).to('cpu'), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    
    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)

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

def get_first_batch_data(data: Iterable[T]) -> Iterable[List[T]]:
    batch, batch_subreddit, batch_content = [], [], []
    for item in data:
        batch.append(f"SUBREDDIT: {item['subreddit']}\nPOST: {item['content']}\nPlease summarize the post by given subreddit: ")
        batch_subreddit.append(item['subreddit'])
        batch_content.append(item['content'])
    return batch, batch_subreddit, batch_content

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

def ret_train_data(prompts, first_ls, sec_ls, api_source):
    p_ls, chosen_ls, rejected_ls = [], [], []
    for prompt, first, second in zip(prompts, first_ls, sec_ls):
        prompt_txt= \
        f"""Which of the following summaries does a better job of summarizing the most \
        important points in the given forum post, without including unimportant or \
        irrelevant details? A good summary is both precise and concise.
        {prompt}
        Summary A:
        {first}
        Summary B:
        {second}
        FIRST provide a one-sentence comparison of the two summaries, explaining which \
        you prefer and why. SECOND, on a new line, state only "A" or "B" to indicate your \
        choice. Your response should use the format:
        Comparison: <one-sentence comparison and explanation>
        Preferred: <"A" or "B">"""
        
        processs=True
        answer="None"
        start_time = time.time()
        end_time = start_time
        # set the maximum waiting time to be 3 seconds
        while processs and end_time-start_time < 3:
            try:
                if api_source == "google":
                    answer = call_gemini(prompt_txt)
                else:
                    answer = completions_with_backoff_openai(config.client, config.system_prompt, prompt_txt, config.model_type, config.n)
                processs=False
            except:
                print("An error occurred! Rerunning")
            end_time = time.time()
        
        if 'Preferred:' in answer:
            if answer.split('Preferred:')[1].strip() == 'A':
                p_ls+=[prompt]
                chosen_ls+=[first]
                rejected_ls+=[second]
            elif answer.split('Preferred:')[1].strip() == 'B':
                p_ls+=[prompt]
                chosen_ls+=[second]
                rejected_ls+=[first]
            else:
                print(answer)

    selected_data={}
    selected_data['prompt'] = p_ls
    selected_data['chosen'] = chosen_ls
    selected_data['rejected'] = rejected_ls

    return selected_data

def split_batch(batch, num_gpus):
    k, m = divmod(len(batch), num_gpus)
    return (batch[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(num_gpus))

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
        temperature=0.0,
        max_tokens=1024,
        top_p=1,
        n=n,
    ).choices[0].message.content
    return response

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

def merge_lora_for_inference(model, num_lora):
    model_names=[f"model_{i}" for i in range(num_lora)]
    # perform model averaging on lora weights
    model.add_weighted_adapter(
        adapters=model_names,
        weights=[1/len(model_names)]*len(model_names),
        adapter_name="dpo_inference",
        combination_type="linear"
    )
    model.set_adapter("dpo_inference")

    # set all sft or weighted_adapter_name parameters to be non-grad
    for name, param in model.named_parameters():
        if "dpo_inference" in name:
            param.requires_grad = False
    return model

def set_up_trainer(model, model_index, training_args, train_feedback_data, tokenizer):
    dpo_trainer = DPOTrainer(
        model=model,
        model_adapter_name=f"model_{model_index}",
        ref_adapter_name="sft",
        args=training_args,
        beta=config.beta,
        train_dataset=train_feedback_data,
        tokenizer=tokenizer,
        max_length=config.max_prompt_length+config.max_new_token,
        max_prompt_length=config.max_prompt_length
        # callbacks=[SaveEvalOutputsCallback(raw_eval_dataset, model, tokenizer, output_dir=f'sample_output/{mode}', num_samples=20)]
    )
    # dpo_trainer.remove_callback(PrinterCallback)
    return dpo_trainer

@click.command()
@click.option('-mode', type=str, default="rand_rand")
@click.option('-seed', type=int, default=0) # 42
@click.option('-max_step', type=int, default=125)
@click.option('-num_lora', type=int, default=8)
@click.option('-prefix', type=str, default="/mnt/data6/")
@click.option('-openai', type=bool)
# @click.option('-sft_lora_addr', type=str, default="xu1998hz/0_sft_lora")
def main(mode, seed, num_lora, prefix, openai, max_step):
    # define all training specific data
    config.prefix=prefix
    config.max_step=max_step

    if not os.path.exists('save_weights'):
        os.mkdir('save_weights')

    # load in all model parameters
    model = AutoModelForCausalLM.from_pretrained('google/gemma-2b', torch_dtype=torch.bfloat16, device_map="cpu").to("cuda") # , attn_implementation="flash_attention_2"
    # You must not use the caches of kv, otherwise you won't learn
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b')

    wandb.init(project='DPO_Master', name=f"two_step_{mode}_{seed}_may_2", config=\
    {
        "seed": seed,
        "epoch": config.num_epoch,
        "train batch size": config.batch_size * config.acc_steps,
        "lr": config.lr,
        "setting": mode
    })

    # initialize trainer
    training_args = TrainingArguments(
        # disable_tqdm=True,
        report_to="wandb", # don't report to wandb through trainer
        per_device_train_batch_size=config.batch_size,
        max_steps=config.max_step*config.num_epoch,
        gradient_accumulation_steps=config.acc_steps, # simulate larger batch sizes
        output_dir=f"{config.prefix}/dpo_two_step_{mode}_{seed}", # debug_dpo_openai_{openai}_full
        save_strategy="steps",
        save_steps=config.max_step,
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=1,
        weight_decay=0,
        warmup_ratio=0,
        seed=None, 
        data_seed=None,
        remove_unused_columns=False,
        bf16=True
        # deepspeed=config.ds_config,
    )
    inference_adapter_name="sft"
    # set up one adapter
    model=set_up_single_lora(model, f"xu1998hz/0_sft_lora_256", f"model_0")
    # load sft lora weights for inference
    sft_lora_addr_ls = [f"xu1998hz/{i}_sft_lora_256" for i in range(num_lora)]
    model=set_up_merge_lora(model, sft_lora_addr_ls, inference_adapter_name)
    print("model is loaded!")
    
    final_data = {'prompt': [], 'chosen': [], 'rejected': []}

    if openai:
        train_feedback_data = load_dataset('CarperAI/openai_summarize_comparisons')['train']
        # ensure fair comparisions using OpenAI data
        for index, cur_data in enumerate(train_feedback_data):
            if index == 2000:
                break
            final_data['prompt'] += [cur_data['prompt']]
            final_data['chosen'] += [cur_data['chosen']]
            final_data['rejected'] += [cur_data['rejected']]
    else:
        for i in range(config.max_step):
            cur_data = json.load(open(f'online_train_data_{seed}_{mode}/{i}_rank.json'))
            final_data['prompt']+=cur_data['prompt']
            final_data['chosen']+=cur_data['chosen']
            final_data['rejected']+=cur_data['rejected']

        train_feedback_data = Dataset.from_dict(final_data)
    print("data is loaded!")
    training_args.seed=seed
    training_args.data_seed=training_args.seed
    dpo_trainer = DPOTrainer(
        model=model, # model,
        model_adapter_name=f"model_0",
        ref_adapter_name=f"sft",
        args=training_args,
        beta=config.beta,
        train_dataset=train_feedback_data,
        tokenizer=tokenizer,
        max_length=config.max_prompt_length+config.max_new_token,
        max_prompt_length=config.max_prompt_length
        # callbacks=[SaveEvalOutputsCallback(raw_eval_dataset, model, tokenizer, output_dir=f'sample_output/{mode}', num_samples=20)]
    )
    dpo_trainer.model.set_adapter(f"model_0")
    train_result = dpo_trainer.train() 

if __name__ == "__main__":
    main()  