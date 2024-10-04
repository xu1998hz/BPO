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
from typing import TypeVar, Iterable, Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import random
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os
import time
from peft import PeftModel
from openai import OpenAI
from transformers.trainer_callback import PrinterCallback
import math
import torch.nn.functional as F
import sys

sys.setrecursionlimit(10000)
current_limit = sys.getrecursionlimit()
print("Current recursion limit:", current_limit)

class config:
    step_per_feedback=1
    vali_size=100
    num_epoch=1
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
        generation_config={"temperature": 1.0, "max_output_tokens": 1024},
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
    loss_mask = (labels != -100).to('cuda')

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0
    torch.cuda.empty_cache()
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.to('cuda').unsqueeze(2)).squeeze(2)
    
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

def ret_selected_ls(final_ls, all_rank_stats, mode, prompts):
    # estimate mean reward of each sample
    ucb_ls, lcb_ls = [], []
    for sen_index in range(len(all_rank_stats[0])):
        cur_ucb_ls, cur_lcb_ls = [], []
        for sample_index in range(len(all_rank_stats[0][sen_index])):
            reward_temp_ls = []
            for i in range(len(all_rank_stats)):
                reward_temp_ls+=[all_rank_stats[i][sen_index][sample_index]]
            # estimate mean reward
            ele_mean_reward = sum(reward_temp_ls)/len(reward_temp_ls)
            # estimate std of rewards
            ele_std_reward = np.std(reward_temp_ls)
            cur_ucb_ls += [ele_mean_reward+ele_std_reward]
            cur_lcb_ls += [ele_mean_reward-ele_std_reward]
        ucb_ls+=[cur_ucb_ls]
        lcb_ls+=[cur_lcb_ls]

    final_prompts, first_ls, sec_ls = [], [], []
    # first one will select ucb, second depends on mode
    if mode == "ucb_sec_ucb": # first and sec are definately different
        for sen_ls, cur_sen_ls, prompt in zip(ucb_ls, final_ls, prompts):
            if len(cur_sen_ls)>1:
                sort_ls=sorted([[score, index] for index, score in enumerate(sen_ls)])
                # select first sentence based on UCB
                first_index=sort_ls[-1][1]
                # select second sentence based on second UCB
                sec_index=sort_ls[-2][1]
                first_ls+=[cur_sen_ls[first_index]]
                sec_ls+=[cur_sen_ls[sec_index]]
                final_prompts+=[prompt]
    elif mode == "ucb_mean_ucb": # first and sec are definately different
        for sen_ls, cur_sen_ls, prompt in zip(ucb_ls, final_ls, prompts):
            if len(cur_sen_ls)>1:
                sort_ls=sorted([[score, index] for index, score in enumerate(sen_ls)])
                # select first sentence based on UCB
                first_index=sort_ls[-1][1]
                # select second sentence based on mean of UCB
                sec_index=sort_ls[int(len(sort_ls)/2)][1]
                first_ls+=[cur_sen_ls[first_index]]
                sec_ls+=[cur_sen_ls[sec_index]]
                final_prompts+=[prompt]
    elif mode == "ucb_lcb": 
        for sen_ucb_ls, sen_lcb_ls, cur_sen_ls, prompt in zip(ucb_ls, lcb_ls, final_ls, prompts):
            if len(cur_sen_ls)>1:
                # seltect first sentence based on UCB
                first_index=sorted([[score, index] for index, score in enumerate(sen_ucb_ls)])[-1][1]
                # select second sentence based on LCB
                sec_index=sorted([[score, index] for index, score in enumerate(sen_lcb_ls)])[-1][1]
                first_ls+=[cur_sen_ls[first_index]]
                # manually make sure second index is different from first
                if first_index == sec_index:
                    sec_index=random.sample(list(set(list(range(len(cur_sen_ls))))-{first_index}), 1)[0]    
                sec_ls+=[cur_sen_ls[sec_index]]
                final_prompts+=[prompt]
    elif mode == "ucb_rand": # first and sec are definately different
        for sen_ls, cur_sen_ls, prompt in zip(ucb_ls, final_ls, prompts):
            if len(cur_sen_ls)>1:
                sort_ls=sorted([[score, index] for index, score in enumerate(sen_ls)])
                # select first sentence based on UCB
                first_index=sort_ls[-1][1]
                # select second sentence based on random sample other than UCB selection
                sec_index=random.sample(sort_ls[:-1], 1)[0][1]
                first_ls+=[cur_sen_ls[first_index]]
                sec_ls+=[cur_sen_ls[sec_index]]
                final_prompts+=[prompt]
    else:
        print("Your mode is not supported")
        exit(1)

    return final_prompts, first_ls, sec_ls

def logits_compute(tokenizer, model, prompts, outputs, adapter_name):
    tokd_all = tokenizer(outputs, return_tensors='pt', max_length=config.max_new_token+config.max_prompt_length, truncation=True, padding=True) # "max_length"
    tokd_gen = tokenizer(prompts, return_tensors='pt', max_length=config.max_new_token, truncation=True, padding=True) # max_length"

    labels = tokd_all["input_ids"].clone().detach()
    # depends on the padding strategy! You need to verify if model is left padded
    labels[:, :labels.shape[1] - tokd_gen["input_ids"].shape[1] + 1] = -100
    torch.cuda.empty_cache()
    # set which lora to use during logits computation
    model.set_adapter(adapter_name)
    all_logits = model(input_ids=tokd_all["input_ids"].to(model.device), attention_mask=tokd_all['attention_mask'].to(model.device)).logits 
    logits = _get_batch_logps(all_logits, labels, True).to('cpu')
    return logits

def inference(prompts, tokenizer, model, num_ret, mode, num_lora, inference_adapter_name):
    model.eval()
    final_ls = []
    if mode != "rand_rand":
        all_rank_stats={i: [] for i in range(num_lora)}
    with torch.no_grad():
        for prompt in prompts:
            # print("Start generation")
            inputs = tokenizer([prompt], return_tensors="pt", padding=True, truncation=True, max_length=config.max_prompt_length, add_special_tokens=False).to(model.device) 
            # make sure the inference adapter name is correct
            model.set_adapter(inference_adapter_name)
            # responses, summary_ls = [], []
            # for _ in range(math.ceil(num_ret/config.gpu_takes_n)):
            output = model.generate(inputs=inputs.input_ids, max_new_tokens=config.max_new_token, do_sample=True, temperature=1.0, top_k=0, top_p=0.95, num_return_sequences=num_ret)
            # obtained returned sample output, responses will include prompt
            responses = list(set(tokenizer.batch_decode(output, skip_special_tokens=True)))
            summary_ls = list(set(tokenizer.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)))
            # print("End generation")
            if mode != "rand_rand":
                # compute rewards for each response
                for model_index in range(0, num_lora):
                    torch.cuda.empty_cache()
                    logits_pi = []
                    for batch_res in batchify(responses, config.gpu_takes_n):
                        logits_pi += logits_compute(tokenizer, model, [prompt]*len(batch_res), batch_res, f"model_{model_index}").tolist()
                    torch.cuda.empty_cache()
                    logits_ref = []
                    for batch_res in batchify(responses, config.gpu_takes_n):
                        logits_ref += logits_compute(tokenizer, model, [prompt]*len(batch_res), batch_res, "sft").tolist()
                    rewards_ls = [pi-ref for pi, ref in zip(logits_pi, logits_ref)]
                    rank_ls = [sorted(rewards_ls).index(x) for x in rewards_ls]
                    all_rank_stats[model_index] += [rank_ls]
            final_ls+=[summary_ls]
    
    first_ls, sec_ls, final_prompt_ls = [], [], []
    if mode == "rand_rand":
        # perform random selections of two samples
        for ele, prompt in zip(final_ls, prompts):
            if len(ele) > 1:
                temp_ls = random.sample(ele, 2)
                first_ls += [temp_ls[0]]
                sec_ls += [temp_ls[1]]
                final_prompt_ls += [prompt]
    else:
        # perform a waiting function for all processes and return selected data
        final_prompt_ls, first_ls, sec_ls = ret_selected_ls(final_ls, all_rank_stats, mode, prompts)
    return final_prompt_ls, first_ls, sec_ls

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

@click.command()
@click.option('-mode', type=str, default="rand_rand")
@click.option('-seed', type=int, default=0) # 42
@click.option('-start_index', type=int, default=10000)
@click.option('-end_index', type=int, default=12000)
@click.option('-save_step', type=int, default=200)
@click.option('-num_ret', type=int, default=6)
@click.option('-num_lora', type=int, default=8)
@click.option('-api_key', type=str, default=None)
@click.option('-api_source', type=str, default="openai")
@click.option('-prefix', type=str, default="/mnt/data6/")
@click.option('-gpu_takes_n', type=int, default=12)
# @click.option('-sft_lora_addr', type=str, default="xu1998hz/0_sft_lora")
def main(mode, seed, start_index, end_index, num_ret, num_lora, api_key, api_source, prefix, save_step, gpu_takes_n):
    # load in API specific parameters
    if api_source=="google":
        genai.configure(api_key=api_key)
    else:
        config.client = OpenAI()
        config.system_prompt="You are a text generation evaluater."
        config.model_type="gpt-3.5-turbo-0125"
        config.n=1

    # define all training specific data
    config.train_size=end_index-start_index
    config.predix=prefix
    config.gpu_takes_n=gpu_takes_n
    config.max_step=int(config.num_epoch*config.train_size/(config.batch_size*config.acc_steps))

    if not os.path.exists('save_weights'):
        os.mkdir('save_weights')

    # load in all model parameters
    model = AutoModelForCausalLM.from_pretrained('google/gemma-2b', torch_dtype=torch.bfloat16, device_map="cpu", attn_implementation="flash_attention_2").to("cuda")
    # You must not use the caches of kv, otherwise you won't learn
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b')

    # define my own lr scheduler
    optimizer = transformers.AdamW(model.parameters(), lr=config.lr)
    lr_scheduler = transformers.get_scheduler("cosine", optimizer, num_warmup_steps=0, num_training_steps=config.max_step)

    wandb.init(project='DPO_Master', name=f"{mode}_{seed}_finalized_april_30_save_step_{save_step}", config=\
    {
        "seed": seed,
        "save_step": save_step,
        "epoch": config.num_epoch,
        "train batch size": config.batch_size * config.acc_steps,
        "lr": config.lr,
        "setting": mode
    })

    # load in dataset
    with open('sft_data/sampling.jsonl') as f:
        sampling_prompt_dataset = [json.loads(line) for line in f]
    sampling_prompt_dataset = sampling_prompt_dataset[start_index:end_index]

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
        logging_strategy="no",
        weight_decay=0,
        warmup_ratio=0,
        seed=None, 
        data_seed=None,
        remove_unused_columns=False,
        bf16=True
        # deepspeed=config.ds_config,
    )
    
    global_step = 0
    # setup model first
    inference_adapter_name="sft"
    # set up one adapter
    model=set_up_single_lora(model, f"xu1998hz/0_sft_lora_256", f"model_0")
    # load sft lora weights for inference
    sft_lora_addr_ls = [f"xu1998hz/{i}_sft_lora_256" for i in range(num_lora)]
    model=set_up_merge_lora(model, sft_lora_addr_ls, inference_adapter_name)

    for _ in range(2):
        with tqdm(total=config.max_step) as pbar:
            for step, prompts in enumerate(prompt_batchify(sampling_prompt_dataset, config.acc_steps*config.step_per_feedback)):            
                # specify saved location
                if not os.path.isfile(f'online_train_data_{seed}_{mode}/{step}_rank.json'):
                    # replace inference model with last model adapter among gpus
                    model=merge_lora_for_inference(model, num_lora)
                    # make sure you are inferencing on sft lora, use dpo_inference
                    print("start inference")
                    prompts, first_ls, sec_ls = inference(prompts, tokenizer, model, num_ret, mode, num_lora, inference_adapter_name)
                    print("finish inference")

                    # begin annotations from desinated LLM
                    selected_data = ret_train_data(prompts, first_ls, sec_ls, api_source)
                    # save feedback data from annotations
                    if not os.path.exists(f'online_train_data_{seed}_{mode}'):
                        os.mkdir(f'online_train_data_{seed}_{mode}')
                    with open(f'online_train_data_{seed}_{mode}/{step}_rank.json', 'w') as f:
                        json.dump(selected_data, f)
                else:
                    selected_data = json.load(open(f'online_train_data_{seed}_{mode}/{step}_rank.json'))
                
                train_feedback_data = Dataset.from_dict(selected_data)
                print("finish annotation")
                # sanity check on the train feedback data
                temp_report_dict={}
                if len(train_feedback_data['prompt'])>0:
                    for model_index in range(0, num_lora):
                        # save weights if it reaches eval step
                        if (step % save_step == 0 or save_step == config.max_step-1) and model_index==num_lora-1:
                            training_args.save_strategy="steps"
                            training_args.save_steps=config.step_per_feedback
                            training_args.output_dir=prefix+f"save_weights_{mode}_seed_{seed}/{global_step}/model_{num_lora-1}"
                            training_args.save_strategy="no"
                        # set the different seeds for different lora
                        training_args.seed=seed+model_index
                        training_args.data_seed=training_args.seed
                        # replace training data with newly updated data
                        model.set_adapter(f"model_{model_index}")
                        
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
                        dpo_trainer.lr_scheduler = copy.deepcopy(lr_scheduler)
                        dpo_trainer.remove_callback(PrinterCallback)
                        train_result = dpo_trainer.train()  
                        temp_report_dict[f'model_{model_index}_loss']=train_result.training_loss 
                        torch.cuda.empty_cache()
                    
                    lr_scheduler.step()
                    wandb.log(temp_report_dict)
                    print("Finish training")
                pbar.update(1)
                global_step+=1

if __name__ == "__main__":
    main()