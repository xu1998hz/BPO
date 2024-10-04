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
from tqdm import tqdm
import datasets 
from datasets import load_dataset
from typing import TypeVar, Iterable, List
import random
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import math
import shutil
import os
import glob
import subprocess
import time
from peft import LoraConfig, get_peft_model, PeftModel
from openai import OpenAI

class config:
    strategy="dpo"
    step_per_feedback=1
    vali_size=100
    num_epoch=1
    batch_size=1
    acc_steps=16
    lr=5e-5
    num_query=1
    ds_config="config/ds_config_zero3.json"
    model_ref_name="xu1998hz/sft"
    gpu_index_ls=[0, 1] # [0, 1, 2, 3, 4, 5, 6, 7] 
    max_new_token=256
    max_prompt_length=2048

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
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2).to('cuda')).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask.to('cuda')).sum(-1) / loss_mask.to('cuda').sum(-1)
    else:
        return (per_token_logps * loss_mask.to('cuda')).sum(-1)

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
        post = prompt.split('\nPlease summarize the post by given subreddit: ')[0]
        prompt_txt= \
        f"""Which of the following summaries does a better job of summarizing the most \
        important points in the given forum post, without including unimportant or \
        irrelevant details? A good summary is both precise and concise.
        Post:
        {post}
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

def ret_selected_ls(final_ls, rank_stats, mode, step, gpu_index):
    if not os.path.exists('online_logits_data'):
            os.mkdir('online_logits_data')
        
    if not os.path.exists(f'online_logits_data/{step}'):
        os.mkdir(f'online_logits_data/{step}')

    with open(f'online_logits_data/{step}/{gpu_index}_rank_stats.json', 'w') as f:
        json.dump({"data": rank_stats}, f)

    # save a sync file to indicate all data is saved at current process
    with open(f'{step}_{gpu_index}_rank_saved.out', 'w') as f:
        f.write("Rank is saved!")
    
    # wait until all training data is saved!
    while not len(glob.glob(f'{step}_*_rank_saved.out')) == config.num_weights:
        continue 
    
    all_rank_stats={}
    # load all stats data from all processes
    for index, file_name in enumerate(glob.glob(f'online_logits_data/{step}/*')):
        cur_data = json.load(open(file_name))
        all_rank_stats[index] = cur_data["data"]

    # estimate mean reward of each sample
    ucb_ls, lcb_ls = [], []
    for sen_index in range(len(all_rank_stats[0])):
        cur_ucb_ls, cur_lcb_ls = [], []
        for sample_index in range(len(all_rank_stats[0][0])):
            reward_temp_ls = [all_rank_stats[i][sen_index][sample_index] for i in range(len(all_rank_stats))]
            # estimate mean reward
            ele_mean_reward = sum(reward_temp_ls)/len(reward_temp_ls)
            # estimate std of rewards
            ele_std_reward = np.std(reward_temp_ls)
            cur_ucb_ls += [ele_mean_reward+ele_std_reward]
            cur_lcb_ls += [ele_mean_reward-ele_std_reward]
        ucb_ls+=[cur_ucb_ls]
        lcb_ls+=[cur_lcb_ls]

    first_ls, sec_ls = [], []
    # first one will select ucb, second depends on mode
    if mode == "ucb_sec_ucb":
        for sen_ls, cur_sen_ls in zip(ucb_ls, final_ls):
            sort_ls=sorted([[score, index] for index, score in enumerate(sen_ls)])
            # select first sentence based on UCB
            first_index=sort_ls[-1][1]
            # select second sentence based on second UCB
            sec_index=sort_ls[-2][1]
            first_ls+=[cur_sen_ls[first_index]]
            sec_ls+=[cur_sen_ls[sec_index]]
    elif mode == "ucb_mean_ucb":
        for sen_ls, cur_sen_ls in zip(ucb_ls, final_ls):
            sort_ls=sorted([[score, index] for index, score in enumerate(sen_ls)])
            # select first sentence based on UCB
            first_index=sort_ls[-1][1]
            # select second sentence based on mean of UCB
            sec_index=sort_ls[int(len(sort_ls)/2)][1]
            first_ls+=[cur_sen_ls[first_index]]
            sec_ls+=[cur_sen_ls[sec_index]]
    elif mode == "ucb_lcb":
        for sen_ucb_ls, sen_lcb_ls, cur_sen_ls in zip(ucb_ls, lcb_ls, final_ls):
            # seltect first sentence based on UCB
            first_index=sorted([[score, index] for index, score in enumerate(sen_ucb_ls)])[-1][1]
            # select second sentence based on LCB
            sec_index=sorted([[score, index] for index, score in enumerate(sen_lcb_ls)])[-1][1]
            first_ls+=[cur_sen_ls[first_index]]
            sec_ls+=[cur_sen_ls[sec_index]]
    elif mode == "ucb_rand":
        for sen_ls, cur_sen_ls in zip(ucb_ls, final_ls):
            sort_ls=sorted([[score, index] for index, score in enumerate(sen_ls)])
            # seltect first sentence based on UCB
            first_index=sort_ls[-1][1]
            # select second sentence based on random sample other than UCB selection
            sec_index=random.sample(sort_ls[:-1], 1)[0][1]
            first_ls+=[cur_sen_ls[first_index]]
            sec_ls+=[cur_sen_ls[sec_index]]
    else:
        print("Your mode is not supported")
        exit(1)

    return first_ls, sec_ls

def logits_compute(tokenizer, model, subreddit_txts, post_txts, texts, adapter_name):
    inp_ls, label_ls, len_ls = [], [], []
    # batchify texts
    for subreddit_txt, post_txt, text in zip(subreddit_txts, post_txts, texts):
        prompt_len = len(tokenizer.tokenize(f"SUBREDDIT: {subreddit_txt}\nPOST: {post_txt}\nPlease summarize the post by given subreddit: "))
        temp_txt = f"SUBREDDIT: {subreddit_txt}\nPOST: {post_txt}\nPlease summarize the post by given subreddit: {text}"
        inp_ls += [temp_txt]
        if prompt_len > config.max_prompt_length+config.max_new_token:
            label_ls += [[-100]*(config.max_prompt_length+config.max_new_token)]
            len_ls += [config.max_prompt_length+config.max_new_token]
        else:
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

    inputs = tokenizer(inp_ls, return_tensors="pt", padding=True, truncation=True, max_length=config.max_prompt_length, add_special_tokens=False).to(model.device) 
    labels=torch.LongTensor(pad_label_ls)
    # set which lora to use during logits computation
    model.set_adapter(adapter_name)
    all_logits = model(input_ids=inputs["input_ids"], attention_mask=inputs['attention_mask']).logits 
    logits = _get_batch_logps(all_logits, labels, True).to('cpu')    
    return logits

def inference(prompts, tokenizer, model, num_ret, mode, batch_sub, batch_content, step, gpu_index, inference_adapter_name):
    model.eval()
    final_ls = []
    if mode != "rand_rand":
        rank_stats=[]
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer([prompt], return_tensors="pt", padding=True, truncation=True, max_length=config.max_prompt_length, add_special_tokens=False).to(model.device) 
            # make sure the inference adapter name is correct
            model.set_adapter(inference_adapter_name)
            output = model.generate(inputs=inputs.input_ids, max_new_tokens=config.max_new_token, do_sample=True, temperature=1.0, top_k=50, top_p=0.95, num_return_sequences=num_ret)
            # obtained returned sample output 
            responses = tokenizer.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            if mode != "rand_rand":
                # compute rewards for each response
                logits_pi = logits_compute(tokenizer, model, batch_sub*num_ret, batch_content*num_ret, responses, f"model_{gpu_index}").tolist()
                logits_ref = logits_compute(tokenizer, model, batch_sub*num_ret, batch_content*num_ret, responses, "sft").tolist()
                rewards_ls = [pi-ref for pi, ref in zip(logits_pi, logits_ref)]
                rank_ls = [sorted(rewards_ls).index(x) for x in rewards_ls]
                rank_stats += [rank_ls]
            final_ls+=[responses]
    
    first_ls, sec_ls = [], []
    if mode == "rand_rand":
        # perform random selections of two samples
        for ele in final_ls:
            temp_ls = random.sample(ele, 2)
            first_ls += [temp_ls[0]]
            sec_ls += [temp_ls[1]]
    else:
        # perform a waiting function for all processes and return selected data
        first_ls, sec_ls = ret_selected_ls(final_ls, rank_stats, mode, step, gpu_index)
    return first_ls, sec_ls

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

def save_load_training_data(cur_selected_data, step, gpu_index):
    selected_data={'prompt': [], 'chosen': [], 'rejected': []}

    if not os.path.exists('online_train_data'):
        os.mkdir('online_train_data')

    if not os.path.exists(f'online_train_data/{step}'):
        os.mkdir(f'online_train_data/{step}')

    with open(f'online_train_data/{step}/{gpu_index}_rank.json', 'w') as f:
        json.dump(cur_selected_data, f)

    # save a sync file to indicate all data is saved at current process
    with open(f'{step}_{gpu_index}_data_saved.out', 'w') as f:
        f.write("Data is saved!")

    print(f"{gpu_index} finishes annotation")

    # wait until all training data is saved!
    while not len(glob.glob(f'{step}_*_data_saved.out')) == config.num_weights:
        continue 
    # load training data
    for file_name in glob.glob(f'online_train_data/{step}/*'):
        cur_data = json.load(open(file_name))
        selected_data['prompt'] += cur_data['prompt']
        selected_data['chosen'] += cur_data['chosen']
        selected_data['rejected'] += cur_data['rejected']

    train_feedback_data = Dataset.from_dict(selected_data)
    return train_feedback_data

def set_up_single_lora(model, sft_lora_addr, adpater_name):
    torch.cuda.empty_cache()
    model = PeftModel.from_pretrained(
        model,
        sft_lora_addr,
        is_trainable=True,
        adapter_name=adpater_name,
    )
    return model

def set_up_merge_lora(model, sft_lora_addr_ls, weighted_adapter_name, gpu_index):
    model_names=[]
    for i, ele in enumerate(sft_lora_addr_ls):    
        torch.cuda.empty_cache()
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

    # delete unrelated adapters
    for adapter_name in set(model_names)-{f"model_{gpu_index}"}:
        model.delete_adapter(adapter_name)

    # set all sft or weighted_adapter_name parameters to be non-grad
    for name, param in model.named_parameters():
        if weighted_adapter_name in name:
            param.requires_grad = False
    
    model.set_adapter(f"model_{gpu_index}")
    return model

def clean_up_space(step, gpu_index, main_gpu_index, mode, seed):
    # other gpu processes will wait until two folders are removed
    while os.path.exists(f"{step-2}_{gpu_index}_weight_seed_dpo_{mode}_data_seed_{seed}"):
        if gpu_index == main_gpu_index:
            # remove weights       
            if os.path.exists(f"{step-2}_{gpu_index}_weight_seed_dpo_{mode}_data_seed_{seed}"):
                if (step-2) == (int(config.max_step/2)-1) or (step-2) == (int(config.max_step)-2):
                    subprocess.Popen(f'mv {step-2}_{gpu_index}_weight_seed_dpo_{mode}_data_seed_{seed} save_weights', shell=True)
                else:
                    subprocess.Popen(f'rm -rf {step-2}_{gpu_index}_weight_seed_dpo_{mode}_data_seed_{seed}', shell=True)
            
            # remove out files       
            if os.path.exists(f'{step-2}_*_data_saved.out'):
                subprocess.Popen(f'rm -rf {step-2}_*_data_saved.out', shell=True)
            
            # remove all data saving out files
            if os.path.exists(f'{step-2}_*_rank_saved.out'):
                subprocess.Popen(f'rm -rf {step-2}_*_rank_saved.out', shell=True)
    print("Remove all files from two steps ago!") 

@click.command()
@click.option('-mode', type=str, default="rand_rand")
@click.option('-seed', type=int, default=0) # 42
@click.option('-start_index', type=int, default=10000)
@click.option('-end_index', type=int, default=12000)
@click.option('-num_ret', type=int, default=6)
@click.option('-gpu_index', type=int, default=0)
@click.option('-main_gpu_index', type=int, help="it can be GPU index other than 0", default=0)
@click.option('-api_key', type=str, default=None)
@click.option('-api_source', type=str, default="openai")
@click.option('-prefix', type=str, default="/mnt/data6/")
# @click.option('-sft_lora_addr', type=str, default="xu1998hz/0_sft_lora")
def main(mode, seed, start_index, end_index, num_ret, gpu_index, main_gpu_index, api_key, api_source, prefix):
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
    config.num_weights=len(config.gpu_index_ls)
    config.model_name=f"xu1998hz/{gpu_index}_sft_lora"
    config.predix=prefix
    config.max_step=int(config.num_epoch*config.train_size/(config.batch_size*config.acc_steps))
    config.model_seed_index=seed+int(gpu_index)
    config.data_seed_index=config.model_seed_index

    if not os.path.exists('save_weights'):
        os.mkdir('save_weights')

    # load in all model parameters
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained('google/gemma-2b', torch_dtype=torch.bfloat16, device_map="cpu").to("cuda")
    tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b')

    # define my own lr scheduler
    optimizer = transformers.AdamW(model.parameters(), lr=config.lr)
    lr_scheduler = transformers.get_scheduler("cosine", optimizer, num_warmup_steps=0, num_training_steps=config.max_step)

    wandb.init(project='DPO_Master', name=f"{mode}_{gpu_index}_{seed}_april_22", config=\
    {
        "strategy": mode,
        "seed": seed,
        "epoch": config.num_epoch,
        "train batch size": config.batch_size * config.acc_steps,
        "lr": config.lr,
        "model_name": config.model_name,
        "gpu_index": gpu_index,
        "model_seed_index": config.model_seed_index,
        "data_seed_index": config.data_seed_index,
        "setting": mode
    })

    # load in dataset
    data = load_dataset('webis/tldr-17', trust_remote_code=True)
    sampling_prompt_dataset=[data['train'][i] for i in range(start_index, end_index, 1)]

    # initialize trainer
    training_args = TrainingArguments(
        report_to="wandb", # don't report to wandb through trainer
        per_device_train_batch_size=config.batch_size,
        max_steps=config.step_per_feedback,
        gradient_accumulation_steps=config.acc_steps, # simulate larger batch sizes
        output_dir=None,
        save_strategy="steps",
        save_steps=config.step_per_feedback,
        evaluation_strategy="no", 
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=0.01, 
        weight_decay=0,
        warmup_ratio=0,
        seed=config.model_seed_index,
        data_seed=config.data_seed_index,
        remove_unused_columns=False
        # deepspeed=config.ds_config,
    )

    for step, (prompts, batch_sub, batch_content) in enumerate(prompt_batchify(sampling_prompt_dataset, config.acc_steps*config.step_per_feedback)):
        # specify saved location
        training_args.output_dir=f"{step}_{gpu_index}_weight_seed_dpo_{mode}_data_seed_{seed}"
        if step == 0:
            inference_adapter_name="sft"
            # we need to setup our gpu specific adapter and this won't be changed (model_{gpu_index})
            model=set_up_single_lora(model, f"xu1998hz/{gpu_index}_sft_lora", f"model_{gpu_index}")
            # load sft lora weights for inference
            sft_lora_addr_ls = [f"xu1998hz/{i}_sft_lora" for i in set(config.gpu_index_ls)]
            model=set_up_merge_lora(model, sft_lora_addr_ls, inference_adapter_name, gpu_index)
            model.print_trainable_parameters()
        else:
            # replace inference model with last model  adapter among gpus
            sft_lora_addr_ls = [f"{step-1}_{index}_weight_seed_dpo_{mode}_data_seed_{seed}/checkpoint-{config.step_per_feedback}/model_{index}" for index in config.gpu_index_ls]
            # This process will hang until we are able to load all lora weights
            success = False
            while not success:
                try:
                    inference_adapter_name="dpo_inference"
                    model=set_up_merge_lora(model, sft_lora_addr_ls, "dpo_inference", gpu_index)
                    model.print_trainable_parameters()
                    success = True
                except Exception as e:
                    continue
        
        # allocate data for only current GPU to inference
        cur_gpu_data=list(split_batch(prompts, len(config.gpu_index_ls)))[gpu_index]
        # make sure you are inferencing on sft lora, use dpo_inference
        first_ls, sec_ls = inference(cur_gpu_data, tokenizer, model, num_ret, mode, batch_sub, batch_content, step, gpu_index, inference_adapter_name)
        
        # begin annotations from desinated LLM
        cur_selected_data = ret_train_data(prompts, first_ls, sec_ls, api_source)
        train_feedback_data = save_load_training_data(cur_selected_data, step, gpu_index)
       
        # sanity check on the train feedback data
        if len(train_feedback_data['prompt'])>0:
            # replace training data with newly updated data
            dpo_trainer = DPOTrainer(
                model=model,
                model_adapter_name=f"model_{gpu_index}",
                ref_adapter_name="sft",
                args=training_args,
                beta=0.5,
                train_dataset=train_feedback_data,
                tokenizer=tokenizer,
                max_length=config.max_prompt_length+config.max_new_token,
                max_prompt_length=config.max_prompt_length
                # callbacks=[SaveEvalOutputsCallback(raw_eval_dataset, model, tokenizer, output_dir=f'sample_output/{mode}', num_samples=20)]
            )
            dpo_trainer.lr_scheduler = lr_scheduler
            # dpo_trainer.train_dataset = train_feedback_data
            num_data = len(train_feedback_data['prompt'])
            print(f"Training size {num_data} is loaded!")
            dpo_trainer.train()
            print("model is saved!")

        # clean up files from two steps ago
        clean_up_space(step, gpu_index, main_gpu_index, mode, seed)

if __name__ == "__main__":
    main()  