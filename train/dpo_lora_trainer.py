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
import torch.nn as nn

sys.setrecursionlimit(100000)
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
            output = model.generate(inputs=inputs.input_ids, max_new_tokens=config.max_new_token, do_sample=True, temperature=1.0, top_k=0, top_p=0.95, num_return_sequences=config.gpu_takes_n)
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

# def set_up_trainer(model, model_index, training_args, train_feedback_data, tokenizer):
#     dpo_trainer = DPOTrainer(
#         model=model,
#         model_adapter_name=f"model_{model_index}",
#         ref_adapter_name="sft",
#         args=training_args,
#         beta=config.beta,
#         train_dataset=train_feedback_data,
#         tokenizer=tokenizer,
#         max_length=config.max_prompt_length+config.max_new_token,
#         max_prompt_length=config.max_prompt_length
#         # callbacks=[SaveEvalOutputsCallback(raw_eval_dataset, model, tokenizer, output_dir=f'sample_output/{mode}', num_samples=20)]
#     )
#     dpo_trainer.remove_callback(PrinterCallback)
#     return dpo_trainer

# def dpo_loss(
#     self,
#     policy_chosen_logps: torch.FloatTensor,
#     policy_rejected_logps: torch.FloatTensor,
#     reference_chosen_logps: torch.FloatTensor,
#     reference_rejected_logps: torch.FloatTensor,
# ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
#     """Compute the DPO loss for a batch of policy and reference model log probabilities.

#     Args:
#         policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
#         policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
#         reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
#         reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

#     Returns:
#         A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
#         The losses tensor contains the DPO loss for each example in the batch.
#         The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
#     """
#     pi_logratios = policy_chosen_logps - policy_rejected_logps
#     if self.reference_free:
#         ref_logratios = torch.tensor([0], dtype=pi_logratios.dtype, device=pi_logratios.device)
#     else:
#         ref_logratios = reference_chosen_logps - reference_rejected_logps

#     pi_logratios = pi_logratios.to(self.accelerator.device)
#     ref_logratios = ref_logratios.to(self.accelerator.device)
#     logits = pi_logratios - ref_logratios

#     # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
#     # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
#     # calculates a conservative DPO loss.
#     if self.loss_type == "sigmoid":
#         losses = (
#             -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
#             - F.logsigmoid(-self.beta * logits) * self.label_smoothing
#         )
#     elif self.loss_type == "hinge":
#         losses = torch.relu(1 - self.beta * logits)
#     elif self.loss_type == "ipo":
#         # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
#         losses = (logits - 1 / (2 * self.beta)) ** 2
#     elif self.loss_type == "kto_pair":
#         # eqn (7) of the HALOs paper
#         chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
#         rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

#         chosen_logratios = policy_chosen_logps - reference_chosen_logps
#         rejected_logratios = policy_rejected_logps - reference_rejected_logps
#         # As described in the KTO report, the KL term for chosen (rejected) is estimated using the rejected (chosen) half.
#         losses = torch.cat(
#             (
#                 1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
#                 1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
#             ),
#             0,
#         )
#     elif self.loss_type == "bco_pair":
#         chosen_logratios = policy_chosen_logps - reference_chosen_logps
#         rejected_logratios = policy_rejected_logps - reference_rejected_logps

#         chosen_rewards = self.beta * chosen_logratios
#         rejected_rewards = self.beta * rejected_logratios
#         rewards = torch.cat((chosen_rewards, rejected_rewards), 0).mean().detach()
#         self.running.update(rewards)
#         delta = self.running.mean

#         losses = -F.logsigmoid((self.beta * chosen_logratios) - delta) - F.logsigmoid(
#             -(self.beta * rejected_logratios - delta)
#         )
#     else:
#         raise ValueError(
#             f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair', 'bco_pair']"
#         )

#     chosen_rewards = (
#         self.beta
#         * (
#             policy_chosen_logps.to(self.accelerator.device) - reference_chosen_logps.to(self.accelerator.device)
#         ).detach()
#     )
#     rejected_rewards = (
#         self.beta
#         * (
#             policy_rejected_logps.to(self.accelerator.device)
#             - reference_rejected_logps.to(self.accelerator.device)
#         ).detach()
#     )

#     return losses, chosen_rewards, rejected_rewards

# def get_train_dataloader(self) -> DataLoader:
#     """
#     Returns the training [`~torch.utils.data.DataLoader`].

#     Subclass of transformers.src.transformers.trainer.get_train_dataloader to precompute `ref_log_probs`.
#     """

#     if self.precompute_ref_log_probs and not self._precomputed_train_ref_log_probs:
#         dataloader_params = {
#             "batch_size": self.args.per_device_train_batch_size,
#             "collate_fn": self.data_collator,
#             "num_workers": self.args.dataloader_num_workers,
#             "pin_memory": self.args.dataloader_pin_memory,
#             "shuffle": False,
#         }

#         # prepare dataloader
#         data_loader = self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))

#         reference_chosen_logps = []
#         reference_rejected_logps = []
#         for padded_batch in tqdm(iterable=data_loader, desc="Train dataset reference log probs"):
#             reference_chosen_logp, reference_rejected_logp = self.compute_reference_log_probs(padded_batch)
#             reference_chosen_logp, reference_rejected_logp = self.accelerator.gather_for_metrics(
#                 (reference_chosen_logp, reference_rejected_logp)
#             )
#             reference_chosen_logps.append(reference_chosen_logp.cpu())
#             reference_rejected_logps.append(reference_rejected_logp.cpu())

#         all_reference_chosen_logps = torch.cat(reference_chosen_logps).float().numpy()
#         all_reference_rejected_logps = torch.cat(reference_rejected_logps).float().numpy()

#         self.train_dataset = self.train_dataset.add_column(
#             name="reference_chosen_logps", column=all_reference_chosen_logps
#         )
#         self.train_dataset = self.train_dataset.add_column(
#             name="reference_rejected_logps", column=all_reference_rejected_logps
#         )

#         self._precomputed_train_ref_log_probs = True

#     return super().get_train_dataloader()

# def concatenated_forward(
#     self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
# ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
#     """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

#     We do this to avoid doing two forward passes, because it's faster for FSDP.
#     """
#     concatenated_batch = self.concatenated_inputs(
#         batch,
#         is_encoder_decoder=self.is_encoder_decoder,
#         label_pad_token_id=self.label_pad_token_id,
#         padding_value=self.padding_value,
#         device=self.accelerator.device,
#     )
#     len_chosen = batch["chosen_labels"].shape[0]

#     model_kwargs = (
#         {
#             "labels": concatenated_batch["concatenated_labels"],
#             "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
#         }
#         if self.is_encoder_decoder
#         else {}
#     )
#     all_logits = model(
#         concatenated_batch["concatenated_input_ids"],
#         attention_mask=concatenated_batch["concatenated_attention_mask"],
#         use_cache=False,
#         **model_kwargs,
#     ).logits

#     all_logps = self.get_batch_logps(
#         all_logits,
#         concatenated_batch["concatenated_labels"],
#         average_log_prob=self.loss_type == "ipo",
#         is_encoder_decoder=self.is_encoder_decoder,
#         label_pad_token_id=self.label_pad_token_id,
#     )

#     chosen_logps = all_logps[:len_chosen]
#     rejected_logps = all_logps[len_chosen:]

#     chosen_logits = all_logits[:len_chosen]
#     rejected_logits = all_logits[len_chosen:]

#     return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

# def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
#     if tensor.size(dim) >= length:
#         return tensor
#     else:
#         pad_size = list(tensor.shape)
#         pad_size[dim] = length - tensor.size(dim)
#         return torch.cat(
#             [
#                 tensor,
#                 pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
#             ],
#             dim=dim,
#         )

# def concatenated_inputs(
#     batch: Dict[str, Union[List, torch.LongTensor]],
#     is_encoder_decoder: bool = False,
#     label_pad_token_id: int = -100,
#     padding_value: int = 0,
#     device: Optional[torch.device] = None,
# ) -> Dict[str, torch.LongTensor]:
#     """Concatenate the chosen and rejected inputs into a single tensor.

#     Args:
#         batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
#         is_encoder_decoder: Whether the model is an encoder-decoder model.
#         label_pad_token_id: The label pad token id.
#         padding_value: The padding value to use for the concatenated inputs_ids.
#         device: The device for the concatenated inputs.

#     Returns:
#         A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
#     """
#     concatenated_batch = {}

#     if is_encoder_decoder:
#         max_length = max(batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1])
#     else:
#         max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

#     for k in batch:
#         if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
#             if "labels" in k or is_encoder_decoder:
#                 pad_value = label_pad_token_id
#             elif k.endswith("_input_ids"):
#                 pad_value = padding_value
#             elif k.endswith("_attention_mask"):
#                 pad_value = 0
#             concatenated_key = k.replace("chosen", "concatenated")
#             concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
#     for k in batch:
#         if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
#             if "labels" in k or is_encoder_decoder:
#                 pad_value = label_pad_token_id
#             elif k.endswith("_input_ids"):
#                 pad_value = padding_value
#             elif k.endswith("_attention_mask"):
#                 pad_value = 0
#             concatenated_key = k.replace("rejected", "concatenated")
#             concatenated_batch[concatenated_key] = torch.cat(
#                 (
#                     concatenated_batch[concatenated_key],
#                     pad_to_length(batch[k], max_length, pad_value=pad_value),
#                 ),
#                 dim=0,
#             ).to(device=device)

#     if is_encoder_decoder:
#         concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1).to(device=device)
#         concatenated_batch["concatenated_attention_mask"] = (
#             batch["prompt_attention_mask"].repeat(2, 1).to(device=device)
#         )

#     return concatenated_batch

# def tokenize_row(self, feature, model: Optional[Union[PreTrainedModel, nn.Module]] = None) -> Dict:
#     """Tokenize a single row from a DPO specific dataset.

#     At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
#     in case the prompt + chosen or prompt + rejected responses is/are too long. First
#         we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

#     We also create the labels for the chosen/rejected responses, which are of length equal to
#         the sum of the length of the prompt and the chosen/rejected response, with
#         label_pad_token_id  for the prompt tokens.
#     """
#     batch = {}
#     prompt = feature["prompt"]
#     chosen = feature["chosen"]
#     rejected = feature["rejected"]

#     if not self.is_encoder_decoder:
#         # Check issues below for more details
#         #  1. https://github.com/huggingface/trl/issues/907
#         #  2. https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
#         #  3. https://github.com/LianjiaTech/BELLE/issues/337

#         if not isinstance(prompt, str):
#             raise ValueError(f"prompt should be an str but got {type(prompt)}")
#         prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
#         prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

#         if not isinstance(chosen, str):
#             raise ValueError(f"chosen should be an str but got {type(chosen)}")
#         chosen_tokens = self.build_tokenized_answer(prompt, chosen)

#         if not isinstance(rejected, str):
#             raise ValueError(f"rejected should be an str but got {type(rejected)}")
#         rejected_tokens = self.build_tokenized_answer(prompt, rejected)

#         # Last prompt token might get merged by tokenizer and
#         # it should not be included for generation if that happens
#         prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])

#         chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
#         rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])
#         prompt_len_input_ids = min(chosen_prompt_len_input_ids, rejected_prompt_len_input_ids)

#         for k, v in prompt_tokens.items():
#             prompt_tokens[k] = v[:prompt_len_input_ids]

#         # Make sure prompts only have one different token at most an
#         # and length only differs by 1 at most
#         num_diff_tokens = sum(
#             [a != b for a, b in zip(chosen_tokens["prompt_input_ids"], rejected_tokens["prompt_input_ids"])]
#         )
#         num_diff_len = abs(chosen_prompt_len_input_ids - rejected_prompt_len_input_ids)
#         if num_diff_tokens > 1 or num_diff_len > 1:
#             raise ValueError(
#                 "Chosen and rejected prompt_input_ids might only differ on the "
#                 "last token due to tokenizer merge ops."
#             )

#         # add BOS token to head of prompt
#         prompt_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + prompt_tokens["prompt_input_ids"]
#         chosen_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + chosen_tokens["prompt_input_ids"]
#         rejected_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + rejected_tokens["prompt_input_ids"]

#         prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
#         chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]
#         rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens["prompt_attention_mask"]

#         # add EOS token to end of answer
#         chosen_tokens["input_ids"].append(self.tokenizer.eos_token_id)
#         chosen_tokens["attention_mask"].append(1)

#         rejected_tokens["input_ids"].append(self.tokenizer.eos_token_id)
#         rejected_tokens["attention_mask"].append(1)

#         longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

#         # if combined sequence is too long, truncate the prompt
#         for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
#             if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
#                 if self.truncation_mode == "keep_start":
#                     for k in ["prompt_input_ids", "prompt_attention_mask"]:
#                         answer_tokens[k] = answer_tokens[k][: self.max_prompt_length]
#                 elif self.truncation_mode == "keep_end":
#                     for k in ["prompt_input_ids", "prompt_attention_mask"]:
#                         answer_tokens[k] = answer_tokens[k][-self.max_prompt_length :]
#                 else:
#                     raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

#         # if that's still too long, truncate the response
#         for answer_tokens in [chosen_tokens, rejected_tokens]:
#             if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
#                 for k in ["input_ids", "attention_mask"]:
#                     answer_tokens[k] = answer_tokens[k][: self.max_length - self.max_prompt_length]

#         # Create labels
#         chosen_sequence_tokens = {
#             k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]
#         }
#         rejected_sequence_tokens = {
#             k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k] for k in ["input_ids", "attention_mask"]
#         }
#         chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
#         chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [
#             self.label_pad_token_id
#         ] * len(chosen_tokens["prompt_input_ids"])
#         rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
#         rejected_sequence_tokens["labels"][: len(rejected_tokens["prompt_input_ids"])] = [
#             self.label_pad_token_id
#         ] * len(rejected_tokens["prompt_input_ids"])

#         for k, toks in {
#             "chosen_": chosen_sequence_tokens,
#             "rejected_": rejected_sequence_tokens,
#             "": prompt_tokens,
#         }.items():
#             for type_key, tokens in toks.items():
#                 if type_key == "token_type_ids":
#                     continue
#                 batch[f"{k}{type_key}"] = tokens

#     else:
#         chosen_tokens = self.tokenizer(
#             chosen, truncation=True, max_length=self.max_target_length, add_special_tokens=True
#         )
#         rejected_tokens = self.tokenizer(
#             rejected, truncation=True, max_length=self.max_target_length, add_special_tokens=True
#         )
#         prompt_tokens = self.tokenizer(
#             prompt, truncation=True, max_length=self.max_prompt_length, add_special_tokens=True
#         )

#         batch["chosen_labels"] = chosen_tokens["input_ids"]
#         batch["rejected_labels"] = rejected_tokens["input_ids"]
#         batch["prompt_input_ids"] = prompt_tokens["input_ids"]
#         batch["prompt_attention_mask"] = prompt_tokens["attention_mask"]

#         if model is not None and hasattr(model, "prepare_decoder_input_ids_from_labels"):
#             batch["rejected_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
#                 labels=torch.tensor(batch["rejected_labels"])
#             )
#             batch["chosen_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
#                 labels=torch.tensor(batch["chosen_labels"])
#             )

#     return batch

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
    for _ in range(2):
        with tqdm(total=config.max_step) as pbar:
            for step, prompts in enumerate(prompt_batchify(sampling_prompt_dataset, config.acc_steps*config.step_per_feedback)):            
                # specify saved location
                if step == 0:
                    inference_adapter_name="sft"
                    # set up one adapter
                    model=set_up_single_lora(model, f"/share/edc/home/wendaxu/greedy_search_0_sft_lora_256_all_linear_True_fixed_april_29/checkpoint-3057", f"model_0")
                    # load sft lora weights for inference
                    sft_lora_addr_ls = [f"/share/edc/home/wendaxu/greedy_search_{i}_sft_lora_256_all_linear_True_fixed_april_29/checkpoint-3057" for i in range(num_lora)]
                    model=set_up_merge_lora(model, sft_lora_addr_ls, inference_adapter_name)
                else:
                    # replace inference model with last model adapter among gpus
                    model=merge_lora_for_inference(model, num_lora)
                
                if not os.path.isfile(f'online_train_data_{seed}_{mode}/{step}_rank.json'):
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
                        # set the different seeds for different lora
                        training_args.seed=seed+model_index
                        training_args.data_seed=training_args.seed
                        # replace training data with newly updated data
                        model.set_adapter(f"model_{model_index}")
                        # dpo_trainer = set_up_trainer(model, model_index, training_args, train_feedback_data, tokenizer)
                        #training_args.learning_rate=config.lr-step*config.lr/config.max_step
                        #training_args.lr_scheduler_type="constant"
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