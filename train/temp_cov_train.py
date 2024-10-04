import json
import datasets
from datasets import Dataset
from typing import TypeVar, Iterable, Any, Dict, List, Optional, Tuple, Union
import torch
from transformers import PreTrainedModel
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import tqdm
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
from peft import PeftModel
import transformers
import click
import os
import random
import time
from openai import OpenAI
import wandb
from transformers import set_seed
import glob

T = TypeVar('T')
# remove tqdm at data loading
datasets.disable_progress_bar()

class config:
    step_per_feedback=1
    num_epoch=10
    batch_size=1
    acc_steps=16
    lr=5e-5
    ds_config="config/ds_config_zero3.json"
    beta=0.1
    tokenizer=None
    truncation_mode="keep_end"
    max_length=512+128
    max_prompt_length=512
    max_new_token=128
    label_pad_token_id=-100
    is_encoder_decoder=False
    loss_type="sigmoid"
    padding_value=0
    label_smoothing=0
    reference_free=False
    max_step=125
    max_grad_norm=1.0
    num_workers=4
    gamma=1

@dataclass
class DPODataCollatorWithPadding:
    r"""
    DPO DataCollator class that pads the tokenized inputs to the maximum length of the batch.
    Args:
        pad_token_id (`int` defaults to 0):
            The tokenizer's pad_token_id.
        label_pad_token_id (`int`, defaults to -100):
            The label used for masking.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            Whether or not you model has an encoder_decoder architecture.
    """

    pad_token_id: int = 0
    label_pad_token_id: int = -100
    is_encoder_decoder: Optional[bool] = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # first, pad everything to the same length
        padded_batch = {}
        for k in features[0].keys():
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                if self.is_encoder_decoder:
                    to_pad = [torch.LongTensor(ex[k]) for ex in features]

                    if (k.startswith("prompt")) and (k.endswith("input_ids")):
                        if self.pad_token_id is None:
                            raise ValueError(
                                "Padding is enabled, but the tokenizer is not configured with a padding token."
                                " Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                                " before calling the trainer."
                            )
                        padding_value = self.pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif k.startswith(("chosen", "rejected", "completion")) or ("decoder" in k):
                        padding_value = self.label_pad_token_id
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")
                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                else:
                    # adapted from https://stackoverflow.com/questions/73256206
                    if "prompt" in k:
                        to_pad = [torch.LongTensor(ex[k][::-1]) for ex in features]
                    else:
                        to_pad = [torch.LongTensor(ex[k]) for ex in features]
                    if k.endswith("_input_ids"):
                        if self.pad_token_id is None:
                            raise ValueError(
                                "Padding is enabled, but the tokenizer is not configured with a padding token."
                                " Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                                " before calling the trainer."
                            )
                        padding_value = self.pad_token_id
                    elif k.endswith("_labels"):
                        padding_value = self.label_pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")

                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                    # for the prompt, flip back so padding is on left side
                    if "prompt" in k:
                        padded_batch[k] = padded_batch[k].flip(dims=[1])
            elif k.endswith("_logps"):
                # the cached reference model logprobs
                padded_batch[k] = torch.tensor([ex[k] for ex in features])
            else:
                padded_batch[k] = [ex[k] for ex in features]

        return padded_batch

config.tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b')

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

def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [
                tensor,
                pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
            ],
            dim=dim,
        )
    
def build_tokenized_answer(prompt, answer):
    """
    Llama tokenizer does satisfy `enc(a + b) = enc(a) + enc(b)`.
    It does ensure `enc(a + b) = enc(a) + enc(a + b)[len(enc(a)):]`.
    Reference:
        https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
    """

    full_tokenized = config.tokenizer(prompt + answer, add_special_tokens=False)
    prompt_input_ids = config.tokenizer(prompt, add_special_tokens=False)["input_ids"]

    answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
    answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids) :]

    # Concat tokens to form `enc(a) + enc(a + b)[len(enc(a)):]`
    full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])

    # Prepare input tokens for token by token comparison
    full_input_ids = np.array(full_tokenized["input_ids"])

    if len(full_input_ids) != len(full_concat_input_ids):
        raise ValueError("Prompt input ids and answer input ids should have the same length.")

    # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
    # can be merged together when tokenizing prompt+answer. This could result
    # on the last token from the prompt being different when tokenized on its own
    # vs when done as prompt+answer.
    response_token_ids_start_idx = len(prompt_input_ids)

    # If tokenized prompt is different than both prompt+answer, then it means the
    # last token has changed due to merging.
    if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
        response_token_ids_start_idx -= 1

    prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
    prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

    if len(prompt_input_ids) != len(prompt_attention_mask):
        raise ValueError("Prompt input ids and attention mask should have the same length.")

    answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
    answer_attention_mask = full_tokenized["attention_mask"][response_token_ids_start_idx:]

    return dict(
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
        input_ids=answer_input_ids,
        attention_mask=answer_attention_mask,
    )

def concatenated_inputs(
    batch: Dict[str, Union[List, torch.LongTensor]],
    is_encoder_decoder: bool = False,
    label_pad_token_id: int = -100,
    padding_value: int = 0,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.

    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
        is_encoder_decoder: Whether the model is an encoder-decoder model.
        label_pad_token_id: The label pad token id.
        padding_value: The padding value to use for the concatenated inputs_ids.
        device: The device for the concatenated inputs.

    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    concatenated_batch = {}

    if is_encoder_decoder:
        max_length = max(batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1])
    else:
        max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

    for k in batch:
        if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
            if "labels" in k or is_encoder_decoder:
                pad_value = label_pad_token_id
            elif k.endswith("_input_ids"):
                pad_value = padding_value
            elif k.endswith("_attention_mask"):
                pad_value = 0
            concatenated_key = k.replace("chosen", "concatenated")
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
    for k in batch:
        if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
            if "labels" in k or is_encoder_decoder:
                pad_value = label_pad_token_id
            elif k.endswith("_input_ids"):
                pad_value = padding_value
            elif k.endswith("_attention_mask"):
                pad_value = 0
            concatenated_key = k.replace("rejected", "concatenated")
            concatenated_batch[concatenated_key] = torch.cat(
                (
                    concatenated_batch[concatenated_key],
                    pad_to_length(batch[k], max_length, pad_value=pad_value),
                ),
                dim=0,
            ).to(device=device)

    if is_encoder_decoder:
        concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1).to(device=device)
        concatenated_batch["concatenated_attention_mask"] = (
            batch["prompt_attention_mask"].repeat(2, 1).to(device=device)
        )

    return concatenated_batch

def tokenize_row(feature, model: Optional[Union[PreTrainedModel, nn.Module]] = None) -> Dict:
    """Tokenize a single row from a DPO specific dataset.

    At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
    in case the prompt + chosen or prompt + rejected responses is/are too long. First
        we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

    We also create the labels for the chosen/rejected responses, which are of length equal to
        the sum of the length of the prompt and the chosen/rejected response, with
        label_pad_token_id  for the prompt tokens.
    """
    batch = {}
    prompt = feature["prompt"]
    chosen = feature["chosen"]
    rejected = feature["rejected"]

    if not isinstance(prompt, str):
        raise ValueError(f"prompt should be an str but got {type(prompt)}")
    prompt_tokens = config.tokenizer(prompt, add_special_tokens=False)
    prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

    if not isinstance(chosen, str):
        raise ValueError(f"chosen should be an str but got {type(chosen)}")
    chosen_tokens = build_tokenized_answer(prompt, chosen)

    if not isinstance(rejected, str):
        raise ValueError(f"rejected should be an str but got {type(rejected)}")
    rejected_tokens = build_tokenized_answer(prompt, rejected)

    # Last prompt token might get merged by tokenizer and
    # it should not be included for generation if that happens
    prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])

    chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
    rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])
    prompt_len_input_ids = min(chosen_prompt_len_input_ids, rejected_prompt_len_input_ids)

    for k, v in prompt_tokens.items():
        prompt_tokens[k] = v[:prompt_len_input_ids]

    # Make sure prompts only have one different token at most an
    # and length only differs by 1 at most
    num_diff_tokens = sum(
        [a != b for a, b in zip(chosen_tokens["prompt_input_ids"], rejected_tokens["prompt_input_ids"])]
    )
    num_diff_len = abs(chosen_prompt_len_input_ids - rejected_prompt_len_input_ids)
    if num_diff_tokens > 1 or num_diff_len > 1:
        raise ValueError(
            "Chosen and rejected prompt_input_ids might only differ on the "
            "last token due to tokenizer merge ops."
        )

    # add BOS token to head of prompt
    prompt_tokens["prompt_input_ids"] = [config.tokenizer.bos_token_id] + prompt_tokens["prompt_input_ids"]
    chosen_tokens["prompt_input_ids"] = [config.tokenizer.bos_token_id] + chosen_tokens["prompt_input_ids"]
    rejected_tokens["prompt_input_ids"] = [config.tokenizer.bos_token_id] + rejected_tokens["prompt_input_ids"]

    prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
    chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]
    rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens["prompt_attention_mask"]

    # add EOS token to end of answer
    chosen_tokens["input_ids"].append(config.tokenizer.eos_token_id)
    chosen_tokens["attention_mask"].append(1)

    rejected_tokens["input_ids"].append(config.tokenizer.eos_token_id)
    rejected_tokens["attention_mask"].append(1)

    longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

    # if combined sequence is too long, truncate the prompt
    for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
        if len(answer_tokens["prompt_input_ids"]) + longer_response_length > config.max_length:
            if config.truncation_mode == "keep_start":
                for k in ["prompt_input_ids", "prompt_attention_mask"]:
                    answer_tokens[k] = answer_tokens[k][: config.max_prompt_length]
            elif config.truncation_mode == "keep_end":
                for k in ["prompt_input_ids", "prompt_attention_mask"]:
                    answer_tokens[k] = answer_tokens[k][-config.max_prompt_length :]
            else:
                raise ValueError(f"Unknown truncation mode: {config.truncation_mode}")

    # if that's still too long, truncate the response
    for answer_tokens in [chosen_tokens, rejected_tokens]:
        if len(answer_tokens["prompt_input_ids"]) + longer_response_length > config.max_length:
            for k in ["input_ids", "attention_mask"]:
                answer_tokens[k] = answer_tokens[k][: config.max_length - config.max_prompt_length]

    # Create labels
    chosen_sequence_tokens = {
        k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]
    }
    rejected_sequence_tokens = {
        k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k] for k in ["input_ids", "attention_mask"]
    }
    chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
    chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [
        config.label_pad_token_id
    ] * len(chosen_tokens["prompt_input_ids"])
    rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
    rejected_sequence_tokens["labels"][: len(rejected_tokens["prompt_input_ids"])] = [
        config.label_pad_token_id
    ] * len(rejected_tokens["prompt_input_ids"])

    for k, toks in {
        "chosen_": chosen_sequence_tokens,
        "rejected_": rejected_sequence_tokens,
        "": prompt_tokens,
    }.items():
        for type_key, tokens in toks.items():
            if type_key == "token_type_ids":
                continue
            batch[f"{k}{type_key}"] = tokens

    return batch

@staticmethod
def get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    average_log_prob: bool = False,
    label_pad_token_id: int = -100,
    is_encoder_decoder: bool = False,
) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
        label_pad_token_id: The label pad token id.
        is_encoder_decoder: Whether the model is an encoder-decoder model.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

    if not is_encoder_decoder:
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
    loss_mask = labels != label_pad_token_id

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == label_pad_token_id] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)

def concatenated_forward(
    model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

    We do this to avoid doing two forward passes, because it's faster for FSDP.
    """
    concatenated_batch = concatenated_inputs(
        batch,
        is_encoder_decoder=config.is_encoder_decoder,
        label_pad_token_id=config.label_pad_token_id,
        padding_value=config.padding_value,
        device=model.device,
    )
    len_chosen = batch["chosen_labels"].shape[0]

    model_kwargs = (
        {
            "labels": concatenated_batch["concatenated_labels"],
            "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
        }
        if config.is_encoder_decoder
        else {}
    )

    all_logits = model(
        concatenated_batch["concatenated_input_ids"],
        attention_mask=concatenated_batch["concatenated_attention_mask"],
        use_cache=False,
        **model_kwargs,
    ).logits

    all_logps = get_batch_logps(
        all_logits,
        concatenated_batch["concatenated_labels"],
        average_log_prob=config.loss_type == "ipo",
        is_encoder_decoder=config.is_encoder_decoder,
        label_pad_token_id=config.label_pad_token_id,
    )

    chosen_logps = all_logps[:len_chosen]
    rejected_logps = all_logps[len_chosen:]

    chosen_logits = all_logits[:len_chosen]
    rejected_logits = all_logits[len_chosen:]

    return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

def dpo_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    if config.reference_free:
        ref_logratios = torch.tensor([0], dtype=pi_logratios.dtype, device=pi_logratios.device)
    else:
        ref_logratios = reference_chosen_logps - reference_rejected_logps

    pi_logratios = pi_logratios.to('cuda')
    ref_logratios = ref_logratios.to('cuda')
    logits = pi_logratios - ref_logratios

    # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
    # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
    # calculates a conservative DPO loss.
    if config.loss_type == "sigmoid":
        losses = (
            -F.logsigmoid(config.beta * logits) * (1 - config.label_smoothing)
            - F.logsigmoid(-config.beta * logits) * config.label_smoothing
        )
    elif config.loss_type == "hinge":
        losses = torch.relu(1 - config.beta * logits)
    elif config.loss_type == "ipo":
        # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
        losses = (logits - 1 / (2 * config.beta)) ** 2
    elif config.loss_type == "kto_pair":
        # eqn (7) of the HALOs paper
        chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
        rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps
        # As described in the KTO report, the KL term for chosen (rejected) is estimated using the rejected (chosen) half.
        losses = torch.cat(
            (
                1 - F.sigmoid(config.beta * (chosen_logratios - rejected_KL)),
                1 - F.sigmoid(config.beta * (chosen_KL - rejected_logratios)),
            ),
            0,
        )
    else:
        raise ValueError(
            f"Unknown loss type: {config.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair', 'bco_pair']"
        )

    chosen_rewards = (
        config.beta
        * (
            policy_chosen_logps.to('cuda') - reference_chosen_logps.to('cuda')
        ).detach()
    )
    rejected_rewards = (
        config.beta
        * (
            policy_rejected_logps.to('cuda')
            - reference_rejected_logps.to('cuda')
        ).detach()
    )

    return losses, chosen_rewards, rejected_rewards

def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0

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

def logits_compute(tokenizer, model, prompts, outputs, adapter_name):
    tokd_all = tokenizer(outputs, return_tensors='pt', max_length=config.max_new_token+config.max_prompt_length, truncation=True, padding=True) # "max_length"
    tokd_gen = tokenizer(prompts, return_tensors='pt', max_length=config.max_new_token, truncation=True, padding=True) # max_length"

    labels = tokd_all["input_ids"].clone().detach()
    # depends on the padding strategy! You need to verify if model is left padded
    labels[:, :labels.shape[1] - tokd_gen["input_ids"].shape[1] + 1] = -100
    torch.cuda.empty_cache()
    # set which lora to use during logits computation
    model.set_adapter(adapter_name)
    all_logits = model(input_ids=tokd_all["input_ids"].to('cuda'), attention_mask=tokd_all['attention_mask'].to('cuda')).logits 
    logits = get_batch_logps(all_logits, labels.to('cuda'), True).to('cpu')
    return logits

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
            cur_ucb_ls += [ele_mean_reward+config.gamma * ele_std_reward]
            cur_lcb_ls += [ele_mean_reward-config.gamma * ele_std_reward]
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

def inference(prompts, tokenizer, model, num_ret, mode, num_lora, inference_adapter_name):
    model.eval()
    final_ls = []
    if mode != "rand_rand":
        all_rank_stats={i: [] for i in range(num_lora)}
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer([prompt], return_tensors="pt", padding=True, truncation=True, max_length=config.max_prompt_length, add_special_tokens=False).to('cuda') 
            # make sure the inference adapter name is correct
            model.set_adapter(inference_adapter_name)
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
                    rev_rank_ls = [sorted(rewards_ls).index(x) for x in rewards_ls]
                    all_rank_stats[model_index] += [rev_rank_ls]
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
                if api_source == "openai":
                    answer = completions_with_backoff_openai(config.client, config.system_prompt, prompt_txt, config.model_type, config.n)
                else:
                    print("We donnot support other APIs at the moment!")
                    exit(1)
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

@click.command()
@click.option('-mode', default="rand_rand", type=str)
@click.option('-seed', type=int, default=None)
@click.option('-api_source', default='openai', type=str)
@click.option('-start_index', type=int, default=0)
@click.option('-end_index', type=int, default=2000)
@click.option('-num_lora', type=int, default=8)
@click.option('-num_ret', type=int, default=12)
@click.option('-gpu_takes_n', type=int, default=12)
@click.option('-prefix', type=str, default="/scratch/wendaxu")
@click.option('-per_epoch_save', type=bool, default=False)
@click.option('-lr', type=float, default=1e-5)
def main(mode, seed, api_source, start_index, end_index, num_lora, num_ret, prefix, gpu_takes_n, per_epoch_save, lr):
    # ensure reproducibility with fixed seed
    if seed:
        set_seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.mps.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    
    config.client = OpenAI()
    config.system_prompt="You are a text generation evaluater."
    config.model_type="gpt-3.5-turbo-0125"
    config.n=1
    config.train_size=end_index-start_index
    config.predix=prefix
    config.gpu_takes_n=gpu_takes_n
    config.max_step=int(config.num_epoch*config.train_size/(config.batch_size*config.acc_steps))
    config.lr=lr

    data_collator = DPODataCollatorWithPadding(
        pad_token_id=config.tokenizer.pad_token_id,
        label_pad_token_id=config.label_pad_token_id,
        is_encoder_decoder=False,
    )
    accelerator = Accelerator(gradient_accumulation_steps=config.acc_steps)
    config.accelerator=accelerator

    model = AutoModelForCausalLM.from_pretrained('google/gemma-2b', torch_dtype=torch.bfloat16, device_map="cpu").to('cuda') # attn_implementation="flash_attention_2"
    inference_adapter_name = "sft"
    # set up one adapter
    model=set_up_single_lora(model, f"xu1998hz/0_sft_lora_256", f"model_0")
    # load sft lora weights for inference
    sft_lora_addr_ls = [f"xu1998hz/{i}_sft_lora_256" for i in range(num_lora)]
    model=set_up_merge_lora(model, sft_lora_addr_ls, inference_adapter_name)
    model.print_trainable_parameters()

    # disable dropout in model
    disable_dropout_in_model(model)

    # initialize lr and optimizer
    optimizer = transformers.AdamW(model.parameters(), lr=config.lr)
    lr_scheduler = transformers.get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=config.max_step)

    # make sure login id is right
    wandb.login(key="377341f70d62101cfbf553f4c38879b82cdae78b", relogin=True)

    wandb.init(project='DPO_Master', name=f"{mode}_{seed}_finalized_may_5", config=\
    {
        "seed": seed,
        "mode": mode,
        "train batch size": config.batch_size * config.acc_steps,
        "lr": config.lr,
        "beta": config.beta,
        "sample size": num_ret,
        "lora size": num_lora 
    })

    # load in dataset
    with open('sft_data/sampling.jsonl') as f:
        sampling_prompt_dataset = [json.loads(line) for line in f]
    sampling_prompt_dataset = sampling_prompt_dataset[start_index:end_index]

    p_ls, c_ls, r_ls = [], [], []
    for i in range(125):
        cur_data = json.load(open(f'online_train_data_{seed}_{mode}/{i}_rank.json'))
        p_ls+=cur_data['prompt']
        c_ls+=cur_data['chosen']
        r_ls+=cur_data['rejected']

    # dataset needs to be dynamically updated
    with tqdm(total=config.max_step) as pbar:
        for epoch_index in range(config.num_epoch):
            tr_loss = 0
            reward_acc = 0
            config.global_step = 0
            model.zero_grad()
            grad_norm: Optional[float] = None
        
            # shuffle all data at the same time
            temp = list(zip(p_ls, c_ls, r_ls))
            random.shuffle(temp)
            p_ls, c_ls, r_ls = zip(*temp)
            # p_ls, c_ls, r_ls come out as tuples, and so must be converted to lists.
            p_ls, c_ls, r_ls = list(p_ls), list(c_ls), list(r_ls)

            for cur_p_ls, cur_c_ls, cur_r_ls in zip(batchify(p_ls, config.acc_steps), batchify(c_ls, config.acc_steps), batchify(r_ls, config.acc_steps)):
                selected_data = {} 
                selected_data['prompt']=cur_p_ls
                selected_data['chosen']=cur_c_ls
                selected_data['rejected']=cur_r_ls
                # sanity check the size of data, it should match with acc_step * step_per_feedback
                if len(selected_data['prompt']) != config.acc_steps * config.step_per_feedback:
                    print("Data size is not matching with online experimental details!")
                    # update gradient accumulation size with current data size
                    cur_acc_steps=int(len(selected_data['prompt'])/config.step_per_feedback)
                else:
                    cur_acc_steps=config.acc_steps
                    
                train_feedback_data = Dataset.from_dict(selected_data)
                train_dataset = train_feedback_data.map(tokenize_row, num_proc=config.num_workers)

                dataloader_params = {
                    "batch_size": config.batch_size,
                    "collate_fn": data_collator,
                    "num_workers": config.num_workers,
                    "pin_memory": True,
                    "shuffle": True,
                }

                # prepare dataloader
                data_loader = DataLoader(train_dataset, **dataloader_params)

                temp_report_dict={}
                # perform gradient accumulation
                for padded_batch in data_loader:
                    # config.accelerator.free_memory()
                    model.set_adapter("model_0")
                    model.train()
                    (policy_chosen_logps, policy_rejected_logps, policy_chosen_logits, policy_rejected_logits) = concatenated_forward(
                        model, padded_batch
                    )

                    with torch.no_grad():
                        # switch to reference adapter
                        model.set_adapter("sft")
                        (reference_chosen_logps, reference_rejected_logps, _, _,) = concatenated_forward(model, padded_batch)
                    model.set_adapter("model_0")
                    
                    losses, chosen_rewards, rejected_rewards = dpo_loss(
                        policy_chosen_logps,
                        policy_rejected_logps,
                        reference_chosen_logps,
                        reference_rejected_logps,
                    )

                    # per accumulated batch gradient update
                    tr_loss+=losses.item() / cur_acc_steps
                    loss = losses / cur_acc_steps
                    loss.backward()

                    reward_acc += (chosen_rewards > rejected_rewards).float() / cur_acc_steps
                    # print("Stepwise reward acc: ", (chosen_rewards > rejected_rewards).float())

                config.global_step+=1
                # only pass the gradient, learning rate, clip gradients at accumulation step   
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.max_grad_norm,
                ) 
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # sum of per batch acc divides steps
                reward_accuracies = reward_acc/config.global_step
                temp_report_dict['LR']=lr_scheduler.get_last_lr()[0]
                temp_report_dict['Train_loss']=tr_loss/config.global_step
                temp_report_dict['Grad_norm']=grad_norm.item()
                temp_report_dict['Reward_acc']=reward_accuracies.item()
                wandb.log(temp_report_dict)
                pbar.update(1)
            
            # save at end of epoch
            if per_epoch_save:
                model.save_pretrained(f'{prefix}/epoch_{epoch_index}_{mode}_{seed}_may_7_{config.lr}_fixed')
                print("Save at end of epoch!")
                
if __name__ == "__main__":
    main()