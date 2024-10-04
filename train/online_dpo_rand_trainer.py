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
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, GemmaTokenizerFast
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
import sys
from itertools import combinations

sys.setrecursionlimit(10000)
current_limit = sys.getrecursionlimit()
print("Current recursion limit:", current_limit)

T = TypeVar('T')
# remove tqdm at data loading
datasets.disable_progress_bar()

class config:
    step_per_feedback=1
    ds_config="config/ds_config_zero3.json"
    beta=0.1
    tokenizer=None
    truncation_mode="keep_end"
    label_pad_token_id=-100
    is_encoder_decoder=False
    padding_value=0
    label_smoothing=0
    reference_free=False
    max_grad_norm=1.0
    logits_batch_size=4

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

def set_up_merge_lora_ref(model, sft_lora_addr_ls, weighted_adapter_name):
    model_names=[]
    for i, ele in enumerate(sft_lora_addr_ls):    
        model.load_adapter(ele, adapter_name=f"model_ref_{i}")
        model_names+=[f"model_ref_{i}"]
    
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

def merge_lora_for_inference(model, num_lora, dpo_inference_name):
    model_names=[f"model_{i}" for i in range(num_lora)]
    # perform model averaging on lora weights
    model.add_weighted_adapter(
        adapters=model_names,
        weights=[1/len(model_names)]*len(model_names),
        adapter_name=dpo_inference_name,
        combination_type="linear"
    )
    model.set_adapter(dpo_inference_name)

    # set all sft or weighted_adapter_name parameters to be non-grad
    for name, param in model.named_parameters():
        if dpo_inference_name in name:
            param.requires_grad = False
    return model

def change_lora_weight_name(model, cur_lora_name, new_lora_name):
    model.add_weighted_adapter(
        adapters=[cur_lora_name],
        weights=[1],
        adapter_name=new_lora_name,
        combination_type="linear"
    )
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

def ret_rank_dict(prompts, tokenizer, model, sampling_strategy):
    model.eval()
    final_ls = []
    
    with torch.no_grad():
        for prompt_batch in batchify(prompts, config.sampling_batch_size):
            new_prompt_batch=[]
            for ele in prompt_batch:
                for _ in range(config.gpu_takes_n):
                    new_prompt_batch+=[ele]
            
            inputs = tokenizer(new_prompt_batch, return_tensors="pt", padding=True, truncation=True, max_length=config.max_prompt_length).to('cuda') 
            # make sure the inference adapter name is correct
            summary_ls = []
            start_time=time.time()

            if config.run_sample_strategy=="merged" or config.run_sample_strategy=="sft" or config.run_sample_strategy=="per_interval_merged":
                # make sure either sft or dpo_inference is activated for sampling
                if config.run_sample_strategy=="sft":
                    model.set_adapter(config.sft_adapter_name)
                elif config.run_sample_strategy=="per_interval_merged":
                    model.set_adapter(f"cur_{config.dpo_inference_name}")
                else:
                    model.set_adapter(config.dpo_inference_name)

                if sampling_strategy == "top-p":
                    if config.rank_strategy == "reward":
                        output = model.generate(inputs=inputs.input_ids, max_new_tokens=config.max_new_token, do_sample=True, temperature=1.0, top_k=0, top_p=0.95, 
                                                num_return_sequences=1)
                    elif config.rank_strategy == "logits":
                        output = model.generate(inputs=inputs.input_ids, max_new_tokens=config.max_new_token, do_sample=True, temperature=1.0, top_k=0, top_p=0.95, 
                                                num_return_sequences=1, return_dict_in_generate=True, output_scores=True)
                    else:
                        print("Your rank strategy is not supported!")
                        exit(1)
                elif sampling_strategy == "ancestral":
                    if config.rank_strategy == "reward":
                        output = model.generate(inputs=inputs.input_ids, max_new_tokens=config.max_new_token, do_sample=True, num_beams=1, 
                                                num_return_sequences=1)
                    elif config.rank_strategy == "logits":
                        output = model.generate(inputs=inputs.input_ids, max_new_tokens=config.max_new_token, do_sample=True, num_beams=1, 
                                                num_return_sequences=1, return_dict_in_generate=True, output_scores=True)  
                    else:
                        print("Your rank strategy is not supported!")
                        exit(1)
                else:
                    print("Other sampling strategies are not supported!")
                    exit(1)
                
                # make sure output is in the order of input
                cur_summ_ls = tokenizer.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
                
                for cur_summ_ls_batch in batchify(cur_summ_ls, config.gpu_takes_n):
                    summary_ls += [list(cur_summ_ls_batch)]
            else:
                print("Other sampling is not supported!")
                exit(1)

            print("Inference time: ", time.time()-start_time)
            print("Current sample size: ", len(summary_ls))
            final_ls+=summary_ls
    return final_ls, None
       

def inference(prompts, tokenizer, model, sampling_strategy):
    final_ls, _ = ret_rank_dict(prompts, tokenizer, model, sampling_strategy)
    
    first_ls, sec_ls, final_prompt_ls = [], [], []
    # perform random selections of two samples
    for ele, prompt in zip(final_ls, prompts):
        temp_ls = random.sample(ele, 2)
        if temp_ls[0] != temp_ls[1]:
            first_ls += [temp_ls[0]]
            sec_ls += [temp_ls[1]]
            final_prompt_ls += [prompt]
    return final_prompt_ls, first_ls, sec_ls

def ret_from_openai(prompt, first, second, api_source):
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
            return prompt, first, second
        elif answer.split('Preferred:')[1].strip() == 'B':
            return prompt, second, first
        else:
            return None
        
def compute_gt_reward(prompts, first_ls):
    reward_model=config.reward_model.to('cuda')
    scores_ls=[]
    for prompt, first in zip(prompts, first_ls):
        inputs = config.reward_tok([prompt], [first], return_tensors='pt', padding=True, truncation=True, max_length=640).to('cuda')
        scores_ls += [reward_model(**inputs).logits.cpu().detach().tolist()[0][0]]
    return scores_ls

def ret_train_data(prompts, first_ls, sec_ls):
    reward_model=config.reward_model.to('cuda')
    chosen_ls, rejected_ls = [], []
    for prompt, first, second in zip(prompts, first_ls, sec_ls):
        inputs = config.reward_tok([prompt], [first], return_tensors='pt', padding=True, truncation=True, max_length=640).to('cuda')
        first_score = reward_model(**inputs).logits.cpu().detach().tolist()[0][0]

        inputs = config.reward_tok([prompt], [second], return_tensors='pt', padding=True, truncation=True, max_length=640).to('cuda')
        sec_score =  reward_model(**inputs).logits.cpu().detach().tolist()[0][0]
    
        if first_score > sec_score:
            chosen_ls+=[first]
            rejected_ls+=[second]
        else:
            chosen_ls+=[second]
            rejected_ls+=[first]

    selected_data={}
    selected_data['prompt'] = prompts
    selected_data['chosen'] = chosen_ls
    selected_data['rejected'] = rejected_ls

    return selected_data

def compute_reward_from_models(selected_data, data_collator, model, model_index, ref_model_index=None, reference_chosen_logps_ls=None, reference_rejected_logps_ls=None):
    policy_chosen_logps_ls, policy_rejected_logps_ls = [], []
    final_select_data = {'prompt': selected_data['prompt'], 'chosen': selected_data['chosen'], 'rejected': selected_data['rejected'], 
                         'chosen_input_ids': [], 'chosen_attention_mask': [], 'chosen_labels': [], 
                         'rejected_input_ids': [], 'rejected_attention_mask': [], 'rejected_labels': [], 
                         'prompt_input_ids': [], 'prompt_attention_mask': []}
    for prompt, chosen, rejectetd in zip(selected_data['prompt'], selected_data['chosen'], selected_data['rejected']):
        cur_dict = tokenize_row({"prompt": prompt, "chosen": chosen, "rejected": rejectetd})
        for key in ['chosen_input_ids', 'chosen_attention_mask', 'chosen_labels', 'rejected_input_ids', 'rejected_attention_mask', 'rejected_labels', 'prompt_input_ids', 'prompt_attention_mask']:
            final_select_data[key]+=[cur_dict[key]]
    train_dataset = Dataset.from_dict(final_select_data)

    dataloader_params = {
        "batch_size": config.logits_batch_size,
        "collate_fn": data_collator,
        "num_workers": 1,
        "pin_memory": False,
        "shuffle": False,
    }

    # prepare dataloader
    data_loader = DataLoader(train_dataset, **dataloader_params)

    model.set_adapter(model_index)
    chosen_ls, rejected_ls = [], []
    # perform gradient accumulation
    for padded_batch in data_loader:
        (policy_chosen_logps, policy_rejected_logps, _, _) = concatenated_forward(model, padded_batch)
        policy_chosen_logps_ls+=[policy_chosen_logps]
        policy_rejected_logps_ls+=[policy_rejected_logps]

    if reference_chosen_logps_ls == None or reference_rejected_logps_ls == None:
        model.set_adapter(ref_model_index)
        for padded_batch in data_loader:
            # switch to reference adapter
            (reference_chosen_logps, reference_rejected_logps, _, _,) = concatenated_forward(model, padded_batch)
            reference_chosen_logps_ls+=[reference_chosen_logps]
            reference_rejected_logps_ls+=[reference_rejected_logps]

    for policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps in zip(policy_chosen_logps_ls, policy_rejected_logps_ls, reference_chosen_logps_ls, reference_rejected_logps_ls):
        _, chosen_rewards, rejected_rewards = dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        chosen_ls+=chosen_rewards.tolist()
        rejected_ls+=rejected_rewards.tolist()
    return chosen_ls, rejected_ls

def compute_ref_reward_from_models(selected_data, data_collator, model, ref_model_index):
    reference_chosen_logps_ls, reference_rejected_logps_ls = [], []
    final_select_data = {'prompt': selected_data['prompt'], 'chosen': selected_data['chosen'], 'rejected': selected_data['rejected'], 
                         'chosen_input_ids': [], 'chosen_attention_mask': [], 'chosen_labels': [], 
                         'rejected_input_ids': [], 'rejected_attention_mask': [], 'rejected_labels': [], 
                         'prompt_input_ids': [], 'prompt_attention_mask': []}
    for prompt, chosen, rejectetd in zip(selected_data['prompt'], selected_data['chosen'], selected_data['rejected']):
        cur_dict = tokenize_row({"prompt": prompt, "chosen": chosen, "rejected": rejectetd})
        for key in ['chosen_input_ids', 'chosen_attention_mask', 'chosen_labels', 'rejected_input_ids', 'rejected_attention_mask', 'rejected_labels', 'prompt_input_ids', 'prompt_attention_mask']:
            final_select_data[key]+=[cur_dict[key]]
    train_dataset = Dataset.from_dict(final_select_data)

    dataloader_params = {
        "batch_size": config.logits_batch_size,
        "collate_fn": data_collator,
        "num_workers": 1,
        "pin_memory": False,
        "shuffle": False,
    }

    # prepare dataloader
    data_loader = DataLoader(train_dataset, **dataloader_params)
    # switch to reference adapter
    model.set_adapter(ref_model_index)

    # perform gradient accumulation
    with torch.no_grad():
        for padded_batch in data_loader:
            (reference_chosen_logps, reference_rejected_logps, _, _,) = concatenated_forward(model, padded_batch)
            reference_chosen_logps_ls+=[reference_chosen_logps]
            reference_rejected_logps_ls+=[reference_rejected_logps]
    return reference_chosen_logps_ls, reference_rejected_logps_ls

def training_step(selected_data, data_collator, model, cur_acc_steps, optimizer_ls, lr_scheduler_ls, num_lora):
    final_select_data = {'prompt': selected_data['prompt'], 'chosen': selected_data['chosen'], 'rejected': selected_data['rejected'], 
                         'chosen_input_ids': [], 'chosen_attention_mask': [], 'chosen_labels': [], 
                         'rejected_input_ids': [], 'rejected_attention_mask': [], 'rejected_labels': [], 
                         'prompt_input_ids': [], 'prompt_attention_mask': []}
    for prompt, chosen, rejectetd in zip(selected_data['prompt'], selected_data['chosen'], selected_data['rejected']):
        cur_dict = tokenize_row({"prompt": prompt, "chosen": chosen, "rejected": rejectetd})
        for key in ['chosen_input_ids', 'chosen_attention_mask', 'chosen_labels', 'rejected_input_ids', 'rejected_attention_mask', 'rejected_labels', 'prompt_input_ids', 'prompt_attention_mask']:
            final_select_data[key]+=[cur_dict[key]]
    train_dataset = Dataset.from_dict(final_select_data)

    dataloader_params = {
        "batch_size": config.batch_size,
        "collate_fn": data_collator,
        "num_workers": 1,
        "pin_memory": False,
        "shuffle": False,
    }

    # prepare dataloader
    data_loader = DataLoader(train_dataset, **dataloader_params)
    temp_report_dict={}
    model.train()

    config.chosen_ref_logp=0
    config.rejected_ref_logp=0
    all_ref_chosen_logs_ls, all_ref_reject_logs_ls = [], []
    # only compute ref lop-p once 
    with torch.no_grad():
        if config.train_baseline_model_mode == 'sft':
            model.set_adapter(config.sft_adapter_name)
        elif config.train_baseline_model_mode == 'merge_on_interval':
            model.set_adapter(f"cur_{config.dpo_inference_name}")
        elif config.train_baseline_model_mode == "ema":
            model.set_adapter("ref_model")
        elif config.train_baseline_model_mode == "constrained_ref":
            model.set_adapter("constrained_ref")
        else:
            print(f"We do not support your current {config.train_baseline_model_mode}")
            exit(1)
            
        for padded_batch in data_loader:
            # switch to reference adapter
            (reference_chosen_logps, reference_rejected_logps, _, _,) = concatenated_forward(model, padded_batch)
            config.chosen_ref_logp += torch.sum(reference_chosen_logps).item() / cur_acc_steps
            config.rejected_ref_logp += torch.sum(reference_rejected_logps).item() / cur_acc_steps
            all_ref_chosen_logs_ls+=[reference_chosen_logps]
            all_ref_reject_logs_ls+=[reference_rejected_logps]

    for model_index in range(num_lora):
        model.set_adapter(f"model_{model_index}")
        config.chosen_policy_logp[model_index]=0
        config.rejected_policy_logp[model_index]=0
        config.chosen_rewards[model_index]=0
        config.rejected_rewards[model_index]=0
        config.rewards_margins[model_index]=0
        # perform gradient accumulation
        for cur_index, padded_batch in enumerate(data_loader):
            # config.accelerator.free_memory()
            (policy_chosen_logps, policy_rejected_logps, _, _) = concatenated_forward(
                model, padded_batch
            )
            
            losses, chosen_rewards, rejected_rewards = dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                all_ref_chosen_logs_ls[cur_index], 
                all_ref_reject_logs_ls[cur_index]
            )
            
            # per accumulated batch gradient update
            config.tr_loss_ls[model_index]+=torch.sum(losses).item() / cur_acc_steps
            loss = torch.sum(losses) / cur_acc_steps
            loss.backward()
            # stepwise rewards
            config.reward_acc_ls[model_index] += torch.sum((chosen_rewards > rejected_rewards)).float() / cur_acc_steps
            # stepwise logps for policy chosen, policy rejected, ref chosen and ref rejected
            config.chosen_policy_logp[model_index] += torch.sum(policy_chosen_logps).item() / cur_acc_steps
            config.rejected_policy_logp[model_index] += torch.sum(policy_rejected_logps).item() / cur_acc_steps
            config.chosen_rewards[model_index] += torch.sum(chosen_rewards).item() / cur_acc_steps
            config.rejected_rewards[model_index] += torch.sum(rejected_rewards).item() / cur_acc_steps
            config.rewards_margins[model_index] += torch.sum(chosen_rewards - rejected_rewards).item() / cur_acc_steps
        
        config.global_step_ls[model_index]+=1
        # only pass the gradient, learning rate, clip gradients at accumulation step   
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.max_grad_norm,
        ) 
        optimizer_ls[model_index].step()
        lr_scheduler_ls[model_index].step()
        optimizer_ls[model_index].zero_grad()
        # sum of per batch acc divides steps
        reward_accuracies=config.reward_acc_ls[model_index]/config.global_step_ls[model_index]
        temp_report_dict[f'LR_{model_index}']=lr_scheduler_ls[model_index].get_last_lr()[0]
        temp_report_dict[f'Train_loss_{model_index}']=config.tr_loss_ls[model_index]/config.global_step_ls[model_index]
        temp_report_dict[f'Grad_norm_{model_index}']=grad_norm.item()
        temp_report_dict[f'Reward_acc_{model_index}']=reward_accuracies.item()
        temp_report_dict[f'chosen_policy_logp_{model_index}']=config.chosen_policy_logp[model_index]
        temp_report_dict[f'rejected_policy_logp_{model_index}']=config.rejected_policy_logp[model_index]
        temp_report_dict[f'chosen_rewards_{model_index}']=config.chosen_rewards[model_index]
        temp_report_dict[f'rejected_rewards_{model_index}']=config.rejected_rewards[model_index]
        temp_report_dict[f'rewards_margins_{model_index}']=config.rewards_margins[model_index]
        print(f"Model {model_index} is trained!")
    temp_report_dict[f'chosen_ref_logp']=config.chosen_ref_logp
    temp_report_dict[f'rejected_ref_logp']=config.rejected_ref_logp
    # calculate std for stats
    temp_report_dict[f'chosen_policy_logp_std']=np.std([config.chosen_policy_logp[i] for i in range(num_lora)])
    temp_report_dict[f'rejected_policy_logp_std']=np.std([config.rejected_policy_logp[i] for i in range(num_lora)])
    temp_report_dict[f'chosen_rewards_std']=np.std([config.chosen_rewards[i] for i in range(num_lora)])
    temp_report_dict[f'rejected_rewards_std']=np.std([config.rejected_rewards[i] for i in range(num_lora)])
    temp_report_dict[f'rewards_margins_std']=np.std([config.rewards_margins[i] for i in range(num_lora)])
        
    if config.logging_selection:
        temp_report_dict['per_mis_rank_hit_rate']=config.per_batch_acc
        temp_report_dict['all_mis_rank_hit_rate']=config.all_rank_acc/config.global_step_ls[model_index]
    wandb.log(temp_report_dict)

def eval_over_steps(model, rank_model, rank_tok, tokenizer, save_name):
    print("Start evaluation!")
    start_time=time.time()
    model.eval()
    model.set_adapter(config.dpo_inference_name)
    # load in testing set for current dataset
    with open(f'sft_{config.task_type}_data/test.jsonl') as f:
        eval_dataset = [json.loads(line) for line in f]

    with torch.no_grad():
        data = [] # obtain all responses from model
        for prompts in prompt_batchify(eval_dataset, config.sampling_batch_size):
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=config.max_prompt_length).to('cuda') 
            output = model.generate(inputs=inputs.input_ids, max_new_tokens=128, do_sample=False, temperature=0, num_return_sequences=1)
            outputs = tokenizer.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            data+=outputs

        # save all data output at given evaluation to leave records
        with open(save_name, 'w') as f:
            f.write('[SEP_WENDA]'.join(data))
            print(f"{save_name} is saved at given evaluation step")

        count, total, reward_diff = 0, 0, 0
        for response_ls, eval_data_batch in zip(batchify(data, config.eval_batch_size), batchify(eval_dataset, config.eval_batch_size)):
            prompt_batch=[ele['prompt'] for ele in eval_data_batch]
            inputs = rank_tok(prompt_batch, [ele['label'] for ele in eval_data_batch], return_tensors='pt', padding=True, truncation=True, max_length=config.max_length).to('cuda')
            ref_score_ls = rank_model(**inputs).logits.cpu().detach().tolist()

            inputs = rank_tok(prompt_batch, response_ls, return_tensors='pt', padding=True, truncation=True, max_length=config.max_length).to('cuda')
            summ_score_ls = rank_model(**inputs).logits.cpu().detach().tolist()

            for ref_score, summ_score in zip(ref_score_ls, summ_score_ls):
                reward_diff+=summ_score[0]-ref_score[0]
                if summ_score[0] > ref_score[0]:
                    count += 1
                total += 1
    print("Total time for evaluation: ", time.time()-start_time)
    return count/total, reward_diff/total

# set data_capacity to be 100
def replay_data_buffer(data_addr, data_capacity, num_lora, step, epoch):
    # uniformly sample all recent data
    final_data = {'prompt': [], 'chosen': [], 'rejected': []}
    for _ in range(num_lora):
        for iter_index in range(max(step-data_capacity+1, 0), step+1):
            cur_data = json.load(open(f'{data_addr}/epoch_{epoch}/{iter_index}_rank.json'))
            final_data['prompt']+=cur_data['prompt']
            final_data['chosen']+=cur_data['chosen']
            final_data['rejected']+=cur_data['rejected']
    
    # shuffle all data at the same time
    temp = list(zip(final_data['prompt'], final_data['chosen'], final_data['rejected']))
    random.shuffle(temp)
    p_ls, c_ls, r_ls = zip(*temp)
    # p_ls, c_ls, r_ls come out as tuples, and so must be converted to lists.
    p_ls, c_ls, r_ls = list(p_ls), list(c_ls), list(r_ls)
    return {'prompt': p_ls[:config.acc_steps], 'chosen': c_ls[:config.acc_steps], 'rejected': r_ls[:config.acc_steps]}

@click.command()
@click.option('-mode', default="rand_rand", type=str)
@click.option('-seed', type=int, default=None)
@click.option('-api_source', default=None, type=str)
@click.option('-start_index', type=int, default=0)
@click.option('-end_index', type=int, default=2000)
@click.option('-num_lora', type=int, default=8)
@click.option('-num_ret', type=int, default=12)
@click.option('-gpu_takes_n', type=int, default=12)
@click.option('-prefix', type=str, default="/data/user_data/xixu/wendaxu")
@click.option('-per_epoch_save', type=bool, default=False)
@click.option('-lr', type=float, default=5e-5)
@click.option('-flash_attn_enable', type=bool, default=False)
@click.option('-sampling_strategy', type=str, default="ancestral", help="ancestral or top-p")
@click.option('-num_epoch', type=int, default=1)
@click.option('-acc_steps', type=int, default=16)
@click.option('-gamma', type=float, default=1)
@click.option('-logging_selection', type=bool, default=False)
@click.option('-resume', type=bool, default=False)
@click.option('-save_step', type=int, default=125)
@click.option('-eval_step', type=int, default=40)
@click.option('-data_capacity', type=int, default=100)
@click.option('-replay_step', type=int, default=1)
@click.option('-shuffle_data', type=bool, default=False)
@click.option('-rank_strategy', type=str, default="logits")
@click.option('-on_policy_rand', type=bool, default=False)
@click.option('-run_sample_strategy', type=str, default='merged, sft, merged_sft or per_interval_merged')
@click.option('-train_baseline_model_mode', type=str, default='sft', help="merge_on_interval or sft")
@click.option('-sampling_update_step', type=int, default=1)
@click.option('-alpha', type=float, default=0.01)
@click.option('-task_type', type=str)
@click.option('-max_prompt_length', type=int, help="tldr: 512, hh: 64, harm: 256")
@click.option('-max_new_token', type=int, help="128")
@click.option('-sampling_batch_size', type=int, help="12")
@click.option('-batch_size', type=int, help="2")
@click.option('-eval_batch_size', type=int, help="256")
@click.option('-loss_type', type=str)
@click.option('-baseline_name', type=str)
@click.option('-load_ref_addr', type=str, default=None)
def main(mode, seed, api_source, start_index, end_index, num_lora, num_ret, prefix, gpu_takes_n, per_epoch_save, lr, 
         flash_attn_enable, sampling_strategy, num_epoch, acc_steps, gamma, logging_selection, resume, save_step, eval_step, 
         data_capacity, replay_step, shuffle_data, rank_strategy, on_policy_rand, run_sample_strategy, 
         train_baseline_model_mode, sampling_update_step, alpha, task_type, max_prompt_length, max_new_token, 
         sampling_batch_size, batch_size, eval_batch_size, loss_type, baseline_name, load_ref_addr):
    # ensure reproducibility with fixed seed
    if seed:
        set_seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED']=str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.mps.manual_seed(seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=True
        torch.use_deterministic_algorithms(True)
    
    if api_source == "openai":
        config.client = OpenAI()
        config.system_prompt="You are a text generation evaluater."
        config.model_type="gpt-3.5-turbo-0125"
        config.n=1
    else:
        reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
        config.reward_model, config.reward_tok = AutoModelForSequenceClassification.from_pretrained(reward_name).to('cpu'), AutoTokenizer.from_pretrained(reward_name)

    config.batch_size=batch_size
    config.eval_batch_size=eval_batch_size
    config.num_epoch=num_epoch
    config.acc_steps=acc_steps
    config.train_size=end_index-start_index
    config.prefix=prefix
    config.gpu_takes_n=gpu_takes_n
    config.max_step=int(config.num_epoch*config.train_size/(1*config.acc_steps))
    config.per_epoch_step=int(config.train_size/(1*config.acc_steps))
    config.lr=lr
    config.seed=seed
    config.gamma=gamma
    config.tokenizer=GemmaTokenizerFast.from_pretrained('google/gemma-2b')
    config.logging_selection=logging_selection
    config.chosen_policy_logp={}
    config.rejected_policy_logp={}
    config.chosen_ref_logp=0
    config.rejected_ref_logp=0
    config.chosen_rewards={}
    config.rejected_rewards={}
    config.rewards_margins={}
    config.rank_strategy=rank_strategy
    config.on_policy_rand=on_policy_rand
    config.num_lora=num_lora
    config.run_sample_strategy=run_sample_strategy
    config.dpo_inference_name="dpo_merged_inference"
    config.train_baseline_model_mode=train_baseline_model_mode
    config.alpha=alpha
    config.max_prompt_length=max_prompt_length
    config.max_new_token=max_new_token
    config.max_length=config.max_prompt_length+config.max_new_token
    config.task_type=task_type
    config.sampling_batch_size=sampling_batch_size
    config.loss_type=loss_type
    config.baseline_name=baseline_name
    
    if logging_selection:
        config.per_batch_acc=None
        config.all_rank_acc=0
    
    data_collator = DPODataCollatorWithPadding(
        pad_token_id=config.tokenizer.pad_token_id,
        label_pad_token_id=config.label_pad_token_id,
        is_encoder_decoder=False,
    )
    config.data_collator=data_collator
    accelerator = Accelerator(gradient_accumulation_steps=config.acc_steps)
    config.accelerator=accelerator
    
    if flash_attn_enable:
        model = AutoModelForCausalLM.from_pretrained('google/gemma-2b', torch_dtype=torch.bfloat16, device_map="cpu", attn_implementation="flash_attention_2").to('cuda') 
    else:
        model = AutoModelForCausalLM.from_pretrained('google/gemma-2b', torch_dtype=torch.bfloat16, device_map="cpu").to('cuda')
    
    if resume:
        model=set_up_single_lora(model, f"{prefix}/epoch_{epoch_index}_{seed}_{config.global_step_ls[i]}_shuffle_{shuffle_data}_ret_{num_ret}_sample_{run_sample_strategy}_{sampling_update_step}_{train_baseline_model_mode}_{baseline_name}/model_0", f"model_0")
        model.load_adapter(f"{prefix}/epoch_{epoch_index}_{seed}_{config.global_step_ls[i]}_shuffle_{shuffle_data}_ret_{num_ret}_sample_{run_sample_strategy}_{sampling_update_step}_{train_baseline_model_mode}_{baseline_name}/{config.sft_adapter_name}", adapter_name=config.sft_adapter_name)
    else:
        config.sft_adapter_name="sft"
        # set up one adapter                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ,ra_addr_ls = [f"xu1998hz/{i}_sft_lora_256" for i in range(num_lora)]
        if task_type == "tldr":
            model=set_up_single_lora(model, f"xu1998hz/0_sft_lora_256", f"model_0")
            # load sft lora weights for inference
            sft_lora_addr_ls = [f"xu1998hz/{i}_sft_lora_256" for i in range(num_lora)]
        elif task_type == "hh":
            model=set_up_single_lora(model, "/data/user_data/xixu/wendaxu/sft_hh_final/model_0/checkpoint-52", f"model_0")
            # load sft lora weights for inference
            sft_lora_addr_ls = [f"/data/user_data/xixu/wendaxu/sft_hh_final/model_{i}/checkpoint-52" for i in range(num_lora)]
        elif task_type == "harm":
            model=set_up_single_lora(model, "/data/user_data/xixu/wendaxu/sft_harm_final/model_0", f"model_0")
            # load sft lora weights for inference
            sft_lora_addr_ls = [f"/data/user_data/xixu/wendaxu/sft_harm_final/model_{i}" for i in range(num_lora)]
        else:
            print(f"task type {task_type} is not supported!")
            exit(1)
        
        model=set_up_merge_lora(model, sft_lora_addr_ls, config.sft_adapter_name)

        if load_ref_addr:
            # load in golden ref model
            model=set_up_single_lora(model, f"{load_ref_addr}/model_0", f"model_0_ref")
            constrained_lora_addr_ls = [f"{load_ref_addr}/model_{i}" for i in range(num_lora)]
            print(f"loading ref model from {load_ref_addr}")
            model=set_up_merge_lora_ref(model, constrained_lora_addr_ls, 'constrained_ref')
        model.print_trainable_parameters()

    # disable dropout in model
    disable_dropout_in_model(model)

    optimizer_ls = [transformers.AdamW(model.parameters(), lr=config.lr) for _ in range(num_lora)]
    lr_scheduler_ls = [transformers.get_scheduler("linear", optimizer_ls[i], num_warmup_steps=0, num_training_steps=config.max_step) for i in range(num_lora)]

    # make sure login id is right
    wandb.login(key="377341f70d62101cfbf553f4c38879b82cdae78b", relogin=True)

    wandb.init(project='DPO_Master', name=f"{mode}_{seed}_sample_{run_sample_strategy}_{sampling_update_step}_{train_baseline_model_mode}_lora_{num_lora}_{task_type}_{baseline_name}", config=\
    {
        "seed": seed,
        "mode": mode,
        "train batch size": 1 * config.acc_steps,
        "lr": config.lr,
        "beta": config.beta,
        "sample size": num_ret,
        "lora size": num_lora,
        "gamma": config.gamma,
        "relay_step": replay_step,
        "capacity": data_capacity,
        "rank_strategy": rank_strategy,
        "on_policy": on_policy_rand, 
        "run_sample_strategy": run_sample_strategy,
        "sampling_update_step": sampling_update_step,
        "train_baseline_model_mode": train_baseline_model_mode,
        "baseline_name": baseline_name,
        "prefix": prefix
    })

    # We don't reset loss or global step or reward acc
    config.tr_loss_ls = [0]*num_lora
    config.reward_acc_ls = [0]*num_lora
    config.global_step_ls = [0]*num_lora

    if not os.path.exists(f'{prefix}'):
        os.mkdir(f'{prefix}')
    # dataset needs to be dynamically updated
    with tqdm(total=config.max_step) as pbar:
        for epoch_index in range(config.num_epoch):
            model.zero_grad()

            with open(f'sft_{task_type}_data/sampling.jsonl') as f:
                sampling_prompt_dataset = [json.loads(line) for line in f]
            sampling_prompt_dataset = sampling_prompt_dataset[start_index:end_index]
            # shuffle the data order
            if shuffle_data:
                random.shuffle(sampling_prompt_dataset)

            for step, prompts in enumerate(prompt_batchify(sampling_prompt_dataset, config.acc_steps*config.step_per_feedback)):
                print("Starts online data training")
                # make sure you are inferencing on sft lora, use dpo_inference
                print("start inference")

                if config.run_sample_strategy == "merged" or config.run_sample_strategy == "merged_sft":
                    print(f"set model with {config.dpo_inference_name} for sampling")
                    model = merge_lora_for_inference(model, num_lora, config.dpo_inference_name)
                elif config.run_sample_strategy == "sft":
                    # set model in sft mode
                    if train_baseline_model_mode == "merge_on_interval":
                        if step % sampling_update_step == 0:
                            if step > 0:
                                model.delete_adapter(f"cur_{config.dpo_inference_name}")
                            # only merge lora under those steps
                            model = merge_lora_for_inference(model, num_lora, f"cur_{config.dpo_inference_name}")
                    elif train_baseline_model_mode == "ema":
                        if step % sampling_update_step == 0:
                            if step > 0:
                                model.delete_adapter(f"cur_{config.dpo_inference_name}")
                            # only merge lora under those steps
                            model = merge_lora_for_inference(model, num_lora, f"cur_{config.dpo_inference_name}")
                            if step == 0:
                                model_names=[config.sft_adapter_name, f"cur_{config.dpo_inference_name}"]
                            else:
                                model_names=["ref_model", f"cur_{config.dpo_inference_name}"]

                            model.add_weighted_adapter(
                                adapters=model_names,
                                weights=[1-config.alpha, config.alpha],
                                adapter_name="new_ref_model",
                                combination_type="linear"
                            )
                            if step > 0:
                                model.delete_adapter("ref_model")
                            model = change_lora_weight_name(model, "new_ref_model", "ref_model")
                            model.delete_adapter("new_ref_model")
                            
                    print(f"set model with {config.sft_adapter_name} for sampling")
                    model.set_adapter(config.sft_adapter_name)
                elif config.run_sample_strategy == "per_interval_merged":
                    if train_baseline_model_mode == "ema":
                        if step % sampling_update_step == 0:
                            if step > 0:
                                model.delete_adapter(f"cur_{config.dpo_inference_name}")
                            # only merge lora under those steps
                            model = merge_lora_for_inference(model, num_lora, f"cur_{config.dpo_inference_name}")
                            if step == 0:
                                model_names=[config.sft_adapter_name, f"cur_{config.dpo_inference_name}"]
                            else:
                                model_names=["ref_model", f"cur_{config.dpo_inference_name}"]

                            model.add_weighted_adapter(
                                adapters=model_names,
                                weights=[1-config.alpha, config.alpha],
                                adapter_name="new_ref_model",
                                combination_type="linear"
                            )
                            if step > 0:
                                model.delete_adapter("ref_model")
                            model = change_lora_weight_name(model, "new_ref_model", "ref_model")
                            model.delete_adapter("new_ref_model")
                    else:
                        if step % sampling_update_step == 0:
                            if step > 0:
                                model.delete_adapter(f"cur_{config.dpo_inference_name}")
                            # only merge lora under those steps
                            model = merge_lora_for_inference(model, num_lora, f"cur_{config.dpo_inference_name}")
                else:
                    print("Your sampling strategy from model is not supported!")
                    exit(1)

                prompts, first_ls, sec_ls = inference(prompts, config.tokenizer, model, sampling_strategy)
                # delete the merged adapter so that it updates
                if config.run_sample_strategy == "merged" or config.run_sample_strategy == "merged_sft":
                    model.delete_adapter(config.dpo_inference_name)
                print("finish inference")

                # begin annotations from desinated LLM
                selected_data = ret_train_data(prompts, first_ls, sec_ls)

                # we log the ranking accuracy of model for through training
                if logging_selection and (mode != "upper_bound" and mode != "lower_bound"):
                    with torch.no_grad():
                        chosen_ls, rejected_ls = compute_reward_from_models(selected_data, data_collator, model, "model_0", config.sft_adapter_name)
                    rate_ls = []
                    for c, r in zip(chosen_ls, rejected_ls):
                        if c>r:
                            rate_ls+=[0]
                        else:
                            rate_ls+=[1]
                    config.per_batch_acc = sum(rate_ls)/len(rate_ls)
                    config.all_rank_acc += sum(rate_ls)/len(rate_ls)

                # save feedback data from annotations
                first_path=f'{prefix}/online_train_data_{seed}_ret_{num_ret}_sample_{run_sample_strategy}_{sampling_update_step}_{train_baseline_model_mode}_{baseline_name}'
                if not os.path.exists(first_path):
                    os.mkdir(first_path)
                if not os.path.exists(f'{first_path}/epoch_{epoch_index}'):
                    os.mkdir(f'{first_path}/epoch_{epoch_index}')

                with open(f'{first_path}/epoch_{epoch_index}/{step}_rank.json', 'w') as f:
                    json.dump(selected_data, f)

                # sanity check the size of data, it should match with acc_step * step_per_feedback
                if len(selected_data['prompt']) != config.acc_steps * config.step_per_feedback:
                    # update gradient accumulation size with current data size
                    cur_acc_steps=int(len(selected_data['prompt'])/config.step_per_feedback)
                    print(f"Data size is not matching with online accumulation steps! {cur_acc_steps}")
                else:
                    cur_acc_steps=config.acc_steps
                
                # evaluate on the holdout set
                if config.global_step_ls[0] % eval_step == 0:    
                    model = merge_lora_for_inference(model, num_lora, config.dpo_inference_name)
                    save_name=f'{prefix}/epoch_{epoch_index}_{seed}_{config.global_step_ls[0]}_shuffle_{shuffle_data}_ret_{num_ret}_sample_{run_sample_strategy}_{sampling_update_step}_{train_baseline_model_mode}_{baseline_name}.txt'
                    win_rate, reward_diff = eval_over_steps(model, config.reward_model, config.reward_tok, config.tokenizer, save_name)
                    # delete adapter after merging
                    model.delete_adapter(config.dpo_inference_name)
                    print("Current win rate: ", win_rate)
                    print("Current reward diff: ", reward_diff)
                    wandb.log({'result': win_rate, 'reward_diff': reward_diff})

                print("start training")
                start_time=time.time()
                training_step(selected_data, data_collator, model, cur_acc_steps, optimizer_ls, lr_scheduler_ls, num_lora)
                print(f"Training time: {time.time()-start_time}")

                for i in range(num_lora):
                    if config.global_step_ls[i] % save_step == 0:
                        model.save_pretrained(f'{prefix}/epoch_{epoch_index}_{seed}_{config.global_step_ls[i]}_shuffle_{shuffle_data}_ret_{num_ret}_sample_{run_sample_strategy}_{sampling_update_step}_{train_baseline_model_mode}_{baseline_name}')
                        # save optimizer and lr scheduler
                        torch.save(optimizer_ls[i].state_dict(), f'{prefix}/epoch_{epoch_index}_{seed}_{config.global_step_ls[i]}_shuffle_{shuffle_data}_ret_{num_ret}_sample_{run_sample_strategy}_{sampling_update_step}_{train_baseline_model_mode}_{baseline_name}/optimizer_{i}.pt')
                        torch.save(lr_scheduler_ls[i].state_dict(), f'{prefix}/epoch_{epoch_index}_{seed}_{config.global_step_ls[i]}_shuffle_{shuffle_data}_ret_{num_ret}_sample_{run_sample_strategy}_{sampling_update_step}_{train_baseline_model_mode}_{baseline_name}/scheduler_{i}.pt')
                pbar.update(1)
            
            # save at end of epoch
            if per_epoch_save:
                for i in range(num_lora):
                    model.save_pretrained(f'{prefix}/epoch_{epoch_index}_{seed}_{config.global_step_ls[i]}_shuffle_{shuffle_data}_ret_{num_ret}_sample_{run_sample_strategy}_{sampling_update_step}_{train_baseline_model_mode}_{baseline_name}')
                    # save optimizer and lr scheduler
                    torch.save(optimizer_ls[i].state_dict(), f'{prefix}/epoch_{epoch_index}_{seed}_{config.global_step_ls[i]}_shuffle_{shuffle_data}_ret_{num_ret}_sample_{run_sample_strategy}_{sampling_update_step}_{train_baseline_model_mode}_{baseline_name}/optimizer_{i}.pt')
                    torch.save(lr_scheduler_ls[i].state_dict(), f'{prefix}/epoch_{epoch_index}_{seed}_{config.global_step_ls[i]}_shuffle_{shuffle_data}_ret_{num_ret}_sample_{run_sample_strategy}_{sampling_update_step}_{train_baseline_model_mode}_{baseline_name}/scheduler_{i}.pt')
                print("Save at end of epoch!")
            
if __name__ == "__main__":
    main()