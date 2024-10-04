from transformers import AutoModelForCausalLM
import torch

# for i in range(0, 8):
#     ckpt=f"/share/edc/home/wendaxu/dpo/{i}_dpo_uncertainty_lcb/checkpoint-219"
#     model = AutoModelForCausalLM.from_pretrained(ckpt, use_safetensors=True, torch_dtype=torch.bfloat16, trust_remote_code=True) #device_map="auto", 
#     model.push_to_hub(f"xu1998hz/ucb_lcb_{i}")
#     print(f"{ckpt} is uploaded!")

# for i in range(0, 8):
#     ckpt=f"/share/edc/home/wendaxu/dpo/{i}_dpo_uncertainty_mean_ucb/checkpoint-218"
#     model = AutoModelForCausalLM.from_pretrained(ckpt, use_safetensors=True, torch_dtype=torch.bfloat16, trust_remote_code=True) #device_map="auto", 
#     model.push_to_hub(f"xu1998hz/ucb_mean_ucb_{i}")
#     print(f"{ckpt} is uploaded!")

# for i in range(0, 8):
#     ckpt=f"/share/edc/home/wendaxu/dpo/{i}_dpo_uncertainty_rand/checkpoint-218"
#     model = AutoModelForCausalLM.from_pretrained(ckpt, use_safetensors=True, torch_dtype=torch.bfloat16, trust_remote_code=True) #device_map="auto", 
#     model.push_to_hub(f"xu1998hz/ucb_rand_{i}")
#     print(f"{ckpt} is uploaded!")

# for i in range(8):
#     model = AutoModelForCausalLM.from_pretrained(f"xu1998hz/rand_rand_dpo_{i}")
#     print("Downloaded!")

# for i in range(8):
#     model = AutoModelForCausalLM.from_pretrained(f"xu1998hz/ucb_rand_{i}")
#     print("Downloaded!")

# for i in range(8):
#     model = AutoModelForCausalLM.from_pretrained(f"xu1998hz/ucb_lcb_{i}")
#     print("Downloaded!")

# for i in range(8):
#     model = AutoModelForCausalLM.from_pretrained(f"xu1998hz/ucb_mean_ucb_{i}")
#     print("Downloaded!")

# for i in range(8):
#     model = AutoModelForCausalLM.from_pretrained(f"xu1998hz/ucb_sec_ucb_dpo_{i}")
#     print("Downloaded!")

# for i in range(0, 8):
#     ckpt=f"/share/edc/home/wendaxu/dpo/{i}_sec/checkpoint-1560"
#     model = AutoModelForCausalLM.from_pretrained(ckpt, use_safetensors=True, torch_dtype=torch.bfloat16, trust_remote_code=True) #device_map="auto", 
#     model.push_to_hub(f"xu1998hz/sft_{i}")
#     print(f"{ckpt} is uploaded!")

# model = AutoModelForCausalLM.from_pretrained(f"xu1998hz/sft")
# print("Downloaded!")

# for i in range(0, 8):
#     ckpt=f"/share/edc/home/wendaxu/greedy_search_{i}_sft_lora_256_all_linear_True_fixed_april_29/checkpoint-3057"
#     model = AutoModelForCausalLM.from_pretrained(ckpt, use_safetensors=True, torch_dtype=torch.bfloat16, trust_remote_code=True) #device_map="auto", 
#     model.push_to_hub(f"xu1998hz/{i}_sft_lora_256")
#     print(f"{ckpt} is uploaded!")

for i in range(8):
    model = AutoModelForCausalLM.from_pretrained(f"xu1998hz/{i}_sft_lora_256")
    print("Downloaded!")