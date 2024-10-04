import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

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

num_lora=8
model = AutoModelForCausalLM.from_pretrained('google/gemma-2b', torch_dtype=torch.bfloat16, device_map="cpu", attn_implementation="flash_attention_2").to("cuda")
inference_adapter_name="sft"
# set up one adapter
model=set_up_single_lora(model, f"xu1998hz/0_sft_lora", f"model_0")
# load sft lora weights for inference
sft_lora_addr_ls = [f"xu1998hz/{i}_sft_lora" for i in range(num_lora)]
model=set_up_merge_lora(model, sft_lora_addr_ls, inference_adapter_name)

for model_index in range(0, num_lora):
    model.set_adapter(f"model_{model_index}")
    for name, param in model.named_parameters():
        if f"model_{model_index}" in name:
            if param.requires_grad!=True:
                print(param.requires_grad)
