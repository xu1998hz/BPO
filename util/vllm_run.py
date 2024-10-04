from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import torch
from peft import PeftModel
import json

mode="vllm" # either "tr" or "vllm"
print(mode)
data = json.load(open('oom.json'))

if mode == "vllm":
    llm = LLM(model='google/gemma-2b', enable_lora=True)
    start=time.time()
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.95,
        top_k=-1,
        max_tokens=128,
        truncate_prompt_tokens=512
        # stop=["[/assistant]"]
    )
    for prompt in data['prompt']:
        print(prompt)
        outputs = llm.generate(
            [prompt]*12,
            sampling_params,
            lora_request=LoRARequest("sft_0", 1, "/share/edc/home/wendaxu/0_sft_lora/checkpoint-800")
        )
        print(outputs)
else:
    tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b')
    model = AutoModelForCausalLM.from_pretrained('google/gemma-2b', torch_dtype=torch.bfloat16, device_map="cpu", attn_implementation="flash_attention_2").to("cuda")
    model = PeftModel.from_pretrained(
        model,
        "xu1998hz/0_sft_lora",
        is_trainable=True,
        adapter_name='sft_lora_0',
    )
    model.set_adapter('sft_lora_0')
    start=time.time()
    for prompt in data['prompt']:
        inputs = tokenizer([prompt], return_tensors="pt", padding=True, truncation=True, max_length=512, add_special_tokens=False).to(model.device) 
        output = model.generate(inputs=inputs.input_ids, max_new_tokens=256, do_sample=True, temperature=1.0, top_k=0, top_p=0.95, num_return_sequences=12)
print(time.time()-start)