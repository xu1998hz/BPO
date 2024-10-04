import torch
import wandb
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, trainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from transformers import TrainerCallback
from tqdm import tqdm
import click
import os
import numpy as np
import copy
import json
import random
from transformers import set_seed

class config:
    acc_steps=16
    lr=5e-5
    ds_config="config/ds_config_zero3.json"
    model_name="google/gemma-2b" 

def formatting_prompts_general_func(example):
    text = f"{example['prompt']}{example['label']}"
    return text 

def formatting_prompts_for_inference(example):
    text = f"{example['prompt']}"
    return text 

def formatting_prompts_hh_for_inference(example):
    text = f"{example['prompt']} Assistant: "
    return text 

def formatting_prompts_hh_func(example):
    text = f"{example['prompt']} Assistant: {example['label']}"
    return text 

class SaveEvalOutputsCallback(TrainerCallback):
    def __init__(self, model, eval_dataset, tokenizer, output_dir): # 
        self.eval_dataset = eval_dataset
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir

    def on_evaluate(self, args, state, control, **kwargs):
        # Randomly select num_samples indices from the evaluation dataset
        self.model.eval()
        with torch.no_grad():
            with open(os.path.join(self.output_dir, f'epoch_{state.epoch}_{state.global_step}.txt'), "w") as f:
                for example in self.eval_dataset:
                    if config.task_type=="tldr" or config.task_type=="harm":
                        inputs = self.tokenizer(formatting_prompts_for_inference(example), return_tensors="pt", truncation=True, max_length=config.max_inp_length).to(args.device)
                    elif config.task_type=="hh":
                        inputs = self.tokenizer(formatting_prompts_hh_for_inference(example), return_tensors="pt", truncation=True, max_length=config.max_inp_length).to(args.device)
                    else:
                        print("We don't support this task!")
                        exit(1)
                    outputs = self.model.generate(inputs['input_ids'], max_new_tokens=config.max_tar_length, num_return_sequences=1, do_sample=False, temperature=0)
                    predicted_text = self.tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
                    f.write(predicted_text.replace('\n','')+'\n')
                    f.flush()
                
        self.model.train()

@click.command()
@click.option('-seed', type=int, default=None)
@click.option('-peft_enable', type=bool, default=True)
@click.option('-rank', type=int, default=256)
@click.option('-batch_size', type=int, default=4)
@click.option('-max_inp_length', type=int)
@click.option('-max_tar_length', type=int)
@click.option('-train_size', type=int, default=10000)
@click.option('-num_epoch', type=int, default=5)
@click.option('-prefix', type=str, default="/data/user_data/xixu/wendaxu")
@click.option('-flash_attn_enable', type=bool, default=False)
@click.option('-task_type', type=str)
def main(seed, peft_enable, rank, max_inp_length, max_tar_length, batch_size, train_size, num_epoch, prefix, flash_attn_enable, task_type):
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
    
    config.batch_size=batch_size
    if peft_enable:
        config.strategy=f"sft_lora_{rank}_{task_type}"
    else:
        config.strategy=f"sft_full_{task_type}"

    config.max_inp_length=max_inp_length
    config.max_tar_length=max_tar_length
    config.num_epoch=num_epoch
    config.seed=seed
    config.model_seed_index=seed
    config.data_seed_index=config.model_seed_index
    config.train_size=train_size
    config.prefix=prefix
    config.task_type=task_type
    config.max_step=int(config.num_epoch*config.train_size/(config.batch_size*config.acc_steps))

    if peft_enable:
        peft_config = LoraConfig(
            r=rank,  # the rank of the LoRA matrices
            lora_alpha=rank*2, # the weight
            lora_dropout=0.1, # dropout to add to the LoRA layers
            bias="none", # add bias to the nn.Linear layers?
            task_type="CAUSAL_LM",
            target_modules="all-linear", 
            modules_to_save=None, # layers to unfreeze and train from the original pre-trained model
        )

    if flash_attn_enable:
        model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.bfloat16, device_map="cpu", attn_implementation="flash_attention_2")
    else:
        model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.bfloat16, device_map="cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if task_type=="tldr":
        response_template="TL;DR:"
    elif task_type=="harm":
        response_template=" assistant:"
    elif task_type=="hh":
        response_template=" Assistant:"
    else:
        print("We don't support this task!")
        exit(1)
        
    # ignore the prompt
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    if peft_enable:
        model = get_peft_model(model, peft_config)
        # trainable params: 1,843,200 || all params: 2,508,015,616 || trainable%: 0.073492365368111
        model.print_trainable_parameters()
        model.config.use_cache=False

    training_args = TrainingArguments(
        report_to="wandb", # enables logging to W&B 😎
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.lr,
        lr_scheduler_type="cosine",
        max_steps=config.max_step,
        save_total_limit=config.num_epoch,
        gradient_accumulation_steps=config.acc_steps, # simulate larger batch sizes
        output_dir=f"{config.prefix}/sft_{seed}_{config.strategy}",
        save_strategy="steps",
        save_steps=1/config.num_epoch, # save five checkpoint corresponding to each epoch
        evaluation_strategy="steps",
        eval_steps=1/config.num_epoch/2,
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=1,
        weight_decay=0,
        warmup_ratio=0,
        seed=config.model_seed_index,
        data_seed=config.data_seed_index
        # deepspeed=config.ds_config,
    )

    # load in train and validation set
    with open(f'sft_{task_type}_data/train.jsonl') as f:
        train_dataset = [json.loads(line) for line in f]
    
    with open(f'sft_{task_type}_data/dev.jsonl') as f:
        eval_dataset = [json.loads(line) for line in f]
        eval_20_dataset = copy.deepcopy(eval_dataset[:20]) 

    if task_type=="tldr" or task_type=="harm":
        formatting_prompts_func=formatting_prompts_general_func
    elif task_type=="hh":
        formatting_prompts_func=formatting_prompts_hh_func
    else:
        print("We currently don't support this task!")
        exit(1)

    train_dataset = trainer.ConstantLengthDataset(
        tokenizer,
        train_dataset,
        formatting_func=formatting_prompts_func,
        seq_length=config.max_inp_length+config.max_tar_length,
    )
    eval_dataset = trainer.ConstantLengthDataset(
        tokenizer,
        eval_dataset,
        formatting_func=formatting_prompts_func,
        seq_length=config.max_inp_length+config.max_tar_length,
    )

    if peft_enable:
        our_trainer = SFTTrainer(
            model,
            peft_config=peft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            packing=False, # pack samples together for efficient training
            max_seq_length=config.max_inp_length+config.max_tar_length, # maximum packed length 
            args=training_args,
            data_collator=collator,
            infinite=True,
            formatting_func=formatting_prompts_func, # format samples with a model schema
            callbacks=[SaveEvalOutputsCallback(model, eval_20_dataset, tokenizer, output_dir=f'{config.prefix}/sft_{seed}_{config.strategy}')],
        )
    else:
        our_trainer = SFTTrainer(
            model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            packing=False, # pack samples together for efficient training
            max_seq_length=config.max_inp_length+config.max_tar_length, # maximum packed length 
            args=training_args,
            data_collator=collator,
            infinite=True,
            formatting_func=formatting_prompts_func, # format samples with a model schema
            callbacks=[SaveEvalOutputsCallback(model, eval_20_dataset, tokenizer, output_dir=f'{config.prefix}/sft_{seed}_{config.strategy}')],
        )

    wandb.init(project='DPO_Master', name=f"{config.model_name}_sft_{seed}_{config.strategy}", config=\
        {
            "strategy": config.strategy,
            "epoch": config.num_epoch,
            "train batch size": config.batch_size * config.acc_steps,
            "lr": config.lr,
            "model_name": config.model_name,
            "model_seed_index": config.model_seed_index,
            "data_seed_index": config.data_seed_index,
        })

    # we then continue as regular
    train_result = our_trainer.train() 
    # good practice to end your run
    wandb.finish()

    our_trainer.save_model() 
    metrics = train_result.metrics
    our_trainer.log_metrics("train", metrics)
    our_trainer.save_metrics("train", metrics)

if __name__ == "__main__":
    main()  