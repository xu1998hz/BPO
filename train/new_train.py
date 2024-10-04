import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig # , get_peft_model
from trl import SFTTrainer, trainer
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from wandb.keras import WandbCallback
from tqdm import tqdm
import click
import os
from transformers import TrainerCallback
import numpy as np
import copy

f = 'data/train_sample.json'
KEY_INSTANCES = "instances"
path_prefix = ''
path_prefix = '/mnt/taurus/home/guangleizhu/instructscore_spanish/'

def formatting_prompts_func(example):
    # text = f"SUBREDDIT: {example['subreddit']}\nPOST: {example['content']}\nPlease summarize the post by given subreddit: {example['summary']}"
    text = f"{example['input']}\n{example['output']}"
    return text 

class SaveEvalOutputsCallback(TrainerCallback):
    def __init__(self, eval_dataset, model, tokenizer, output_dir="/ft_sample", num_samples=20):
        self.eval_dataset = eval_dataset
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.num_samples = num_samples

    def on_evaluate(self, args, state, control, **kwargs):
        # Randomly select num_samples indices from the evaluation dataset
        indices = np.random.choice(len(self.eval_dataset), self.num_samples, replace=False)

        self.model.eval()
        with torch.no_grad():
            with open(os.path.join(self.output_dir, f'epoch_{state.epoch}.txt'), "w") as f:
                for idx in indices:
                    # f.write('1')
                    # continue
                    example = self.eval_dataset[idx]
                    inputs = self.tokenizer(example['input'], return_tensors="pt", truncation=True, max_length=1024).to(args.device)
                    outputs = self.model.generate(**inputs, max_new_tokens=512, num_return_sequences=1, do_sample=False)
                    predicted_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    f.write(predicted_text)
                    f.write('\n')
                    f.write('='*70)
                    f.write('\n')
                    f.flush()
                
        self.model.train()

def main():
    class config:
        strategy="sft"
        train_size=9000
        num_epoch=10
        num_gpus=4
        batch_size=1
        acc_steps=16
        lr=2e-5
        ds_config="config/old_ds_config_zero3.json"
        model_name="mistralai/Mistral-7B-v0.1" # "mistralai/Mistral-7B-v0.1"

    config.max_step=int(config.num_epoch*config.train_size/(config.batch_size*config.acc_steps)/config.num_gpus)
    config.model_seed_index = 42
    config.data_seed_index = config.model_seed_index

    model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # model = get_peft_model(model, peft_config)
    # # trainable params: 54,525,952 || all params: 7,296,258,048 || trainable%: 0.7473139195638273
    # model.print_trainable_parameters()

    training_args = TrainingArguments(
        report_to="wandb", # enables logging to W&B 😎
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.lr,
        lr_scheduler_type="cosine",
        num_train_epochs=config.num_epoch,
        # max_steps=config.max_step,
        gradient_accumulation_steps=config.acc_steps, # simulate larger batch sizes
        output_dir=os.path.join(path_prefix, "new_ft"),
        # save_strategy="no",
        save_total_limit=7,
        save_strategy="epoch",
        # save_steps=0.2,
        evaluation_strategy="epoch", # "steps",
        # eval_steps=0.2,
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=1,
        weight_decay=0,
        warmup_ratio=0,
        seed=config.model_seed_index,
        data_seed=config.data_seed_index,
        deepspeed=config.ds_config,
        run_name="instructscore_spanish_mistral",
        # bf16=True,
    )

    extensions = "json"
    data = load_dataset(
        extensions,
        data_files=[f],
        field=KEY_INSTANCES,
        split="train",
        use_auth_token=None,
    )

    train_dataset=[data[i] for i in range(0, config.train_size, 1)]
    eval_dataset=[data[i] for i in range(config.train_size, len(data), 1)]
    # train_dataset=[data[i] for i in range(0, 300, 1)]
    # eval_dataset=[data[i] for i in range(300, 600, 1)]
    raw_eval_dataset = copy.deepcopy(eval_dataset)
    print(len(train_dataset), len(eval_dataset))

    train_dataset = trainer.ConstantLengthDataset(
        tokenizer,
        train_dataset,
        formatting_func=formatting_prompts_func,
        seq_length=1024,
    )
    eval_dataset = trainer.ConstantLengthDataset(
        tokenizer,
        eval_dataset,
        formatting_func=formatting_prompts_func,
        seq_length=1024,
    )

    # wandb.init(project='DPO_Master', name=f"{config.strategy}_{config.model_name}_{config.model_seed_index}_{config.data_seed_index}", config=\
    #     {
    #         "strategy": config.strategy,
    #         "epoch": config.num_epoch,
    #         # "train batch size": config.batch_size * config.acc_steps,
    #         "lr": config.lr,
    #         "model_name": config.model_name,
    #         "model_seed_index": config.model_seed_index,
    #         "data_seed_index": config.data_seed_index,
    #     })

    our_trainer = SFTTrainer(
        model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        packing=True, # pack samples together for efficient training
        max_seq_length=1024, # maximum packed length 
        args=training_args,
        formatting_func=formatting_prompts_func, # format samples with a model schema
        callbacks=[SaveEvalOutputsCallback(raw_eval_dataset, model, tokenizer, output_dir=os.path.join(path_prefix, "ft_sample"), num_samples=5)],
    )

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