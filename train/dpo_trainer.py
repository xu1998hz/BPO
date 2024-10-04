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
                    inputs = self.tokenizer(example['prompt'], return_tensors="pt", truncation=True, max_length=1024).to(args.device)
                    outputs = self.model.generate(**inputs, max_new_tokens=512, num_return_sequences=1, do_sample=False)
                    predicted_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    f.write(predicted_text)
                    f.write('\n')
                    f.write('='*70)
                    f.write('\n')
                    f.flush()
                
        self.model.train()

@click.command()
@click.option('-w_index', type=int)
@click.option('-mode', type=str)
@click.option('-seed', type=int) # 42
def main(w_index, mode, seed):
    class config:
        strategy="dpo_trainer_original_check_bugs_april17"
        train_size=1848
        vali_size=100
        num_epoch=5
        batch_size=1
        acc_steps=16
        lr=5e-5
        ds_config="config/ds_config_zero3.json"
        model_name=f"xu1998hz/sft_{w_index}" # "mistralai/Mistral-7B-v0.1"
        ref_model_name=f"xu1998hz/sft"
        weight_index=w_index
        max_length=512+128 
        max_prompt_length=512
    
    config.max_step=int(config.num_epoch*config.train_size/(config.batch_size*config.acc_steps))
    config.model_seed_index=seed+int(config.weight_index)
    config.data_seed_index=config.model_seed_index

    model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.bfloat16, device_map="auto")
    ref_model = AutoModelForCausalLM.from_pretrained(config.ref_model_name, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b')

    training_args = TrainingArguments(
        report_to="none",
        # report_to="wandb", # enables logging to W&B 😎
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.lr,
        lr_scheduler_type="cosine",
        # num_train_epochs=config.num_epoch,
        max_steps=125,
        gradient_accumulation_steps=1, # simulate larger batch sizes
        output_dir="output", # f"/mnt/data6/wendaxu/dpo_trainer_weights/{config.weight_index}_dpo_{mode}_{seed}",
        save_strategy="no",
        # save_steps=0.2,
        evaluation_strategy="no", 
        # eval_steps=0.2,
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=1,
        weight_decay=0,
        warmup_ratio=0,
        seed=config.model_seed_index,
        data_seed=config.data_seed_index
        # deepspeed=config.ds_config,
    )

    selected_data = json.load(open(f"online_train_data_43_rand_rand/0_rank.json"))
    # train_data = {key: data[key][:len(data[key])-config.vali_size] for key in data}
    selected_data['prompt']=[selected_data['prompt'][0]]
    selected_data['chosen']=[selected_data['chosen'][0].replace(selected_data['prompt'][0], '')]
    selected_data['rejected']=[selected_data['rejected'][0].replace(selected_data['prompt'][0], '')]
    # rejected_ls, chosen_ls = [], []
    # for prompt, reject, chosen in zip([selected_data['prompt'][0]], [selected_data['rejected'][0]], [selected_data['chosen'][0]]):
    #     chosen_ls+=[chosen.replace(prompt, '').strip()]
    #     rejected_ls+=[reject.replace(prompt, '').strip()]
    #     print(chosen.replace(prompt, '').strip())
    #     print(reject.replace(prompt, '').strip())
    #     print()
    # selected_data['rejected']=rejected_ls
    # selected_data['chosen']=chosen_ls
    train_feedback_data = Dataset.from_dict(selected_data)
    print(selected_data)
    # vali_data = {key: data[key][-config.vali_size:] for key in data}
    # vali_feedback_data = Dataset.from_dict(vali_data)
    # raw_eval_dataset=copy.deepcopy(vali_feedback_data)
    
    dpo_trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        beta=0.5,
        train_dataset=train_feedback_data,
        # eval_dataset=vali_feedback_data,
        tokenizer=tokenizer,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
        # callbacks=[SaveEvalOutputsCallback(raw_eval_dataset, model, tokenizer, output_dir=f'sample_output/{mode}', num_samples=20)],
    )

    # wandb.init(project='DPO_Master', name=f"{mode}_{config.strategy}_{config.model_name}_{config.weight_index}", config=\
    # {
    #     "strategy": config.strategy,
    #     "epoch": config.num_epoch,
    #     "train batch size": config.batch_size * config.acc_steps,
    #     "lr": config.lr,
    #     "model_name": config.model_name,
    #     "weight_index": config.weight_index,
    #     "model_seed_index": config.model_seed_index,
    #     "data_seed_index": config.data_seed_index,
    #     "setting": mode
    # })

    # we then continue as regular
    train_result = dpo_trainer.train() 
    # good practice to end your run
    # wandb.finish()

    # dpo_trainer.save_model() 
    # metrics = train_result.metrics
    # dpo_trainer.log_metrics("train", metrics)
    # dpo_trainer.save_metrics("train", metrics)

if __name__ == "__main__":
    main()  