from trl import DPOTrainer
import torch
import wandb
from datasets import Dataset
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
import click
import json
import transformers
from typing import TypeVar, Iterable, List
import math

T = TypeVar('T')

"""This file is used to debug the training pipeline:
1) Investigate whether DPO is able """

def prompt_batchify(data: Iterable[T], batch_size: int) -> Iterable[List[T]]:
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

@click.command()
@click.option('-mode', type=str)
@click.option('-seed', type=int) # 42
@click.option('-gpu_index', type=int)
@click.option('-step_per_f', type=int)
def main(mode, seed, gpu_index, step_per_f):
    class config:
        strategy="debug_dpo_iterative"
        step_per_feedback=step_per_f
        vali_size=100
        num_epoch=5
        batch_size=1
        acc_steps=8
        lr=5e-5
        num_query=1
        ds_config="config/ds_config_zero3.json"
        model_ref_name="xu1998hz/sft" # "test_out/0_dpo_rand_None_0_sec/checkpoint-5" # "mistralai/Mistral-7B-v0.1"
        num_weights=7
        gpu_index_ls=[0, 1, 2, 3, 4, 5, 6]
        max_length=1152 
        max_prompt_length=1024
        
    
    config.model_name=f"xu1998hz/sft_{gpu_index}"
    config.gpu_index=gpu_index
    config.model_seed_index=seed+gpu_index
    config.data_seed_index=config.model_seed_index

    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.bfloat16, device_map="cpu").to('cuda')
    torch.cuda.empty_cache()
    model_ref = AutoModelForCausalLM.from_pretrained(config.model_ref_name, torch_dtype=torch.bfloat16, device_map="cpu").to('cuda')
    
    tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b')

    wandb.init(project='DPO_Master', name=f"{mode}_{config.strategy}_{config.model_name}_{config.gpu_index}", config=\
    {
        "strategy": config.strategy,
        "epoch": config.num_epoch,
        "train batch size": config.batch_size * config.acc_steps,
        "lr": config.lr,
        "model_name": config.model_name,
        "gpu_index": config.gpu_index,
        "model_seed_index": config.model_seed_index,
        "data_seed_index": config.data_seed_index,
        "setting": mode
    })

    data = json.load(open('dpo_data/rand_None_rank_0.json'))

    # define my own lr scheduler
    optimizer = transformers.AdamW(model.parameters(), lr=config.lr)
    lr_scheduler = transformers.get_scheduler("cosine", optimizer, num_warmup_steps=0, num_training_steps=len(data['prompt'])*config.num_epoch/(config.batch_size*config.acc_steps))

    step = 0
    for _ in range(config.num_epoch):
        for _, (prompts, chosens, rejecteds) in enumerate(zip(prompt_batchify(data['prompt'], config.acc_steps*config.step_per_feedback), \
                                            prompt_batchify(data['chosen'], config.acc_steps*config.step_per_feedback), \
                                                prompt_batchify(data['rejected'], config.acc_steps*config.step_per_feedback))):
            # prompts, chosens, rejecteds = data['prompt'][:16], data['chosen'][:16], data['rejected'][:16]
            # generate training data on the fly
            selected_data = {'prompt': prompts, 'chosen': chosens, 'rejected': rejecteds}

            print("Begin training")

            train_feedback_data = Dataset.from_dict(selected_data)

            torch.cuda.empty_cache()
            if step != 0:
                model = AutoModelForCausalLM.from_pretrained(f'/mnt/data6/wendaxu/{config.strategy}_{gpu_index}_weight_seed_dpo_{mode}_data_seed_{seed}_updated/checkpoint-{cur_max_step}', torch_dtype=torch.bfloat16, device_map="cpu").to('cuda')
            
            # update current max step after loading weight
            cur_max_step = math.ceil(len(prompts)/(config.acc_steps))

            training_args = TrainingArguments(
                report_to=None, # don't report to wandb through trainer
                per_device_train_batch_size=config.batch_size,
                max_steps=cur_max_step,
                gradient_accumulation_steps=config.acc_steps, # simulate larger batch sizes
                output_dir=f"/mnt/data6/wendaxu/{config.strategy}_{gpu_index}_{mode}_seed_{seed}_max_step_{cur_max_step}",
                save_strategy="steps",
                save_steps=cur_max_step, 
                logging_strategy="steps",
                logging_first_step=True,
                logging_steps=1,
                weight_decay=0,
                warmup_ratio=0,
                seed=config.model_seed_index,
                data_seed=config.data_seed_index,
                remove_unused_columns=False
                # deepspeed=config.ds_config,
            )

            dpo_trainer = DPOTrainer(
                model=model,
                ref_model=model_ref,
                args=training_args,
                beta=0.5,
                train_dataset=train_feedback_data,
                # eval_dataset=vali_feedback_data,
                tokenizer=tokenizer,
                max_length=config.max_length,
                max_prompt_length=config.max_prompt_length
                # callbacks=[SaveEvalOutputsCallback(raw_eval_dataset, model, tokenizer, output_dir=f'sample_output/{mode}', num_samples=20)]
            )
            
            # dynamic pass lr and optimizer into the model
            dpo_trainer.lr_scheduler = lr_scheduler
            
            # we then continue as regular
            if step == 0:
                train_results = dpo_trainer.train()
            else:
                train_results = dpo_trainer.train(resume_from_checkpoint=True)
            
            wandb.log({"Training loss": train_results.training_loss}, step=step)
            print(train_results)

if __name__ == "__main__":
    main()  