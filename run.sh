# seed=46
# CUDA_VISIBLE_DEVICES=3 OPENAI_API_KEY="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" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 train/dpo_lora_train.py -mode rand_rand -seed ${seed} -start_index 0 -end_index 2000 -num_ret 12 -save_step 5 -api_source openai -prefix /share/edc/home/wendaxu/ -num_lora 8 -gpu_takes_n 6 > rand_rand_${seed}.out 2>&1 &
# CUDA_VISIBLE_DEVICES=4 OPENAI_API_KEY="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" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 train/dpo_lora_train.py -mode ucb_sec_ucb -seed ${seed} -start_index 0 -end_index 2000 -num_ret 12 -save_step 5 -api_source openai -prefix /share/edc/home/wendaxu/ -num_lora 8 -gpu_takes_n 6 > ucb_sec_ucb_${seed}.out 2>&1 &
# CUDA_VISIBLE_DEVICES=5 OPENAI_API_KEY="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" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 train/dpo_lora_train.py -mode ucb_lcb -seed ${seed} -start_index 0 -end_index 2000 -num_ret 12 -save_step 5 -api_source openai -prefix /share/edc/home/wendaxu/ -num_lora 8 -gpu_takes_n 6 > ucb_lcb_${seed}.out 2>&1 &
# CUDA_VISIBLE_DEVICES=6 OPENAI_API_KEY="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" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 train/dpo_lora_train.py -mode ucb_mean_ucb -seed ${seed} -start_index 0 -end_index 2000 -num_ret 12 -save_step 5 -api_source openai -prefix /share/edc/home/wendaxu/ -num_lora 8 -gpu_takes_n 6 > ucb_mean_ucb_${seed}.out 2>&1 &
# CUDA_VISIBLE_DEVICES=7 OPENAI_API_KEY="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" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 train/dpo_lora_train.py -mode ucb_rand -seed ${seed} -start_index 0 -end_index 2000 -num_ret 12 -save_step 5 -api_source openai -prefix /share/edc/home/wendaxu/ -num_lora 8 -gpu_takes_n 6 > ucb_rand_${seed}.out 2>&1 &


# train in an offline way
# seed=45
# CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 train/debug_loss_converge.py -mode "ucb_mean_ucb" -seed "${seed}" -max_step 125 -num_lora 8 -prefix /share/edc/home/wendaxu/ -openai False > "0.out" 2>&1 &
# CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 train/debug_loss_converge.py -mode "rand_rand" -seed "${seed}" -max_step 125 -num_lora 8 -prefix /share/edc/home/wendaxu/ -openai False > "1.out" 2>&1 &
# CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 train/debug_loss_converge.py -mode "ucb_rand" -seed "${seed}" -max_step 125 -num_lora 8 -prefix /share/edc/home/wendaxu/ -openai False > "2.out" 2>&1 &
# CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 train/debug_loss_converge.py -mode "ucb_sec_ucb" -seed "${seed}" -max_step 125 -num_lora 8 -prefix /share/edc/home/wendaxu/ -openai False > "3.out" 2>&1 &
# CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 train/debug_loss_converge.py -mode "ucb_lcb" -seed "${seed}" -max_step 125 -num_lora 8 -prefix /share/edc/home/wendaxu/ -openai False > "4.out" 2>&1 &

# for i in "6" "7" # "3" "4" "5"
# do
#     CUDA_VISIBLE_DEVICES="${i}" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 inference/inference_adapter.py -ckpt /share/edc/home/wendaxu/greedy_search_0_sft_lora_256_all_linear_True_fixed_april_29/checkpoint-3057 -prefix "sampling_${i}_3" -data_split offline > "${i}.out" 2>&1 &
# done


# login-4
# seed=42
# CUDA_VISIBLE_DEVICES=0 OPENAI_API_KEY="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" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 train/online_dpo_trainer.py -mode ucb_sec_ucb -seed "${seed}" -api_source openai -start_index 0 -end_index 2000 -num_lora 8 -num_ret 12 -gpu_takes_n 12 -prefix /data/user_data/xixu/wendaxu -per_epoch_save True -lr 5e-5 -flash_attn_enable False -sampling_strategy ancestral -num_epoch 2 -acc_steps 16 > ucb_sec_ucb_${seed}.out 2>&1 &
# CUDA_VISIBLE_DEVICES=1 OPENAI_API_KEY="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" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 train/online_dpo_trainer.py -mode ucb_lcb -seed "${seed}" -api_source openai -start_index 0 -end_index 2000 -num_lora 8 -num_ret 12 -gpu_takes_n 12 -prefix /data/user_data/xixu/wendaxu -per_epoch_save True -lr 5e-5 -flash_attn_enable False -sampling_strategy ancestral -num_epoch 2 -acc_steps 16 > ucb_lcb_${seed}.out 2>&1 &
# CUDA_VISIBLE_DEVICES=2 OPENAI_API_KEY="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" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 train/online_dpo_trainer.py -mode ucb_mean_ucb -seed "${seed}" -api_source openai -start_index 0 -end_index 2000 -num_lora 8 -num_ret 12 -gpu_takes_n 12 -prefix /data/user_data/xixu/wendaxu -per_epoch_save True -lr 5e-5 -flash_attn_enable False -sampling_strategy ancestral -num_epoch 2 -acc_steps 16 > ucb_mean_ucb_${seed}.out 2>&1 &
# CUDA_VISIBLE_DEVICES=3 OPENAI_API_KEY="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" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 train/online_dpo_trainer.py -mode ucb_rand -seed "${seed}" -api_source openai -start_index 0 -end_index 2000 -num_lora 8 -num_ret 12 -gpu_takes_n 12 -prefix /data/user_data/xixu/wendaxu -per_epoch_save True -lr 5e-5 -flash_attn_enable False -sampling_strategy ancestral -num_epoch 2 -acc_steps 16 > ucb_rand_${seed}.out 2>&1 &
# login-1
# seed=43
# CUDA_VISIBLE_DEVICES=0 OPENAI_API_KEY="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" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 train/online_dpo_trainer.py -mode ucb_sec_ucb -seed "${seed}" -api_source openai -start_index 0 -end_index 2000 -num_lora 8 -num_ret 12 -gpu_takes_n 12 -prefix /data/user_data/xixu/wendaxu -per_epoch_save True -lr 5e-5 -flash_attn_enable False -sampling_strategy ancestral -num_epoch 2 -acc_steps 16 > ucb_sec_ucb_${seed}.out 2>&1 &
# CUDA_VISIBLE_DEVICES=1 OPENAI_API_KEY="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" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 train/online_dpo_trainer.py -mode ucb_lcb -seed "${seed}" -api_source openai -start_index 0 -end_index 2000 -num_lora 8 -num_ret 12 -gpu_takes_n 12 -prefix /data/user_data/xixu/wendaxu -per_epoch_save True -lr 5e-5 -flash_attn_enable False -sampling_strategy ancestral -num_epoch 2 -acc_steps 16 > ucb_lcb_${seed}.out 2>&1 &
# CUDA_VISIBLE_DEVICES=2 OPENAI_API_KEY="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" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 train/online_dpo_trainer.py -mode ucb_mean_ucb -seed "${seed}" -api_source openai -start_index 0 -end_index 2000 -num_lora 8 -num_ret 12 -gpu_takes_n 12 -prefix /data/user_data/xixu/wendaxu -per_epoch_save True -lr 5e-5 -flash_attn_enable False -sampling_strategy ancestral -num_epoch 2 -acc_steps 16 > ucb_mean_ucb_${seed}.out 2>&1 &
# CUDA_VISIBLE_DEVICES=3 OPENAI_API_KEY="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" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 train/online_dpo_trainer.py -mode ucb_rand -seed "${seed}" -api_source openai -start_index 0 -end_index 2000 -num_lora 8 -num_ret 12 -gpu_takes_n 12 -prefix /data/user_data/xixu/wendaxu -per_epoch_save True -lr 5e-5 -flash_attn_enable False -sampling_strategy ancestral -num_epoch 2 -acc_steps 16 > ucb_rand_${seed}.out 2>&1 &
# seed=44
# CUDA_VISIBLE_DEVICES=4 OPENAI_API_KEY="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" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 train/online_dpo_trainer.py -mode ucb_sec_ucb -seed "${seed}" -api_source openai -start_index 0 -end_index 2000 -num_lora 8 -num_ret 12 -gpu_takes_n 12 -prefix /data/user_data/xixu/wendaxu -per_epoch_save True -lr 5e-5 -flash_attn_enable False -sampling_strategy ancestral -num_epoch 2 -acc_steps 16 > ucb_sec_ucb_${seed}.out 2>&1 &
# CUDA_VISIBLE_DEVICES=5 OPENAI_API_KEY="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" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 train/online_dpo_trainer.py -mode ucb_lcb -seed "${seed}" -api_source openai -start_index 0 -end_index 2000 -num_lora 8 -num_ret 12 -gpu_takes_n 12 -prefix /data/user_data/xixu/wendaxu -per_epoch_save True -lr 5e-5 -flash_attn_enable False -sampling_strategy ancestral -num_epoch 2 -acc_steps 16 > ucb_lcb_${seed}.out 2>&1 &
# CUDA_VISIBLE_DEVICES=6 OPENAI_API_KEY="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" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 train/online_dpo_trainer.py -mode ucb_mean_ucb -seed "${seed}" -api_source openai -start_index 0 -end_index 2000 -num_lora 8 -num_ret 12 -gpu_takes_n 12 -prefix /data/user_data/xixu/wendaxu -per_epoch_save True -lr 5e-5 -flash_attn_enable False -sampling_strategy ancestral -num_epoch 2 -acc_steps 16 > ucb_mean_ucb_${seed}.out 2>&1 &
# CUDA_VISIBLE_DEVICES=7 OPENAI_API_KEY="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" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 train/online_dpo_trainer.py -mode ucb_rand -seed "${seed}" -api_source openai -start_index 0 -end_index 2000 -num_lora 8 -num_ret 12 -gpu_takes_n 12 -prefix /data/user_data/xixu/wendaxu -per_epoch_save True -lr 5e-5 -flash_attn_enable False -sampling_strategy ancestral -num_epoch 2 -acc_steps 16 > ucb_rand_${seed}.out 2>&1 &

# seed=42
# CUDA_VISIBLE_DEVICES=3 OPENAI_API_KEY="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" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 train/online_dpo_trainer.py -mode ucb_ref -seed "${seed}" -api_source openai -start_index 0 -end_index 2000 -num_lora 8 -num_ret 12 -gpu_takes_n 12 -prefix /data/user_data/xixu/wendaxu -per_epoch_save True -lr 5e-5 -flash_attn_enable False -sampling_strategy ancestral -num_epoch 2 -acc_steps 16 > ucb_ref_${seed}.out 2>&1 &
# seed=43
# CUDA_VISIBLE_DEVICES=4 OPENAI_API_KEY="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" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 train/online_dpo_trainer.py -mode ucb_ref -seed "${seed}" -api_source openai -start_index 0 -end_index 2000 -num_lora 8 -num_ret 12 -gpu_takes_n 12 -prefix /data/user_data/xixu/wendaxu -per_epoch_save True -lr 5e-5 -flash_attn_enable False -sampling_strategy ancestral -num_epoch 2 -acc_steps 16 > ucb_ref_${seed}.out 2>&1 &
# seed=44
# CUDA_VISIBLE_DEVICES=5 OPENAI_API_KEY="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" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 train/online_dpo_trainer.py -mode ucb_ref -seed "${seed}" -api_source openai -start_index 0 -end_index 2000 -num_lora 8 -num_ret 12 -gpu_takes_n 12 -prefix /data/user_data/xixu/wendaxu -per_epoch_save True -lr 5e-5 -flash_attn_enable False -sampling_strategy ancestral -num_epoch 2 -acc_steps 16 > ucb_ref_${seed}.out 2>&1 &

# CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 train/sft_train.py -seed 42 -peft_enable True -rank 256 -batch_size 16 -max_inp_length 64 -max_tar_length 128 -train_size 10000 -num_epoch 5 -prefix /mnt/data6/wendaxu -task_type hh > sft_rank_0.out 2>&1 &
# CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 train/sft_train.py -seed 43 -peft_enable True -rank 256 -batch_size 16 -max_inp_length 64 -max_tar_length 128 -train_size 10000 -num_epoch 5 -prefix /mnt/data6/wendaxu -task_type hh > sft_rank_1.out 2>&1 &
# CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 train/sft_train.py -seed 44 -peft_enable True -rank 256 -batch_size 16 -max_inp_length 64 -max_tar_length 128 -train_size 10000 -num_epoch 5 -prefix /mnt/data6/wendaxu -task_type hh > sft_rank_2.out 2>&1 &
# CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 train/sft_train.py -seed 46 -peft_enable True -rank 256 -batch_size 16 -max_inp_length 64 -max_tar_length 128 -train_size 10000 -num_epoch 5 -prefix /mnt/data6/wendaxu -task_type hh > sft_rank_4.out 2>&1 &
# CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 train/sft_train.py -seed 47 -peft_enable True -rank 256 -batch_size 16 -max_inp_length 64 -max_tar_length 128 -train_size 10000 -num_epoch 5 -prefix /mnt/data6/wendaxu -task_type hh > sft_rank_5.out 2>&1 &
# CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 train/sft_train.py -seed 48 -peft_enable True -rank 256 -batch_size 16 -max_inp_length 64 -max_tar_length 128 -train_size 10000 -num_epoch 5 -prefix /mnt/data6/wendaxu -task_type hh > sft_rank_6.out 2>&1 &
# CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 train/sft_train.py -seed 49 -peft_enable True -rank 256 -batch_size 16 -max_inp_length 64 -max_tar_length 128 -train_size 10000 -num_epoch 5 -prefix /mnt/data6/wendaxu -task_type hh > sft_rank_7.out 2>&1 &
# evaluation
# OPENAI_API_KEY="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" python3 eval/benchmark_llm.py -file_name test_ucb_mean_ucb_43.txt -save_folder test -data_split test -save_name test_ucb_mean_ucb_43_vs_rand_rand -api_source openai -h2h True -h2h_file test_rand_rand_43.txt


# OPENAI_API_KEY="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" nohup python3 eval/benchmark_llm.py -file_name test_sft_lora_256.txt -save_folder test -data_split test -save_name test_sft_lora_256_vs_ref -api_source openai > test_sft_lora_256_vs_ref.out 2>&1 &
# mode="ucb_sec_ucb"
# OPENAI_API_KEY="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" nohup python3 eval/benchmark_llm.py -file_name test_${mode}_42_dpo_2.txt -save_folder test -data_split test -save_name test_${mode}_42_dpo_vs_rand_2 -api_source openai -ref_addr test_rand_rand_42_dpo.txt -task_type tldr > test_${mode}_42_dpo_vs_rand_2.out 2>&1 &
# OPENAI_API_KEY="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" nohup python3 eval/benchmark_llm.py -file_name test_${mode}_43_dpo_2.txt -save_folder test -data_split test -save_name test_${mode}_43_dpo_vs_rand_2 -api_source openai -ref_addr test_rand_rand_43_dpo.txt -task_type tldr > test_${mode}_43_dpo_vs_rand_2.out 2>&1 &
# OPENAI_API_KEY="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" nohup python3 eval/benchmark_llm.py -file_name test_${mode}_44_dpo_2.txt -save_folder test -data_split test -save_name test_${mode}_44_dpo_vs_rand_2 -api_source openai -ref_addr test_rand_rand_44_dpo.txt -task_type tldr > test_${mode}_44_dpo_vs_rand_2.out 2>&1 &
# mode="ucb_mean_ucb"
# OPENAI_API_KEY="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" nohup python3 eval/benchmark_llm.py -file_name test_${mode}_42_dpo.txt -save_folder test -data_split test -save_name test_${mode}_42_dpo_vs_ref -api_source openai -ref_addr test_rand_rand_42_dpo.txt > test_${mode}_42_dpo_vs_rand.out 2>&1 &
# OPENAI_API_KEY="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" nohup python3 eval/benchmark_llm.py -file_name test_${mode}_43_dpo.txt -save_folder test -data_split test -save_name test_${mode}_43_dpo_vs_ref -api_source openai -ref_addr test_rand_rand_43_dpo.txt > test_${mode}_43_dpo_vs_rand.out 2>&1 &
# OPENAI_API_KEY="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" nohup python3 eval/benchmark_llm.py -file_name test_${mode}_44_dpo.txt -save_folder test -data_split test -save_name test_${mode}_44_dpo_vs_ref -api_source openai -ref_addr test_rand_rand_44_dpo.txt > test_${mode}_44_dpo_vs_rand.out 2>&1 &
# mode="ucb_lcb"
# OPENAI_API_KEY="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" nohup python3 eval/benchmark_llm.py -file_name test_${mode}_42_dpo.txt -save_folder test -data_split test -save_name test_${mode}_42_dpo_vs_ref -api_source openai -ref_addr test_rand_rand_42_dpo.txt > test_${mode}_42_dpo_vs_rand.out 2>&1 &
# OPENAI_API_KEY="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" nohup python3 eval/benchmark_llm.py -file_name test_${mode}_43_dpo.txt -save_folder test -data_split test -save_name test_${mode}_43_dpo_vs_ref -api_source openai -ref_addr test_rand_rand_43_dpo.txt > test_${mode}_43_dpo_vs_rand.out 2>&1 &
# OPENAI_API_KEY="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" nohup python3 eval/benchmark_llm.py -file_name test_${mode}_44_dpo.txt -save_folder test -data_split test -save_name test_${mode}_44_dpo_vs_ref -api_source openai -ref_addr test_rand_rand_44_dpo.txt > test_${mode}_44_dpo_vs_rand.out 2>&1 &
# mode="ucb_rand"
# OPENAI_API_KEY="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" nohup python3 eval/benchmark_llm.py -file_name test_${mode}_42_dpo.txt -save_folder test -data_split test -save_name test_${mode}_42_dpo_vs_rand_rev_ordered -api_source openai -task_type tldr > test_${mode}_42_dpo_vs_rand_rev_ordered.out 2>&1 &
# OPENAI_API_KEY="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" nohup python3 eval/benchmark_llm.py -file_name test_${mode}_43_dpo.txt -save_folder test -data_split test -save_name test_${mode}_43_dpo_vs_rand_rev_ordered -api_source openai -task_type tldr > test_${mode}_43_dpo_vs_rand_rev_ordered.out 2>&1 &
# python3 eval/benchmark_llm.py -file_name test_${mode}_44_dpo.txt -save_folder test -data_split test -save_name test_${mode}_44_dpo_vs_rand_rev_ordered -api_source openai -task_type tldr > test_${mode}_44_dpo_vs_rand_rev_ordered.out 2>&1 &

# OPENAI_API_KEY="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" nohup python3 eval/benchmark_llm.py -file_name test_${mode}_42_dpo_4.txt -save_folder test -data_split test -save_name test_${mode}_42_dpo_vs_ref -api_source openai -task_type tldr > test_${mode}_42_dpo_vs_ref_4.out 2>&1 &
# OPENAI_API_KEY="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" nohup python3 eval/benchmark_llm.py -file_name test_${mode}_43_dpo_4.txt -save_folder test -data_split test -save_name test_${mode}_43_dpo_vs_ref -api_source openai -task_type tldr > test_${mode}_43_dpo_vs_ref_4.out 2>&1 &
# OPENAI_API_KEY="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" nohup python3 eval/benchmark_llm.py -file_name test_${mode}_44_dpo_4.txt -save_folder test -data_split test -save_name test_${mode}_44_dpo_vs_ref -api_source openai -task_type tldr > test_${mode}_44_dpo_vs_ref_4.out 2>&1 &

# pid=2218109
# tail --pid=$pid -f /dev/null

# gamma=4
# mode="ucb_mean_ucb"
# seed=42
# CUDA_VISIBLE_DEVICES=0 OPENAI_API_KEY="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" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 train/online_dpo_trainer.py -mode "${mode}" -seed "${seed}" -api_source openai -start_index 0 -end_index 2000 -num_lora 8 -num_ret 12 -gpu_takes_n 12 -prefix /data/user_data/xixu/wendaxu -per_epoch_save True -lr 5e-5 -flash_attn_enable False -sampling_strategy ancestral -num_epoch 2 -acc_steps 16 -gamma ${gamma} > ${mode}_${seed}_${gamma}.out 2>&1 &
# seed=43
# CUDA_VISIBLE_DEVICES=0 OPENAI_API_KEY="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" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 train/online_dpo_trainer.py -mode "${mode}" -seed "${seed}" -api_source openai -start_index 0 -end_index 2000 -num_lora 8 -num_ret 12 -gpu_takes_n 12 -prefix /data/user_data/xixu/wendaxu -per_epoch_save True -lr 5e-5 -flash_attn_enable False -sampling_strategy ancestral -num_epoch 2 -acc_steps 16 -gamma ${gamma} > ${mode}_${seed}_${gamma}.out 2>&1 &
# seed=44
# CUDA_VISIBLE_DEVICES=1 OPENAI_API_KEY="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" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 train/online_dpo_trainer.py -mode "${mode}" -seed "${seed}" -api_source openai -start_index 0 -end_index 2000 -num_lora 8 -num_ret 12 -gpu_takes_n 12 -prefix /data/user_data/xixu/wendaxu -per_epoch_save True -lr 5e-5 -flash_attn_enable False -sampling_strategy ancestral -num_epoch 2 -acc_steps 16 -gamma ${gamma} > ${mode}_${seed}_${gamma}.out 2>&1 &


# seed=43
# gamma=2
# mode="ucb_lcb"
# CUDA_VISIBLE_DEVICES=3 OPENAI_API_KEY="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" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 train/online_dpo_trainer.py -mode "${mode}" -seed "${seed}" -api_source openai -start_index 0 -end_index 2000 -num_lora 8 -num_ret 12 -gpu_takes_n 12 -prefix /data/user_data/xixu/wendaxu -per_epoch_save True -lr 5e-5 -flash_attn_enable False -sampling_strategy ancestral -num_epoch 2 -acc_steps 16 -gamma ${gamma} > ${mode}_${seed}_${gamma}.out 2>&1 &
# seed=44
# gamma=2
# mode="ucb_lcb"
# CUDA_VISIBLE_DEVICES=4 OPENAI_API_KEY="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" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 train/online_dpo_trainer.py -mode "${mode}" -seed "${seed}" -api_source openai -start_index 0 -end_index 2000 -num_lora 8 -num_ret 12 -gpu_takes_n 12 -prefix /data/user_data/xixu/wendaxu -per_epoch_save True -lr 5e-5 -flash_attn_enable False -sampling_strategy ancestral -num_epoch 2 -acc_steps 16 -gamma ${gamma} > ${mode}_${seed}_${gamma}.out 2>&1 &

# seed=42
# gamma=4
# mode="ucb_lcb"
# CUDA_VISIBLE_DEVICES=5 OPENAI_API_KEY="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" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 train/online_dpo_trainer.py -mode "${mode}" -seed "${seed}" -api_source openai -start_index 0 -end_index 2000 -num_lora 8 -num_ret 12 -gpu_takes_n 12 -prefix /data/user_data/xixu/wendaxu -per_epoch_save True -lr 5e-5 -flash_attn_enable False -sampling_strategy ancestral -num_epoch 2 -acc_steps 16 -gamma ${gamma} > ${mode}_${seed}_${gamma}.out 2>&1 &
# seed=43
# gamma=4
# mode="ucb_lcb"
# CUDA_VISIBLE_DEVICES=6 OPENAI_API_KEY="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" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 train/online_dpo_trainer.py -mode "${mode}" -seed "${seed}" -api_source openai -start_index 0 -end_index 2000 -num_lora 8 -num_ret 12 -gpu_takes_n 12 -prefix /data/user_data/xixu/wendaxu -per_epoch_save True -lr 5e-5 -flash_attn_enable False -sampling_strategy ancestral -num_epoch 2 -acc_steps 16 -gamma ${gamma} > ${mode}_${seed}_${gamma}.out 2>&1 &
# seed=44
# gamma=4
# mode="ucb_lcb"
# CUDA_VISIBLE_DEVICES=7 OPENAI_API_KEY="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" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 train/online_dpo_trainer.py -mode "${mode}" -seed "${seed}" -api_source openai -start_index 0 -end_index 2000 -num_lora 8 -num_ret 12 -gpu_takes_n 12 -prefix /data/user_data/xixu/wendaxu -per_epoch_save True -lr 5e-5 -flash_attn_enable False -sampling_strategy ancestral -num_epoch 2 -acc_steps 16 -gamma ${gamma} > ${mode}_${seed}_${gamma}.out 2>&1 &

# pid=2248901
# tail --pid=$pid -f /dev/null
# gamma=4
# seed=43
# CUDA_VISIBLE_DEVICES=4 OPENAI_API_KEY="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" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 train/online_dpo_trainer.py -mode ucb_rand -seed "${seed}" -api_source openai -start_index 0 -end_index 2000 -num_lora 8 -num_ret 12 -gpu_takes_n 12 -prefix /data/user_data/xixu/wendaxu -per_epoch_save True -lr 5e-5 -flash_attn_enable False -sampling_strategy ancestral -num_epoch 2 -acc_steps 16 -gamma ${gamma} > ucb_rand_${seed}_${gamma}.out 2>&1 &
# seed=44
# CUDA_VISIBLE_DEVICES=5 OPENAI_API_KEY="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" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 train/online_dpo_trainer.py -mode ucb_rand -seed "${seed}" -api_source openai -start_index 0 -end_index 2000 -num_lora 8 -num_ret 12 -gpu_takes_n 12 -prefix /data/user_data/xixu/wendaxu -per_epoch_save True -lr 5e-5 -flash_attn_enable False -sampling_strategy ancestral -num_epoch 2 -acc_steps 16 -gamma ${gamma} > ucb_rand_${seed}_${gamma}.out 2>&1 &


# inference
# mode="rand_rand"
# CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 inference/inference_adapter.py -ckpt /data/user_data/xixu/wendaxu/epoch_1_${mode}_42_may_8_5e-05 -prefix ${mode}_42_dpo -data_split test > inf_${mode}_42_dpo.out 2>&1 &
# CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 inference/inference_adapter.py -ckpt /data/user_data/xixu/wendaxu/epoch_1_${mode}_43_may_8_5e-05 -prefix ${mode}_43_dpo -data_split test > inf_${mode}_43_dpo.out 2>&1 &
# CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 inference/inference_adapter.py -ckpt /data/user_data/xixu/wendaxu/epoch_1_${mode}_44_may_8_5e-05 -prefix ${mode}_44_dpo -data_split test > inf_${mode}_44_dpo.out 2>&1 &
# CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 inference/inference_adapter.py -ckpt sft_lora_256 -prefix sft_lora_256 -data_split test > inf_sft_lora_256.out 2>&1 &

# mode="ucb_mean_ucb"
# CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 inference/inference_adapter.py -ckpt /data/user_data/xixu/wendaxu/epoch_1_${mode}_42_may_8_5e-05 -prefix ${mode}_42_dpo -data_split test > inf_${mode}_42_dpo.out 2>&1 &
# CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 inference/inference_adapter.py -ckpt /data/user_data/xixu/wendaxu/epoch_1_${mode}_43_may_8_5e-05 -prefix ${mode}_43_dpo -data_split test > inf_${mode}_43_dpo.out 2>&1 &
# CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 inference/inference_adapter.py -ckpt /data/user_data/xixu/wendaxu/epoch_1_${mode}_44_may_8_5e-05 -prefix ${mode}_44_dpo -data_split test > inf_${mode}_44_dpo.out 2>&1 &
# mode="ucb_rand"
# CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 inference/inference_adapter.py -ckpt /data/user_data/xixu/wendaxu/epoch_1_${mode}_42_may_8_5e-05_4 -prefix ${mode}_42_dpo_4 -data_split test -ensemble False > inf_${mode}_42_dpo_4.out 2>&1 &
# CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 inference/inference_adapter.py -ckpt /data/user_data/xixu/wendaxu/epoch_1_${mode}_43_may_8_5e-05_4 -prefix ${mode}_43_dpo_4 -data_split test -ensemble False > inf_${mode}_43_dpo_4.out 2>&1 &
# CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 inference/inference_adapter.py -ckpt /data/user_data/xixu/wendaxu/epoch_1_${mode}_44_may_8_5e-05_4 -prefix ${mode}_44_dpo_4 -data_split test -ensemble False > inf_${mode}_44_dpo_4.out 2>&1 &
# mode="ucb_sec_ucb"
# CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 inference/inference_adapter.py -ckpt /data/user_data/xixu/wendaxu/epoch_1_${mode}_42_may_8_5e-05_2 -prefix ${mode}_42_dpo_2 -data_split test -ensemble False > inf_${mode}_42_dpo_2.out 2>&1 &
# CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 inference/inference_adapter.py -ckpt /data/user_data/xixu/wendaxu/epoch_1_${mode}_43_may_8_5e-05_2 -prefix ${mode}_43_dpo_2 -data_split test -ensemble False > inf_${mode}_43_dpo_2.out 2>&1 &
# CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 inference/inference_adapter.py -ckpt /data/user_data/xixu/wendaxu/epoch_1_${mode}_44_may_8_5e-05_2 -prefix ${mode}_44_dpo_2 -data_split test -ensemble False > inf_${mode}_44_dpo_2.out 2>&1 &

# mode="ucb_rand"
# seed=42
# CUDA_VISIBLE_DEVICES=0 nohup python3 inference/inference_adapter.py -ckpt /data/user_data/xixu/wendaxu/epoch_1_${mode}_${seed}_may_10_5e-05_2 -prefix ${mode}_${seed}_reward_2 -data_split test -ensemble False > inf_${mode}_${seed}_2.out 2>&1 &
# seed=43
# CUDA_VISIBLE_DEVICES=1 nohup python3 inference/inference_adapter.py -ckpt /data/user_data/xixu/wendaxu/epoch_1_${mode}_${seed}_may_10_5e-05_2 -prefix ${mode}_${seed}_reward_2 -data_split test -ensemble False > inf_${mode}_${seed}_2.out 2>&1 &
# seed=44
# CUDA_VISIBLE_DEVICES=2 nohup python3 inference/inference_adapter.py -ckpt /data/user_data/xixu/wendaxu/epoch_1_${mode}_${seed}_may_10_5e-05_2 -prefix ${mode}_${seed}_reward_2 -data_split test -ensemble False > inf_${mode}_${seed}_2.out 2>&1 &

# mode="ucb_rand"
# for seed in 42 43 44
# do 
#     CUDA_VISIBLE_DEVICES=0 nohup python3 eval/benchmark_llm_reward.py -file_name test_${mode}_${seed}_reward.txt -data_split test  > inf_${mode}_${seed}.out 2>&1 &
# done

# CUDA_VISIBLE_DEVICES=0 nohup python3 eval/benchmark_llm_reward.py -file_name test_sft_lora_256.txt -data_split test  > eval_sft.out 2>&1 &

# gamma=0
# seed=42
# num_lora=1
# mode="rand_rand"
# CUDA_VISIBLE_DEVICES=3 nohup python3 inference/inference_adapter.py -ckpt /data/user_data/xixu/wendaxu/epoch_1_${mode}_${seed}_may_10_5e-05_0.0 -prefix ${mode}_${seed} -data_split test -ensemble False -start_index 0 -end_index 2000 -decode_strategy greedy -seed "${seed}" > ${mode}_${seed}.out 2>&1 &
# seed=43
# CUDA_VISIBLE_DEVICES=4 nohup python3 inference/inference_adapter.py -ckpt /data/user_data/xixu/wendaxu/epoch_1_${mode}_${seed}_may_10_5e-05_0.0 -prefix ${mode}_${seed} -data_split test -ensemble False -start_index 0 -end_index 2000 -decode_strategy greedy -seed "${seed}" > ${mode}_${seed}.out 2>&1 &
# seed=44
# CUDA_VISIBLE_DEVICES=5 nohup python3 inference/inference_adapter.py -ckpt /data/user_data/xixu/wendaxu/epoch_1_${mode}_${seed}_may_10_5e-05_0.0 -prefix ${mode}_${seed} -data_split test -ensemble False -start_index 0 -end_index 2000 -decode_strategy greedy -seed "${seed}" > ${mode}_${seed}.out 2>&1 &

# gamma=2
# mode="upper_bound"
# seed=42
# CUDA_VISIBLE_DEVICES=0 nohup python3 eval/benchmark_llm_reward.py -file_name test_${mode}_${seed}.txt -ref_addr test_rand_rand_${seed}.txt -data_split test > eval_${mode}_${seed}_rand.out 2>&1 & # -ref_addr test_rand_rand_${seed}_reward.txt 
# seed=43
# CUDA_VISIBLE_DEVICES=1 nohup python3 eval/benchmark_llm_reward.py -file_name test_${mode}_${seed}.txt -ref_addr test_rand_rand_${seed}.txt -data_split test > eval_${mode}_${seed}_rand.out 2>&1 & # -ref_addr test_rand_rand_${seed}_reward.txt
# seed=44
# CUDA_VISIBLE_DEVICES=2 nohup python3 eval/benchmark_llm_reward.py -file_name test_${mode}_${seed}.txt -ref_addr test_rand_rand_${seed}.txt -data_split test > eval_${mode}_${seed}_rand.out 2>&1 & # -ref_addr test_rand_rand_${seed}_reward.txt  

# CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 inference/inference_adapter.py -ckpt /data/user_data/xixu/wendaxu/may_26_unmerged/epoch_0_ucb_rand_42_may_10_5e-05_3.5_resume_False_500_replay_0_offline_False_shuffle_True_reward_on_policy_False_ret_60 -prefix ucb_rand_500_unmerged -data_split test -ensemble True -decode_strategy greedy -seed 42 -num_lora 5 -start_index 0 -end_index 1000 > 42.out 2>&1 &
# CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 inference/inference_adapter.py -ckpt /data/user_data/xixu/wendaxu/may_26_unmerged/epoch_0_ucb_rand_43_may_10_5e-05_3.5_resume_False_500_replay_0_offline_False_shuffle_True_reward_on_policy_False_ret_60 -prefix ucb_rand_500_unmerged -data_split test -ensemble True -decode_strategy greedy -seed 43 -num_lora 5 -start_index 0 -end_index 1000 > 43.out 2>&1 &
# CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 inference/inference_adapter.py -ckpt /data/user_data/xixu/wendaxu/may_26_unmerged/epoch_0_ucb_rand_44_may_10_5e-05_3.5_resume_False_500_replay_0_offline_False_shuffle_True_reward_on_policy_False_ret_60 -prefix ucb_rand_500_unmerged -data_split test -ensemble True -decode_strategy greedy -seed 44 -num_lora 5 -start_index 0 -end_index 1000 > 44.out 2>&1 &

# mode="ucb_rand"
# seed=42
# CUDA_VISIBLE_DEVICES=0 nohup python3 eval/benchmark_llm_reward.py -file_name test_${mode}_500_unmerged_${seed}.txt -data_split test > eval_${mode}_500_unmerged_${seed}.out 2>&1 & # -ref_addr test_rand_rand_${seed}_reward.txt 
# seed=43
# CUDA_VISIBLE_DEVICES=1 nohup python3 eval/benchmark_llm_reward.py -file_name test_${mode}_500_unmerged_${seed}.txt -data_split test > eval_${mode}_500_unmerged_${seed}.out 2>&1 & # -ref_addr test_rand_rand_${seed}_reward.txt
# seed=44
# CUDA_VISIBLE_DEVICES=2 nohup python3 eval/benchmark_llm_reward.py -file_name test_${mode}_500_unmerged_${seed}.txt -data_split test > eval_${mode}_500_unmerged_${seed}.out 2>&1 & # -ref_addr test_rand_rand_${seed}_reward.txt  

# unused variables
gamma=0
replay_step=0
rank_strategy="reward"
on_policy_rand=False

# basic setup
mode="rand_rand"
num_lora=5
num_epoch=1
shuffle_data=True
num_ret=2
gpu_takes_n=2
flash_attn_enable=True
end_index=10000
alpha=5e-3

# task specific variable
loss_type="hinge" # "ipo" "sigmoid" "hinge"
task_type="tldr" # "tldr", "hh", "harm"
prefix="/data/user_data/xixu/wendaxu/${task_type}_${loss_type}_more_june_13"
max_prompt_length=512 # 512, 64, 256
max_new_token=128

# try different baselins
sampling_update_step=1
run_sample_strategy="per_interval_merged" # "per_interval_merged", "sft"
train_baseline_model_mode="merge_on_interval"  # "merge_on_interval", "sft" 
alg_type="opo"
run_type="online"
method_name="${run_type}_${loss_type}_${alg_type}"
baseline_name="${method_name}_${loss_type}" # "offline_dpo" "online_dpo" "online_opo"

# dynamically ajust for different GPUs
save_step=625 # 125, 30, 125 
eval_step=40 # 40, 15, 40
acc_steps=16 # 16, 64, 16 for hh
sampling_batch_size=8 # 8, 64, 16
batch_size=1
eval_batch_size=32 # 32, 256, 64
# loaded_ref_addr="/data/user_data/xixu/wendaxu/all_hh_results_june_10_dpo/epoch_0_rand_rand_42_may_10_5e-05_0.0_resume_False_157_replay_0_offline_False_shuffle_True_reward_on_policy_False_ret_2_sample_per_interval_merged_1_merge_on_interval"

seed=42
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
TOKENIZERS_PARALLELISM=true nohup python train/online_dpo_rand_trainer.py -mode "${mode}" -seed "${seed}" \
-start_index 0 -end_index "${end_index}" -num_lora "${num_lora}" -num_ret ${num_ret} -gpu_takes_n ${gpu_takes_n} \
-prefix ${prefix} -per_epoch_save True -lr 5e-5 -flash_attn_enable ${flash_attn_enable} -sampling_strategy ancestral \
-num_epoch ${num_epoch} -acc_steps ${acc_steps} -gamma ${gamma} -logging_selection False -save_step ${save_step} \
-eval_step ${eval_step} -data_capacity 100 -replay_step ${replay_step} -baseline_name ${baseline_name} \
-shuffle_data ${shuffle_data} -rank_strategy ${rank_strategy} -on_policy_rand ${on_policy_rand} -loss_type ${loss_type} \
-run_sample_strategy ${run_sample_strategy} -sampling_update_step ${sampling_update_step} -eval_batch_size ${eval_batch_size} \
-train_baseline_model_mode ${train_baseline_model_mode} -alpha ${alpha} -task_type "${task_type}" -batch_size ${batch_size} \
-max_prompt_length "${max_prompt_length}" -max_new_token "${max_new_token}" -sampling_batch_size "${sampling_batch_size}" -load_ref_addr "${loaded_ref_addr}" \
> ${mode}_${seed}_${gamma}_reward_replay_${replay_step}_${offline_enable}_${shuffle_data}_${rank_strategy}_${num_ret}_on_policy_${on_policy_rand}_${run_sample_strategy}_${sampling_update_step}_${train_baseline_model_mode}_${num_lora}_${alpha}_${task_type}_${loss_type}.out 2>&1 &

seed=45
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
TOKENIZERS_PARALLELISM=true nohup python train/online_dpo_rand_trainer.py -mode "${mode}" -seed "${seed}" \
-start_index 0 -end_index "${end_index}" -num_lora "${num_lora}" -num_ret ${num_ret} -gpu_takes_n ${gpu_takes_n} \
-prefix ${prefix} -per_epoch_save True -lr 5e-5 -flash_attn_enable ${flash_attn_enable} -sampling_strategy ancestral \
-num_epoch ${num_epoch} -acc_steps ${acc_steps} -gamma ${gamma} -logging_selection False -save_step ${save_step} \
-eval_step ${eval_step} -data_capacity 100 -replay_step ${replay_step} -baseline_name ${baseline_name} \
-shuffle_data ${shuffle_data} -rank_strategy ${rank_strategy} -on_policy_rand ${on_policy_rand} -loss_type ${loss_type} \
-run_sample_strategy ${run_sample_strategy} -sampling_update_step ${sampling_update_step} -eval_batch_size ${eval_batch_size} \
-train_baseline_model_mode ${train_baseline_model_mode} -alpha ${alpha} -task_type "${task_type}" -batch_size ${batch_size} \
-max_prompt_length "${max_prompt_length}" -max_new_token "${max_new_token}" -sampling_batch_size "${sampling_batch_size}" -load_ref_addr "${loaded_ref_addr}" \
> ${mode}_${seed}_${gamma}_reward_replay_${replay_step}_${offline_enable}_${shuffle_data}_${rank_strategy}_${num_ret}_on_policy_${on_policy_rand}_${run_sample_strategy}_${sampling_update_step}_${train_baseline_model_mode}_${num_lora}_${alpha}_${task_type}_${loss_type}.out 2>&1 &

# seed=46
# CUDA_VISIBLE_DEVICES=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
# TOKENIZERS_PARALLELISM=true nohup python3 train/online_dpo_rand_trainer.py -mode "${mode}" -seed "${seed}" \
# -start_index 0 -end_index "${end_index}" -num_lora "${num_lora}" -num_ret ${num_ret} -gpu_takes_n ${gpu_takes_n} \
# -prefix ${prefix} -per_epoch_save True -lr 5e-5 -flash_attn_enable ${flash_attn_enable} -sampling_strategy ancestral \
# -num_epoch ${num_epoch} -acc_steps ${acc_steps} -gamma ${gamma} -logging_selection False -save_step ${save_step} \
# -eval_step ${eval_step} -data_capacity 100 -replay_step ${replay_step} -baseline_name ${baseline_name} \
# -shuffle_data ${shuffle_data} -rank_strategy ${rank_strategy} -on_policy_rand ${on_policy_rand} -loss_type ${loss_type} \
# -run_sample_strategy ${run_sample_strategy} -sampling_update_step ${sampling_update_step} -eval_batch_size ${eval_batch_size} \
# -train_baseline_model_mode ${train_baseline_model_mode} -alpha ${alpha} -task_type "${task_type}" -batch_size ${batch_size} \
# -max_prompt_length "${max_prompt_length}" -max_new_token "${max_new_token}" -sampling_batch_size "${sampling_batch_size}" \
# -load_ref_addr "${loaded_ref_addr}" \
# > ${mode}_${seed}_${gamma}_reward_replay_${replay_step}_${offline_enable}_${shuffle_data}_${rank_strategy}_${num_ret}_on_policy_${on_policy_rand}_${run_sample_strategy}_${sampling_update_step}_${train_baseline_model_mode}_${num_lora}_${alpha}_${task_type}_${loss_type}.out 2>&1 &

# task_type="hh"
# max_inp_length=64
# max_tar_length=128
# batch_size=24
# max_inp_length=256
# max_tar_length=128
# seed=42
# CUDA_VISIBLE_DEVICES=0 nohup python3 train/sft_train.py -seed "${seed}" -peft_enable True -rank 256 -batch_size ${batch_size} -max_inp_length ${max_inp_length} -max_tar_length ${max_tar_length} -train_size 10000 -num_epoch 10 -prefix /data/user_data/xixu/wendaxu/sft_${task_type} -flash_attn_enable False -task_type ${task_type} > ${task_type}_${seed}.out 2>&1 &
# seed=43
# CUDA_VISIBLE_DEVICES=1 nohup python3 train/sft_train.py -seed "${seed}" -peft_enable True -rank 256 -batch_size ${batch_size} -max_inp_length ${max_inp_length} -max_tar_length ${max_tar_length} -train_size 10000 -num_epoch 10 -prefix /data/user_data/xixu/wendaxu/sft_${task_type} -flash_attn_enable False -task_type ${task_type} > ${task_type}_${seed}.out 2>&1 &
# seed=44
# CUDA_VISIBLE_DEVICES=2 nohup python3 train/sft_train.py -seed "${seed}" -peft_enable True -rank 256 -batch_size ${batch_size} -max_inp_length ${max_inp_length} -max_tar_length ${max_tar_length} -train_size 10000 -num_epoch 10 -prefix /data/user_data/xixu/wendaxu/sft_${task_type} -flash_attn_enable False -task_type ${task_type} > ${task_type}_${seed}.out 2>&1 &
# seed=45
# CUDA_VISIBLE_DEVICES=3 nohup python3 train/sft_train.py -seed "${seed}" -peft_enable True -rank 256 -batch_size ${batch_size} -max_inp_length ${max_inp_length} -max_tar_length ${max_tar_length} -train_size 10000 -num_epoch 10 -prefix /data/user_data/xixu/wendaxu/sft_${task_type} -flash_attn_enable False -task_type ${task_type} > ${task_type}_${seed}.out 2>&1 &
# seed=46
# CUDA_VISIBLE_DEVICES=0 nohup python3 train/sft_train.py -seed "${seed}" -peft_enable True -rank 256 -batch_size ${batch_size} -max_inp_length ${max_inp_length} -max_tar_length ${max_tar_length} -train_size 10000 -num_epoch 10 -prefix /data/user_data/xixu/wendaxu/sft_${task_type} -flash_attn_enable False -task_type ${task_type} > ${task_type}_${seed}.out 2>&1 &

# CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 inference/inference_adapter.py -ckpt harm52 -prefix rand_rand -data_split test -ensemble True -decode_strategy greedy -num_lora 5 -task_type harm > harm_test_52.out 2>&1 &
# CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 inference/inference_adapter.py -ckpt harm104 -prefix rand_rand -data_split dev -ensemble True -decode_strategy greedy -num_lora 5 -task_type harm > harm_test_104.out 2>&1 &
# CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 inference/inference_adapter.py -ckpt harm156 -prefix rand_rand -data_split dev -ensemble True -decode_strategy greedy -num_lora 5 -task_type harm > harm_test_156.out 2>&1 &

# CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 inference/inference_adapter.py -ckpt hh26 -prefix rand_rand -data_split dev -ensemble True -decode_strategy greedy -num_lora 5 -task_type hh > hh_test.out 2>&1 &
# CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 inference/inference_adapter.py -ckpt hh52 -prefix rand_rand -data_split test -ensemble True -decode_strategy greedy -num_lora 5 -task_type hh > hh_test_52.out 2>&1 &
# CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python3 inference/inference_adapter.py -ckpt hh78 -prefix rand_rand -data_split dev -ensemble True -decode_strategy greedy -num_lora 5 -task_type hh > hh_test_78.out 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python3 eval/benchmark_llm_reward.py -file_name test_rand_rand_harm_52.txt -data_split test -task_type harm > harm_52_sft.out 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python3 eval/benchmark_llm_reward.py -file_name dev_rand_rand_harm_104.txt -data_split dev -task_type harm > harm_104.out 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python3 eval/benchmark_llm_reward.py -file_name dev_rand_rand_harm_156.txt -data_split dev -task_type harm > harm_156.out 2>&1 &

# for seed in 44 42 43 
# do  
#     echo "Under ${seed} run"
#     CUDA_VISIBLE_DEVICES=3 python3 eval/benchmark_llm_reward.py -file_name test_online_opo_${seed}_hh.txt -data_split test -task_type hh -ref_addr test_offline_dpo_${seed}_hh.txt -max_length 192 -batch_size 256
# done

# for seed in 42 44
# do  
#     echo "Under ${seed} run"
#     CUDA_VISIBLE_DEVICES=0 python3 eval/benchmark_llm_reward.py -file_name test_online_opo_${seed}_harm.txt -data_split test -task_type harm -ref_addr test_online_dpo_${seed}_harm.txt -max_length 384 -batch_size 64
# done
# seed=43 
# echo "Under ${seed} run"
# CUDA_VISIBLE_DEVICES=0 python3 eval/benchmark_llm_reward.py -file_name test_online_opo_${seed}_harm.txt -data_split test -task_type harm -ref_addr test_online_dpo_${seed}_harm.txt -max_length 384 -batch_size 64 -special_tok True

# CUDA_VISIBLE_DEVICES=1 nohup python3 eval/benchmark_llm_reward.py -file_name test_rand_rand_hh_52.txt -data_split test -task_type hh > hh_52_sft.out 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python3 eval/benchmark_llm_reward.py -file_name dev_rand_rand_hh_78.txt -data_split dev -task_type hh > hh_78.out 2>&1 &

# inference time
# for seed in 42 43
# do  
#     ckpt="/data/user_data/xixu/wendaxu/all_hh_results_june_9/epoch_0_rand_rand_${seed}_may_10_5e-05_0.0_resume_False_150_replay_0_offline_False_shuffle_True_reward_on_policy_False_ret_2_sample_per_interval_merged_1_merge_on_interval"
#     CUDA_VISIBLE_DEVICES=3 python3 inference/inference_adapter.py -ckpt ${ckpt} -prefix online_opo_${seed} -data_split test -ensemble True -decode_strategy greedy -num_lora 5 -task_type hh -batch_size 32
# done

# seed=44
# ckpt="/data/user_data/xixu/wendaxu/all_hh_results_june_10/epoch_0_rand_rand_${seed}_may_10_5e-05_0.0_resume_False_150_replay_0_offline_False_shuffle_True_reward_on_policy_False_ret_2_sample_per_interval_merged_1_merge_on_interval"
# CUDA_VISIBLE_DEVICES=3 python3 inference/inference_adapter.py -ckpt ${ckpt} -prefix online_opo_${seed} -data_split test -ensemble True -decode_strategy greedy -num_lora 5 -task_type hh -batch_size 32

# for seed in 42 43 44
# do  
# seed=44
# ckpt="/data/user_data/xixu/wendaxu/all_hh_results_june_9/epoch_0_rand_rand_${seed}_may_10_5e-05_0.0_resume_False_157_replay_0_offline_False_shuffle_True_reward_on_policy_False_ret_2_sample_per_interval_merged_1_sft"
# CUDA_VISIBLE_DEVICES=3 python3 inference/inference_adapter.py -ckpt ${ckpt} -prefix online_dpo_${seed} -data_split test -ensemble True -decode_strategy greedy -num_lora 5 -task_type hh -batch_size 32
# done

# for seed in 42 43 44
# do  
#     ckpt="/data/user_data/xixu/wendaxu/all_hh_results_june_9/epoch_0_rand_rand_${seed}_may_10_5e-05_0.0_resume_False_150_replay_0_offline_False_shuffle_True_reward_on_policy_False_ret_2_sample_sft_1_sft"
#     CUDA_VISIBLE_DEVICES=3 python3 inference/inference_adapter.py -ckpt ${ckpt} -prefix offline_dpo_${seed} -data_split test -ensemble True -decode_strategy greedy -num_lora 5 -task_type hh -batch_size 32
# done

# for seed in 43 # 42 
# do  
#     ckpt="/data/user_data/xixu/wendaxu/all_hh_results_june_10/epoch_0_rand_rand_${seed}_may_10_5e-05_0.0_resume_False_625_replay_0_offline_False_shuffle_True_reward_on_policy_False_ret_2_sample_per_interval_merged_1_merge_on_interval"
#     CUDA_VISIBLE_DEVICES=0 python3 inference/inference_adapter.py -ckpt ${ckpt} -prefix online_opo_${seed} -data_split test -ensemble True -decode_strategy greedy -num_lora 5 -task_type harm -batch_size 32
# done

# ckpt="/data/user_data/xixu/wendaxu/all_harm_results_june_10_rerun/epoch_0_rand_rand_45_may_10_5e-05_0.0_resume_False_625_replay_0_offline_False_shuffle_True_reward_on_policy_False_ret_2_sample_per_interval_merged_1_merge_on_interval"
# CUDA_VISIBLE_DEVICES=0 python3 inference/inference_adapter.py -ckpt ${ckpt} -prefix online_opo_44 -data_split test -ensemble True -decode_strategy greedy -num_lora 5 -task_type harm -batch_size 32

# for seed in 42 43 44
# do  
#     ckpt="/data/user_data/xixu/wendaxu/all_hh_results_june_10/epoch_0_rand_rand_${seed}_may_10_5e-05_0.0_resume_False_625_replay_0_offline_False_shuffle_True_reward_on_policy_False_ret_2_sample_per_interval_merged_1_sft"
#     CUDA_VISIBLE_DEVICES=0 python3 inference/inference_adapter.py -ckpt ${ckpt} -prefix online_dpo_${seed} -data_split test -ensemble True -decode_strategy greedy -num_lora 5 -task_type harm -batch_size 32
# done

# for seed in 42 43 44
# do  
#     ckpt="/data/user_data/xixu/wendaxu/all_hh_results_june_10/epoch_0_rand_rand_${seed}_may_10_5e-05_0.0_resume_False_625_replay_0_offline_False_shuffle_True_reward_on_policy_False_ret_2_sample_sft_1_sft"
#     CUDA_VISIBLE_DEVICES=3 python3 inference/inference_adapter.py -ckpt ${ckpt} -prefix offline_dpo_${seed} -data_split test -ensemble True -decode_strategy greedy -num_lora 5 -task_type harm -batch_size 32
# done