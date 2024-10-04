 
# for rand rand
for weight_prefix in "rand_None_142" "rand_None_242" "rand_None_342" "rand_None_442"
do
    # python3 util/model_averaging.py -weight_prefix ${weight_prefix} -ckpt checkpoint-218
    cp /share/edc/home/wendaxu/dpo/0_dpo_${weight_prefix}/checkpoint-218/*.json /share/edc/home/wendaxu/dpo/all_dpo_${weight_prefix}/
done

# for ucb lcb
for weight_prefix in "uncertainty_lcb_142" "uncertainty_lcb_242" "uncertainty_lcb_342" "uncertainty_lcb_442"
do
    # python3 util/model_averaging.py -weight_prefix ${weight_prefix} -ckpt checkpoint-219
    cp /share/edc/home/wendaxu/dpo/0_dpo_${weight_prefix}/checkpoint-219/*.json /share/edc/home/wendaxu/dpo/all_dpo_${weight_prefix}/
done

# for ucb mean
for weight_prefix in "uncertainty_mean_ucb_142" "uncertainty_mean_ucb_242" "uncertainty_mean_ucb_342" "uncertainty_mean_ucb_442"
do
    # python3 util/model_averaging.py -weight_prefix ${weight_prefix} -ckpt checkpoint-218
    cp /share/edc/home/wendaxu/dpo/0_dpo_${weight_prefix}/checkpoint-218/*.json /share/edc/home/wendaxu/dpo/all_dpo_${weight_prefix}/
done

# for ucb rand
for weight_prefix in "uncertainty_rand_142" "uncertainty_rand_242" "uncertainty_rand_342" "uncertainty_rand_442"
do
    # python3 util/model_averaging.py -weight_prefix ${weight_prefix} -ckpt checkpoint-218
    cp /share/edc/home/wendaxu/dpo/0_dpo_${weight_prefix}/checkpoint-218/*.json /share/edc/home/wendaxu/dpo/all_dpo_${weight_prefix}/
done

# for ucb sec_ucb
for weight_prefix in "uncertainty_sec_ucb_142" "uncertainty_sec_ucb_242" "uncertainty_sec_ucb_342" "uncertainty_sec_ucb_442"
do
    # python3 util/model_averaging.py -weight_prefix ${weight_prefix} -ckpt checkpoint-219
    cp /share/edc/home/wendaxu/dpo/0_dpo_${weight_prefix}/checkpoint-219/*.json /share/edc/home/wendaxu/dpo/all_dpo_${weight_prefix}/
done