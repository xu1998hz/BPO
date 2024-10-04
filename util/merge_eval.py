import json

# for i in range(1, 5):
file_ls = [f"rand_None_greedy_rank_0.json", f"uncertainty_lcb_greedy_rank_0.json", f"uncertainty_mean_ucb_greedy_rank_0.json", f"uncertainty_rand_greedy_rank_0.json", f"uncertainty_sec_ucb_greedy_rank_0.json"]

cur_dict = set()
for f_name in file_ls:
    data = json.load(open(f_name))
    if len(cur_dict) == 0:
        cur_dict=set(data['prompt'])
    else:
        cur_dict=cur_dict&set(data['prompt'])

# only 466 prompts are overlapping

for f_name in file_ls:
    data = json.load(open(f_name))
    print(f_name)
    cor=0
    total=0
    for ele, score in zip(data['prompt'], data['scores']):
        if ele in cur_dict:
            cor+=score
            total+=1
    print(cor/total)
    print(total)
    print()
print('-'*20)