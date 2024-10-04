import json
from itertools import combinations
import math
import random

def ret_selected_ls(num_response, num_sys, mode):
    final_dict={}

    for i in range(num_sys):
        data = json.load(open(f'data/{i}_sec_0_2000.json'))
        for ele, scores_dict in data.items():
            rewards_ls = [pi_score-ref_score for pi_score, ref_score in zip(scores_dict['pi'], scores_dict['ref'])]
            if ele not in final_dict:
                final_dict[ele]={}
            final_dict[ele][i]=rewards_ls
    
    selected_ls = []
    for ele, ind_pair_dict in final_dict.items():
        inds_ucb_ls=[]
        for i in range(num_response):
            response_ucb_ls = []
            for j in range(num_sys):
                response_ucb_ls+=[ind_pair_dict[j][i]]
            
            temp_ls=[(ele-sum(response_ucb_ls)/len(response_ucb_ls))**2 for ele in response_ucb_ls]
            inds_ucb_ls+=[(sum(response_ucb_ls)/len(response_ucb_ls) + math.sqrt(sum(temp_ls)/len(temp_ls)), i)]
        # first one will select ucb, second depends on mode
        if mode == "sec_ucb":
            selected_ls+=[[sorted(inds_ucb_ls)[-1][1], sorted(inds_ucb_ls)[-2][1]]]
        elif mode == "lcb":
            selected_ls+=[[sorted(inds_ucb_ls)[-1][1], sorted(inds_ucb_ls)[0][1]]]
        elif mode == "mean_ucb":
            selected_ls+=[[sorted(inds_ucb_ls)[-1][1], sorted(inds_ucb_ls)[int(len(inds_ucb_ls)/2)][1]]]
        elif mode == "rand":
            selected_ls+=[[sorted(inds_ucb_ls)[-1][1], random.sample(sorted(inds_ucb_ls)[:-1], 1)[0][1]]]
        else:
            print("Your mode is not supported")
    return selected_ls

if __name__ == "__main__":
    # mode can be sec_ucb, lcb, mean_ucb, rand
    print(ret_selected_ls(num_response=12, num_sys=8, mode="sec_ucb"))