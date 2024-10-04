for mode in ["ucb_sec_ucb", "ucb_mean_ucb", "ucb_lcb", "ucb_rand", "ucb_ref"]:
    lines = open(f'/data/user_data/xixu/wendaxu/reward_std_{mode}_44.txt', 'r').readlines()
    lines = [ele[:-1] for ele in lines]

    count = 0
    total = 0
    gemma = 1.5
    for line in lines:
        rank_ls = [float(ele) for ele in line.split('\t')[0].split()]
        rank_std_ls = [float(ele) for ele in line.split('\t')[1].split()]
        utd_ls = [rank+gemma*rank_std for rank, rank_std in zip(rank_ls, rank_std_ls)]

        # get utd formulation
        max_rank_index = rank_ls.index(max(rank_ls))
        max_utd_index = utd_ls.index(max(utd_ls))
   
        if max_rank_index == max_utd_index:
            count+=1
        total+=1
    print(count)
    print(total)
    print(count/total)