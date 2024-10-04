from transformers import AutoTokenizer
from datasets import load_dataset

max_inp_length=512
max_gen_length=128

def formatting_prompts_func(example):
    text = f"SUBREDDIT: {example['subreddit']}\nPOST: {example['content']}\nPlease summarize the post by given subreddit: " # {example['summary']}
    return text 

def data_analysis(tokenizer, data):
    count = 0
    more_inp, less_inp = 0, 0
    more_gen, less_gen = 0, 0

    for ele in data['train']:
        gen_cur_len = len(tokenizer.tokenize(ele['summary'])) 
        inp_cur_len = len(tokenizer.tokenize(formatting_prompts_func(ele)))
        
        if count > 50000:
            break
        count+=1
        
        if inp_cur_len > max_inp_length:
            more_inp+=1
        else:
            less_inp+=1

        if gen_cur_len > max_gen_length:
            more_gen+=1
        else:
            less_gen+=1

    print(f"inp > {max_inp_length}: ", more_inp)
    print(f"inp <= {max_inp_length}: ", less_inp)
    print(f"gen > {max_gen_length}: ", more_gen)
    print(f"gen <= {max_gen_length}: ", less_gen)

# extract training (10,000), sampling (10,000), validation (500) and testing data (500)
data = load_dataset('/share/edc/home/wendaxu/tldr-17')
tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b')

total_1 = 0
count_1 = 0
surpass_1 = 0
for ele in data['train']:
    cur_len = len(tokenizer.tokenize(formatting_prompts_func(ele)))
    total_1 += cur_len
    count_1 += 1
    if cur_len > 512:
        surpass_1 += 1
    if count_1 == 10000:
        break
    
print(total_1/count_1)
print(count_1)
print(surpass_1)
print()

total_2 = 0
count_2 = 0
surpass_2 = 0
for ele in data['train']:
    cur_len = len(tokenizer.tokenize(formatting_prompts_func(ele)))
    total_2 += cur_len
    count_2 += 1
    if cur_len > 512:
        surpass_2 += 1
    if count_2 == 20000:
        break
    
print((total_2-total_1)/(count_2-count_1))
print(count_2)
print(surpass_2-surpass_1)
print()

total_3 = 0
count_3 = 0
surpass_3 = 0
for ele in data['train']:
    cur_len = len(tokenizer.tokenize(formatting_prompts_func(ele)))
    total_3 += cur_len
    count_3 += 1
    if cur_len > 512:
        surpass_3 += 1
    if count_3 == 20500:
        break
    
print((total_3-total_2)/(count_3-count_2))
print(count_3)
print(surpass_3-surpass_2)
print()

total_4 = 0
count_4 = 0
surpass_4 = 0
for ele in data['train']:
    cur_len = len(tokenizer.tokenize(formatting_prompts_func(ele)))
    total_4 += cur_len
    count_4 += 1
    if cur_len > 512:
        surpass_4 += 1
    if count_4 == 21000:
        break
    
print((total_4-total_3)/(count_4-count_3))
print(count_4)
print(surpass_4-surpass_3)
print()