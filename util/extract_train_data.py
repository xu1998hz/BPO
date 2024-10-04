from transformers import AutoTokenizer
from datasets import load_dataset
import jsonlines

task="harmfulness"
tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b')

if task=="tldr":
    max_inp_length=512
    max_gen_length=128

    # extract training (10,000), sampling (10,000), validation (500) and testing data (500)
    data = load_dataset('CarperAI/openai_summarize_tldr')
    train_data_ls = []
    for ele in data['train']:
        cur_len = len(tokenizer.tokenize(ele['prompt']))
        if cur_len <= 512:
            train_data_ls+=[ele]

    print(len(train_data_ls))

    with jsonlines.open('train.jsonl', 'w') as writer:
        writer.write_all(train_data_ls[:65215])

    with jsonlines.open('sampling.jsonl', 'w') as writer:
        writer.write_all(train_data_ls[65215:])

    vali_data_ls = []
    for ele in data['valid']:
        cur_len = len(tokenizer.tokenize(ele['prompt']))
        if cur_len <= 512:
            vali_data_ls+=[ele]

    print(len(vali_data_ls))

    with jsonlines.open('dev.jsonl', 'w') as writer:
        writer.write_all(vali_data_ls[:1000])

    test_data_ls = []
    for ele in data['test']:    
        cur_len = len(tokenizer.tokenize(ele['prompt']))
        if cur_len <= 512:
            test_data_ls+=[ele]

    print(len(test_data_ls))

    with jsonlines.open('test.jsonl', 'w') as writer:
        writer.write_all(test_data_ls[:1000])

elif task=="hh":
    src_len_ls=[]
    tar_len_ls=[]
    count = 0
    dataset = load_dataset("HuggingFaceH4/helpful-anthropic-raw")['train']
    all_data=[]
    for ele in dataset:
        cur_src_len=len(tokenizer.tokenize(ele['instruction']))
        cur_tar_len=len(tokenizer.tokenize(ele['demonstration']))
        src_len_ls+=[cur_src_len]
        tar_len_ls+=[cur_tar_len]
        if cur_src_len<=64 and cur_tar_len<=128:
            count+=1
            all_data+=[{'prompt': f"{ele['instruction']} Assistant: ", "label": ele['demonstration']}]

    with jsonlines.open('sft_hh_data/train.jsonl', 'w') as writer:
        writer.write_all(all_data[:10000])

    with jsonlines.open('sft_hh_data/sampling.jsonl', 'w') as writer:
        writer.write_all(all_data[10000:20000])

    with jsonlines.open('sft_hh_data/dev.jsonl', 'w') as writer:
        writer.write_all(all_data[20000:20500])

    with jsonlines.open('sft_hh_data/test.jsonl', 'w') as writer:
        writer.write_all(all_data[20500:21000])

    print(sum(src_len_ls)/len(src_len_ls))
    print(max(src_len_ls))
    print(sum(tar_len_ls)/len(tar_len_ls))
    print(max(tar_len_ls))
    print(count)

elif task=="harmfulness":
    dataset = load_dataset("HuggingFaceH4/hh-rlhf-h4")['train']
    all_data = []
    train_src_len_ls=[]
    train_tar_len_ls=[]
    test_src_len_ls=[]
    test_tar_len_ls=[]
    train_count,test_count = 0,0
    for cur_data in dataset['chosen']:
        cur_ls = []
        for ele in cur_data:
            cur_ls+=[f"{ele['role']}: {ele['content']}"]

        cur_src_len=len(tokenizer.tokenize(" ".join(cur_ls[:-1])+" assistant: "))
        cur_tar_len=len(tokenizer.tokenize(cur_data[-1]['content']))
        train_src_len_ls+=[cur_src_len]
        train_tar_len_ls+=[cur_tar_len]

        if cur_src_len <= 256 and cur_tar_len<=128:
            all_data+=[{"prompt": " ".join(cur_ls[:-1])+" assistant: ", "label": cur_data[-1]['content']}]
            train_count+=1

    print("Train portion:")
    print(sum(train_src_len_ls)/len(train_src_len_ls))
    print(max(train_src_len_ls))
    print(sum(train_tar_len_ls)/len(train_tar_len_ls))
    print(max(train_tar_len_ls))
    print(train_count)
    print()
    print('-'*100)
    print()
    
    with jsonlines.open('sft_harm_data/train.jsonl', 'w') as writer:
        writer.write_all(all_data[:10000])

    with jsonlines.open('sft_harm_data/sampling.jsonl', 'w') as writer:
        writer.write_all(all_data[10000:20000])

    test_data = []
    dataset = load_dataset("HuggingFaceH4/hh-rlhf-h4")['test']
    for cur_data in dataset['chosen']:
        cur_ls = []
        for ele in cur_data:
            cur_ls+=[f"{ele['role']}: {ele['content']}"]

        cur_src_len=len(tokenizer.tokenize(" ".join(cur_ls[:-1])+" assistant: "))
        cur_tar_len=len(tokenizer.tokenize(cur_data[-1]['content']))
        test_src_len_ls+=[cur_src_len]
        test_tar_len_ls+=[cur_tar_len]

        if cur_src_len <= 256 and cur_tar_len<=128:
            test_data+=[{"prompt": " ".join(cur_ls[:-1])+" assistant: ", "label": cur_data[-1]['content']}]
            test_count+=1

    print("Test portion:")
    print(sum(test_src_len_ls)/len(test_src_len_ls))
    print(max(test_src_len_ls))
    print(sum(test_tar_len_ls)/len(test_tar_len_ls))
    print(max(test_tar_len_ls))
    print(test_count)
        
    with jsonlines.open('sft_harm_data/dev.jsonl', 'w') as writer:
        writer.write_all(test_data[0:500])

    with jsonlines.open('sft_harm_data/test.jsonl', 'w') as writer:
        writer.write_all(test_data[500:1000])
else:
    print("Task is not supported!")
    exit(1)