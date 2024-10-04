import google.generativeai as genai
import json
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from tqdm import tqdm
import click
from openai import OpenAI
import json
import random

def call_gemini(prompt_txt):
    model = genai.GenerativeModel(model_name="gemini-pro")
    completion = model.generate_content(
        prompt_txt,
        generation_config={"temperature": 1.0, "max_output_tokens": 1024},
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }

    )
    try:
        return completion.text
    except:
        return "[BLOCKED]"

def completions_with_backoff_openai(client, system_prompt, prompt_txt, model_type,n):
    response = client.chat.completions.create(
        model=model_type,  # "gpt-3.5-turbo", "gpt-4"
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": prompt_txt},
        ],
        temperature=0.0,
        max_tokens=1024,
        top_p=1,
        n=n,
    ).choices[0].message.content
    return response

@click.command()
@click.option('-file_name', help='sample_10000_12000.txt')
@click.option('-save_folder', type=str)
@click.option('-data_split', type=str)
@click.option('-save_name', type=str)
@click.option('-api_key', type=str)
@click.option('-api_source', type=str)
@click.option('-ref_addr', type=str, default=None)
@click.option('-task_type', type=str)
def main(file_name, save_folder, save_name, api_key, api_source, data_split, ref_addr, task_type):
    if api_source == "openai":
        client = OpenAI()
        system_prompt="You are a text generation evaluater."
        model_type="gpt-4-turbo" # "gpt-3.5-turbo-0125"
    else:
        genai.configure(api_key=api_key)

    data = open(file_name).readlines()
    prompt_ls, better_ls, worse_ls, scores_ls, ref_ls = [], [], [], [], []

    with open(f'sft_data/{data_split}.jsonl') as f:
        eval_dataset = [json.loads(line) for line in f]

    if ref_addr:
        ref_txt_ls = open(ref_addr, 'r').readlines()
        ref_txt_ls = [line[:-1] for line in ref_txt_ls]

    with tqdm(total=len(data)) as pbar:
        for cur_index, response in enumerate(data):
            score_a_b_dict = {"A": None, "B": None} 
            rand_index = random.sample([0, 1], 1)[0]
            # evaluate response against human written reference
            if ref_addr:
                cur_ref_txt = ref_txt_ls[cur_index]
            else:
                cur_ref_txt = eval_dataset[cur_index]['label']

            if rand_index == 0:
                selected_2_ls = [response, cur_ref_txt]
                score_a_b_dict["A"] = 1
                score_a_b_dict["B"] = 0
            else:
                selected_2_ls = [cur_ref_txt, response]
                score_a_b_dict["A"] = 0
                score_a_b_dict["B"] = 1

            if task_type == "tldr":
                prompt_txt= \
                f"""Which of the following summaries does a better job of summarizing the most \
                important points in the given forum post, without including unimportant or \
                irrelevant details? A good summary is both precise and concise.
                {eval_dataset[cur_index]['prompt']}
                Summary A:
                {selected_2_ls[0]}
                Summary B:
                {selected_2_ls[1]}
                FIRST provide a one-sentence comparison of the two summaries, explaining which \
                you prefer and why. SECOND, on a new line, state only "A" or "B" to indicate your \
                choice. Your response should use the format:
                Comparison: <one-sentence comparison and explanation>
                Preferred: <"A" or "B">"""
            elif task_type == "hh":
                prompt_txt= \
                f"""The most important part of the task is choosing which of the Assistant's responses is more helpful and honest.
                Please consider the responses carefully, and use your overall judgment. Ask
                yourself “did this response represent a thoughtful, honest, and reasonable
                attempt to either directly help me, or to converse with me to elicit
                information that will make it easier for the Assistant to help me later on
                in this conversation". Note that this is subtle - if the context is about
                about a difficult personal situation, a helpful response might involve
                showing empathy and understanding. But if the context is about something
                practical, a helpful response may be a concise response, and not one that
                prolongs the interaction with irrelevant questions and comments. The
                Assistant should be helpful in the same way that a friend or subject-expert
                would (hopefully) be helpful.
                Note response containing "Human:" and/or "Assistant:" that tries to extend
                the conversation should be considered as not helpful and honest.
                Given the context and the two responses choose the most helpful and honest
                response based on the definitions above.
                Context - How do I make fried tofu?
                Response A - Fried tofu isn’t a thing I know about. What do you want to make?
                Response B - Fried tofu can be fried
                FIRST provide a one-sentence comparison of the two summaries, explaining which you prefer and why. SECOND, on a new line, state only "A" or "B" to indicate your choice. Your response should use the format: Comparison: <one-sentence comparison and explanation> Preferred: <"A" or "B">"""

            try:
                if api_source == "google":
                    answer = call_gemini(prompt_txt)
                else:
                    answer = completions_with_backoff_openai(client, system_prompt, prompt_txt, model_type, 1)
            except:
                print("An error occurred! Rerunning")

            if 'Preferred:' in answer:
                if answer.split('Preferred:')[1].strip() == 'A':
                    prompt_ls+=[eval_dataset[cur_index]['prompt']]
                    better_ls+=[selected_2_ls[0]]
                    worse_ls+=[selected_2_ls[1]]
                    # score depends on where we place our response text
                    scores_ls+=[score_a_b_dict['A']]
                    if ref_addr:
                        ref_ls+=ref_txt_ls[cur_index]
                    else:
                        ref_ls+=[eval_dataset[cur_index]['label']]

                elif answer.split('Preferred:')[1].strip() == 'B':
                    prompt_ls+=[eval_dataset[cur_index]['prompt']]
                    better_ls+=[selected_2_ls[1]]
                    worse_ls+=[selected_2_ls[0]]
                    # score depends on where we place our response text
                    scores_ls+=[score_a_b_dict['B']]
                    if ref_addr:
                        ref_ls+=ref_txt_ls[cur_index]
                    else:
                        ref_ls+=[eval_dataset[cur_index]['label']]
                else:
                    print(answer)
                
                pbar.update(1)

    final_dict = {"prompt": prompt_ls, "chosen": better_ls, "rejected": worse_ls, "scores": scores_ls, "refs": ref_ls}
    
    with open(f'{save_folder}/{save_name}.json', 'w') as f:
        json.dump(final_dict, f)
        print("File is saved!")
    
    print("Win rate: ", sum(scores_ls)/len(scores_ls))

if __name__ == "__main__":
    main()