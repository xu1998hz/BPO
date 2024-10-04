import google.generativeai as genai
import json
import random
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from tqdm import tqdm
import click
import math

genai.configure(api_key="AIzaSyD6TPDOsho_SsIGneOHNLjAyN07JCGnwyk")

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

@click.command()
@click.option('-file_name', help='sample_10000_12000.json')
@click.option('-iter_index', type=int)
@click.option('-mode', type=str)
@click.option('-mode_2', type=str)
def main(file_name, iter_index, mode, mode_2):
    data = json.load(open(file_name))
    prompt_ls, better_ls, worse_ls = [], [], []

    with tqdm(total=len(data)) as pbar:
        for cur_index, (key, response_ls) in enumerate(data.items()):
            post = key.split('\nPlease summarize the post by given subreddit: ')[0]
            if mode == "rand":
                selected_2_ls = random.sample(response_ls, 2)
            elif mode == "uncertainty":
                selected_2_ls = [response_ls[selected_index[cur_index][0]], response_ls[selected_index[cur_index][1]]]

            prompt_txt= \
            f"""Which of the following summaries does a better job of summarizing the most \
            important points in the given forum post, without including unimportant or \
            irrelevant details? A good summary is both precise and concise.
            Post:
            {post}
            Summary A:
            {selected_2_ls[0]}
            Summary B:
            {selected_2_ls[1]}
            FIRST provide a one-sentence comparison of the two summaries, explaining which \
            you prefer and why. SECOND, on a new line, state only "A" or "B" to indicate your \
            choice. Your response should use the format:
            Comparison: <one-sentence comparison and explanation>
            Preferred: <"A" or "B">"""
            
            processs=True
            while processs:
                try:
                    answer = call_gemini(prompt_txt)
                    processs=False
                except:
                    print("An error occurred! Rerunning")

            if 'Preferred: ' in answer:
                if answer.split('Preferred: ')[1] == 'A':
                    prompt_ls+=[key]
                    better_ls+=[selected_2_ls[0]]
                    worse_ls+=[selected_2_ls[1]]
                elif answer.split('Preferred: ')[1] == 'B':
                    prompt_ls+=[key]
                    better_ls+=[selected_2_ls[1]]
                    worse_ls+=[selected_2_ls[0]]
            
            pbar.update(1)

    final_dict = {"prompt": prompt_ls, "chosen": better_ls, "rejected": worse_ls}
    
    with open(f'{mode}_{mode_2}_rank_{iter_index}.json', 'w') as f:
        json.dump(final_dict, f)
        print("File is saved!")

if __name__ == "__main__":
    main()
    