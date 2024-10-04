def split_batch(batch, num_gpus):
    k, m = divmod(len(batch), num_gpus)
    return (batch[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(num_gpus))

batch = ['1']*16

print(list(split_batch(batch, num_gpus=7))[0])

# import google.generativeai as genai
# from google.generativeai.types import HarmCategory, HarmBlockThreshold
# import click

# def call_gemini(prompt_txt):
#     model = genai.GenerativeModel(model_name="gemini-pro")
#     completion = model.generate_content(
#         prompt_txt,
#         generation_config={"temperature": 1.0, "max_output_tokens": 1024},
#         safety_settings={
#             HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
#             HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
#             HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
#             HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
#         }

#     )
#     try:
#         return completion.text
#     except:
#         return "[BLOCKED]"

# @click.command()
# @click.option('-api_key') # AIzaSyDlKsAs90Ia4evu1BnXzCeommgHV6ZmPIs AIzaSyDl2WIk29kxjhw6XFOL3vrQZ62lW-VWfqo AIzaSyAviMYc26t1TyTmSxqRJfByJ8BBWHylghM AIzaSyAnoumVXX9TVyrhKg2N2pprG7XtzZz0eUw AIzaSyDR09F80B5iuTXZSXprPms7gpwsiIekv1k AIzaSyD_W8IU-ee0rv2-eKbQyXEAYcQOEPg-kg4 AIzaSyBt0TIOxBNRKycwSvB9vZeVTtav3ctKb-Q AIzaSyBmJhBJvDzPjVG1DX8iZXckp-gdnqiqqqg AIzaSyCq5yyWEkK0oAC1cAyiG2w1xpkinPIRIFc
# def main(api_key):
#     genai.configure(api_key=api_key)

#     for i in range(60):
#         answer = call_gemini('hello')
#         print(answer)

# if __name__ == "__main__":
#     main()  