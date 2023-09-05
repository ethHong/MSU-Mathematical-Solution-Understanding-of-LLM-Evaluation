import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import openai
import time

f = open("api_key.txt", "r") #Please load your API Keyhere
openai.api_key = f.readline()
new_df = pd.read_csv("output/scoring_output.csv")

def model(input_prompt, max_length = 256, model="text-davinci-003"):
    """Generate solution to question based on input prompt"""
    response = openai.Completion.create(
        model=model,
        prompt=input_prompt,
        temperature=0.7,
        max_tokens=max_length,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response["choices"][0]["text"]

#Evaluation prompt engineering
def evaluate(solution_original, solution_candidate):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": """You are validator of mathematical solutions. I will give you original solution, and a candidate solution. 
                Please evaluate if the answer is correct or not.
                """},
                {"role": "assistant", "content": "Okay, I'll evaluate of the answer is correct or not. I'll only say True or False. "},
                {"role": "user", "content": "This is the origianl solution: {}".format(solution_original)},
                {"role": "user", "content": "This is the candidate solution: {}".format(solution_candidate)},
                {"role": "user", "content": "Is answer of candidate solution correct? Tuplize your response: (True/False)"}
                
        ]
    
    )
    return response["choices"][0]["message"]["content"]

colnames = [i.split("_notion_SCORE")[0] for i in new_df.columns if i.split("_")[-1]=="SCORE"]

for col in tqdm(colnames):
    print("proecssing {}...".format(col))
    Accurate = []
    for(solution_original, solution_candidate) in zip(tqdm(new_df["solution"].values), new_df[col].values):
        try:
            result = evaluate(solution_original, solution_candidate)
        except:
            time.sleep(60)
            result = evaluate(solution_original, solution_candidate)
            
        Accurate.append(result)
    new_df[col + "_accurate"] = Accurate

for col in tqdm(colnames):
    print("proecssing {}...".format(col))
    new_df[col + "_accurate"] = new_df[col + "_accurate"].apply(lambda x : True if "True" in x else False)
    
#Plotting
models = ["davinci2", "davinci3", "chatgpt"]
prompts = ["zeroshot", "fewshot", "zeroshotCOT"]
from prompting import *

data = {
    "model": [],
    "prompt" : [],
    "score" : [],
    "accuracy" : []
    
}

for m in models:
    for p in prompts:
        grouped_score = "gpt_{}_{}_solution_notion_SCORE".format(m, p)
        grouped_accuracy = "gpt_{}_{}_solution_accurate".format(m, p)
        temp = new_df[[grouped_score, grouped_accuracy]]
        
        score = temp[grouped_score].mean()
        accuracy = temp[grouped_accuracy].sum()/temp.shape[0]
        
        data["model"].append(m)
        data["prompt"].append(p)
        data["score"].append(score)
        data["accuracy"].append(accuracy)
        
styles = {'zeroshot': 'o', 'fewshot': 's', 'zeroshotCOT': 'v'}
sns.scatterplot(data=data, x="score", y="accuracy", hue="model", style="prompt", markers = styles,s = 100)
plt.title("Evaluation score and accuracy of answers")
plt.savefig("output/stats.pdf")