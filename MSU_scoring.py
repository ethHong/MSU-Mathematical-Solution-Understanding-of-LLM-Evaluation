import pandas as pd
import json
from tqdm import tqdm
import numpy as np
import random
import scipy.stats as stats
from prompting import *
from dataload import *
import matplotlib.pyplot as plt
from openai.error import RateLimitError

category = ["algebra", "prealgebra", "intermediate_algebra"]

# Pre-processing dataset
questions_train = {}
questions_test = {}
for cat in category:
    questions_train[cat] = get_questions(cat, path_train)
    questions_test[cat] = get_questions(cat, path_test)
    
input_data = {}
for type in category:
    input_data[type] = {}
    for lev in range(1, 6):
        input_data[type][lev]=filter_by_level(lev, questions_train, type)

#Sample only 30 Questions per each
sampled = {}
for c in list(input_data.keys()):
    sampled[c]={}
    for lev in list(input_data[c].keys()):
        pool = input_data[c][lev]
        sampled_key = random.sample(pool.keys(), 30)
        s = {k: pool[k] for k in sampled_key}
        sampled[c][lev]=s
input_data = sampled

df = pd.Series()
for c in category:
    for lev in range(1, 6):
        problems = input_data[c][lev]
        for index in list(problems.keys()):
            problem = problems[index]
            df = pd.concat([df, pd.DataFrame.from_dict(columns = list(problem.keys()), data = {index :  list(problem.values())}, orient = "index")])

df = df.reset_index()[list(problem.keys())]

#Zero shot prompt - Solution generation
tqdm.pandas() 
df["gpt_davinci2_zeroshot_solution"] = df["problem"].progress_apply(lambda x: gpt(prompt_solution_generation_zeroshot.format(x), model="text-davinci-002"))
tqdm.pandas() 
df["gpt_davinci3_zeroshot_solution"] = df["problem"].progress_apply(lambda x: gpt(prompt_solution_generation_zeroshot.format(x), model="text-davinci-003"))
tqdm.pandas() 
df["gpt_chatgpt_zeroshot_solution"] = df["problem"].progress_apply(lambda x: chatgpt(prompt_solution_generation_zeroshot.format(x)))

#Few shot prompt - Solution generation
#Extracting few shot examples
prompts = {}
for i in ["Algebra", "Intermediate Algebra", "Prealgebra"]:
    prompts[i] = {}
    for lev in range(1, 6):
        prompts[i][lev] = {}
        temp = df.loc[df["type"]==i].loc[df["level"]=="Level "+str(lev)]
        prompts[i][lev]["problem"] = temp.sample()["problem"].values[0]
        prompts[i][lev]["solution"] = temp.sample()["solution"].values[0]
        
tqdm.pandas() 
df["gpt_davinci2_fewshot_solution"] = df["problem"].progress_apply(lambda x: gpt(prompt_solution_generation_fewshot.format(x), model="text-davinci-002"))

new_df = pd.DataFrame()
for i in ["Algebra", "Intermediate Algebra", "Prealgebra"]:
    print (i)
    for lev in range(1, 6):
        temp_df = df.loc[df["type"]==i].loc[df["level"]=="Level "+str(lev)]
        sampled_fewshot = prompts[i][lev]
        
        tqdm.pandas() 
        temp_df["gpt_davinci2_fewshot_solution"] = temp_df["problem"].progress_apply(lambda x: gpt(prompt_solution_generation_fewshot.format(sampled_fewshot["problem"], sampled_fewshot["solution"], x), model="text-davinci-002"))
        
        tqdm.pandas() 
        temp_df["gpt_davinci3_fewshot_solution"] = temp_df["problem"].progress_apply(lambda x: gpt(prompt_solution_generation_fewshot.format(sampled_fewshot["problem"], sampled_fewshot["solution"], x), model="text-davinci-003"))
        
        tqdm.pandas() 
        temp_df["gpt_chatgpt_fewshot_solution"] = temp_df["problem"].progress_apply(lambda x: chatgpt(prompt_solution_generation_fewshot.format(sampled_fewshot["problem"], sampled_fewshot["solution"], x)))
        new_df = pd.concat([new_df, temp_df])
        

#Zero Shot COT Prompt
tqdm.pandas() 
new_df["gpt_davinci2_zeroshotCOT_solution"] = new_df["problem"].progress_apply(lambda x: gpt(prompt_solution_generation_zeroshot_COT.format(x), model="text-davinci-002"))
tqdm.pandas() 
new_df["gpt_davinci3_zeroshotCOT_solution"] = new_df["problem"].progress_apply(lambda x: gpt(prompt_solution_generation_zeroshot_COT.format(x), model="text-davinci-003"))
tqdm.pandas() 
new_df["gpt_chatgpt_zeroshotCOT_solution"] = new_df["problem"].progress_apply(lambda x: chatgpt(prompt_solution_generation_zeroshot_COT.format(x)))

#Extract notions
solutions = ['solution',
       'gpt_davinci2_zeroshot_solution', 'gpt_davinci3_zeroshot_solution',
       'gpt_chatgpt_zeroshot_solution', 'gpt_davinci2_fewshot_solution',
       'gpt_davinci3_fewshot_solution', 'gpt_chatgpt_fewshot_solution',
       'gpt_davinci2_zeroshotCOT_solution',
       'gpt_davinci3_zeroshotCOT_solution',
       'gpt_chatgpt_zeroshotCOT_solution']

for s in solutions: 
    print(s)
    colname = s+"_notion"
    tqdm.pandas() 
    new_df[colname] = new_df[s].progress_apply(lambda x : chatgpt(prompt_notion_extraction.format(x)))
    
#Compute scores
notion_data = [
       'gpt_davinci2_zeroshot_solution_notion',
       'gpt_davinci3_zeroshot_solution_notion',
       'gpt_chatgpt_zeroshot_solution_notion',
       'gpt_davinci2_fewshot_solution_notion',
       'gpt_davinci3_fewshot_solution_notion',
       'gpt_chatgpt_fewshot_solution_notion',
       'gpt_davinci2_zeroshotCOT_solution_notion',
       'gpt_davinci3_zeroshotCOT_solution_notion',
       'gpt_chatgpt_zeroshotCOT_solution_notion']

human_solution = "solution_notion"

def evaluate_and_store(n, new_df, human_solution, evaluate_for_question):
    target = new_df[n].values
    comparing = new_df[human_solution].values

    target = [i.split(",") for i in target]
    comparing = [i.split(",") for i in comparing]

    out = []
    for N_g, N_h in zip(tqdm(target), comparing):
        try:
            out.append(evaluate_for_question(N_g, N_h)[0])
        except:
            print ("cooldown...")
            time.sleep(100)
            out.append(evaluate_for_question(N_g, N_h)[0])

    new_df[n + "_SCORE"] = out

# Usage:
models = ["gpt_davinci2_zeroshotCOT_solution_notion", 
         "gpt_davinci3_zeroshotCOT_solution_notion", 
         "gpt_chatgpt_zeroshotCOT_solution_notion"]

for n in models:
    evaluate_and_store(n, new_df, human_solution, evaluate_for_question)
new_df.to_csv("output/scoring_output.csv", index = False)