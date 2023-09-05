import numpy as np
import openai
from numpy import dot
from numpy.linalg import norm

f = open("api_key.txt", "r")  #Please load your API Key here
openai.api_key = f.readline()

# Prompting to extract notion with processed files:

prompt_notion_extraction = """
From a solution of mathematical problem, extract general mathematical concepts required in given solution.
- Only list up mathematical knowledge or concepts
- Use comma seperation

#
concepts: concept1, concept2, concept3 ...

#
solution: {}
concepts:
"""

# Step 2 - prompt for generating solution
prompt_solution_generation_zeroshot = """
Solve the question below - give solution and get the answer. 

Problem: {}
Solution:
"""
prompt_solution_generation_fewshot = """
Solve the question below - give solution and get the answer. 
#
    
Problem: {}
Solution: {}

#
Problem: {}
Solution:
"""

prompt_solution_generation_zeroshot_COT = """
Solve the question below - give solution and get the answer. 

Problem: {}
Solution: Let's think step by step.
"""


def filter_by_level(level, questions, type):
    """Get only questions of specific level"""
    questions = questions[type]
    out = {}
    for i in list(questions.keys()):
        if questions[i]["level"] == "Level {}".format(level):
            out[i] = questions[i]
    return out


def gpt(input_prompt, model="text-davinci-003"):
    """Generate solution to question based on input prompt"""
    response = openai.Completion.create(
        model=model,
        prompt=input_prompt,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    return response["choices"][0]["text"]


def chatgpt(input_prompt):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": input_prompt}]
    )

    return completion["choices"][0]["message"]["content"]


def extract_notion(input_prompt, solution):
    text = input_prompt.format(solution)
    return chatgpt(text)


### Evaluate scores
def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))


def get_embedding(text):
    out = openai.Embedding.create(input=str(text), engine="text-embedding-ada-002")[
        "data"
    ][0]["embedding"]
    return out


def compare(a, b):
    return cos_sim(get_embedding(a), get_embedding(b))


def evaluate_for_question(N_g, N_h):
    score = []
    for n_g in N_g:
        score.append(max([compare(n_g, i) for i in N_h]))
    return (np.mean(score), score)


def evaluate_precision_recall(N_g, N_h):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for n_g in N_g:
        max_score = max([compare(n_g, i) for i in N_h])
        if max_score >= 0.9:
            true_positives += 1
        else:
            false_positives += 1
    false_negatives = len(N_h) - true_positives

    if true_positives + false_negatives == 0:
        recall = "N/A"
    else:
        recall = true_positives / (true_positives + false_negatives)

    if (true_positives + false_positives) == 0:
        precision = "N/A"
    else:
        precision = true_positives / (true_positives + false_positives)

    if precision + recall == 0 or "N/A" in [precision, recall]:
        f1_score = "N/A"
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return {"precision": precision, "recall": recall, "f1_score": f1_score}

