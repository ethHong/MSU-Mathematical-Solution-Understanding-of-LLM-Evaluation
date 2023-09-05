import os
import json

path_train = "MATH/train"
path_test = "MATH/test"
category = [i for i in os.listdir(path_train) if i != ".DS_Store"]

# Get Questions with answers
def get_questions(c, path):
    """With filepath, load questions with names of questions"""
    p = path + "/{}".format(c)
    questions = [
        p + "/" + i for i in sorted(os.listdir(p), key=lambda x: int(x.split(".")[0]))
    ]

    output = {}

    for s in questions:
        with open(s, "rb") as f:
            q = json.load(f)
            output[questions.index(s)] = q

    return output


# paths for extracting notion
# path_output = "output_step1"
# processed_files = [i for i in os.listdir(path_output) if i != ".DS_Store"]


def get_processed_output(c, path):
    p = path
    questions = [
        p + "/" + i for i in sorted(os.listdir(p), key=lambda x: x.split(".")[0])
    ]

    output = {}
    for s in questions:
        with open(s, "rb") as f:
            q = json.load(f)
            index = [int(i) for i in list(q.keys())]
            for i in index:
                output[i] = {}
                output[i]["problem"] = q[i]["problem"]
                output[i]["solution"] = q[i]["solution"]
    return output


# Step 2- load data from step 1
