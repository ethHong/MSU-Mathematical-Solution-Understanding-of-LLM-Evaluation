import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator
from scipy.stats import mannwhitneyu
from itertools import combinations

new_df = pd.read_csv("output/scoring_output.csv")

colnames = [i for i in new_df.columns if i.split("_")[-1]=="SCORE"]

pairs_zeroshot = [ i for i in colnames if "zeroshot_" in i]
pairs_fewshot = [ i for i in colnames if "fewshot_" in i]
pairs_COT = [ i for i in colnames if "COT" in i]

data = {}
for i in colnames:
    data[i] = new_df[i].values

def get_ttest_pair(pair, data ,figsize = (10, 10)):

    pairs = list(combinations(pair, 2))
    
    subcat_palette = sns.dark_palette("#8BF", reverse=True, n_colors=5)
    
    pvalues = [mannwhitneyu(data[i[0]], data[i[1]], alternative="two-sided").pvalue for i in pairs]
    formatted_pvalues = [f'p={pvalue:.2e}' for pvalue in pvalues]
    
    
    with sns.plotting_context('notebook', font_scale = 1.4):
        # Create new plot
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        #modify format of input data for sns plot
        input = pd.DataFrame()
        for i in pair:
            temp = pd.DataFrame({"Score" : new_df[pair][i].values, "Model": i})
            input = pd.concat([input, temp])
        

        # Plot with seaborn
        plotting_parameters = {
            "data" : input, 
            "x" : "Model", 
            "y" : "Score",
            'palette': subcat_palette[1:]
        }
        sns.violinplot(**plotting_parameters)

        # Add annotations
        annotator = Annotator(ax, pairs, **plotting_parameters)
        annotator.configure(text_format="simple")
        annotator.set_pvalues_and_annotate(pvalues)
        plt.ylim([0.65, 1.15])  
        plt.xticks([0, 1, 2], [i.split("_")[1] for i in pair])
        plt.title("Prompt type: {}".format(pair[0].split("_")[2]))
        plt.savefig("output/{}.png".format(pair[0].split("_")[2]))
        plt.show()

get_ttest_pair(pairs_zeroshot, data)
get_ttest_pair(pairs_fewshot, data)
get_ttest_pair(pairs_COT, data)

#Comparing Pairs
pairs_davinci2 = [ i for i in colnames if "davinci2_" in i]
pairs_davinci3 = [ i for i in colnames if "davinci3_" in i]
pairs_chatgpt = [ i for i in colnames if "chatgpt" in i]

data = {}
for i in colnames:
    data[i] = new_df[i].values

def get_ttest_pair(pair, data ,figsize = (10, 10)):

    pairs = list(combinations(pair, 2))
    
    subcat_palette = sns.dark_palette("#8BF", reverse=True, n_colors=5)
    
    pvalues = [mannwhitneyu(data[i[0]], data[i[1]], alternative="two-sided").pvalue for i in pairs]
    formatted_pvalues = [f'p={pvalue:.2e}' for pvalue in pvalues]
    
    
    with sns.plotting_context('notebook', font_scale = 1.4):
        # Create new plot
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        #modify format of input data for sns plot
        input = pd.DataFrame()
        for i in pair:
            temp = pd.DataFrame({"Score" : new_df[pair][i].values, "Model": i})
            input = pd.concat([input, temp])
        

        # Plot with seaborn
        plotting_parameters = {
            "data" : input, 
            "x" : "Model", 
            "y" : "Score",
            'palette': subcat_palette[1:]
        }
        sns.violinplot(**plotting_parameters)

        # Add annotations
        annotator = Annotator(ax, pairs, **plotting_parameters)
        annotator.configure(text_format="simple")
        plt.ylim([0.65, 1.15])  
        annotator.set_pvalues_and_annotate(pvalues)
        plt.xticks([0, 1, 2], [i.split("_")[2] for i in pair])
        plt.title("Model type: {}".format(pair[0].split("_")[1]))
        plt.savefig('output/{}.png'.format(pair[0].split("_")[1]))
        plt.show()

get_ttest_pair(pairs_davinci2, data)
get_ttest_pair(pairs_davinci3, data)
get_ttest_pair(pairs_chatgpt, data)