# MSU (Mathematical Solution Understanding) of LLM Evaluation
---
## 📌 Introduction

- **Purpose:** This repository provides tools to evaluate the MSU (Mathematical Solution Understanding) scores of LLM models.
- **Dataset Reference:** [MATH Dataset by Hendrycks et al, 2021](https://github.com/hendrycks/math)

## 🚀 How to Use

1. **Setup**: Install required packages using:
   ```bash
   pipenv install
2. API Configuration: This project relies on the [OpenAI APIs](https://openai.com/blog/openai-api). Ensure to create an `api_key.txt` file containing your API key.
3. Evaluation: Execute the MSU_scoring.py script for generating mathematical solutions and evaluating different models. For custom evaluations, you can modify the gpt and extract_notion functions inside `prompting.py`.
4. Validation: The `answer_validation.py` script helps in determining the accuracy of the answers.
5. Visualization: Use `plotting.py` for visual comparisons of MSU scores across models like `davinci-002`, `davinci-003`, `and ChatGPT`.

## Formulation of MSU

1. **Problem and Solution Definitions from dataset:** Begin with a problem set defined as:
   
   $$X = (x_1, x_2, \ldots, x_n)$$
   For each problem, corresponding human-generated solutions are represented as:
   $$S_h = (s_{h_1}, s_{h_2}, \ldots, s_{h_n})$$

2.  **Solution generation using LLMs:** Produce LLM generated solution \(S_l\) for each problem. This set is derived through prompt engineering, utilizing various LLMs:
   $$S_l = LLM(prompt_{solving}, X) = (s_{l_1}, s_{l_2}, \ldots, s_{l_n})$$

3. **Keyword extraction from solutions:** From solutions in \(S_l\) and \(S_h\), extract keywords and vectorize them.
   $$K_{h}= LLM(promt_{extraction}, S_h)$$
   $$K_{l}= LLM(promt_{extraction}, S_l)$$

4. **Compute the embedding distance:** Evaluate \(K_h\) and \(K_l\) by computing the cosine similarity, which is the MSU score:
   $$\text{MSU}(X, S_h, LLM)  = \frac{{K_h \cdot K_l}}{{\|K_h\|_2 \cdot \|K_l\|_2}}$$

---
## Citation

> Hendrycks, Dan et al. (2021). Measuring Mathematical Problem Solving With the MATH Dataset. *NeurIPS*.
