# Persian Textual Entailment (FarsTail) using Gemma-3

This repository contains the implementation and evaluation of Large Language Model (LLM) prompting techniques (Zero-shot and Few-shot) for the Farsi Textual Entailment (Farsi NLI) task using the efficient **Gemma-3-270m** model. The goal is to classify the relationship between a *premise* and a *hypothesis* into one of three categories: Entailment ('e'), Contradiction ('c'), or Neutral ('n').

##  Setup and Dependencies

This project requires Python and several libraries, notably Hugging Face Transformers and PyTorch. It was run in a Google Colab environment utilizing a GPU (T4 ) for faster inference with the Gemma model.

1.  **Clone the Repository (Implicit):** Assuming this `README.md` is in the root of your project directory.
2.  **Install Libraries:**

    ```bash
    !pip install transformers torch pandas accelerate
    ```
3.  **Hugging Face Login:** The notebook includes a cell for `huggingface_hub.notebook_login()` to access the Gemma model, which may require a Hugging Face token.
4.  **Data Download:** The FarsTail dataset is cloned directly in the notebook:
    ```bash
    !git clone https://github.com/dml-qom/FarsTail.git
    ```

##  Methodology

### 1. Model Selection

The project utilizes the **`google/gemma-3-270m`** model from the Hugging Face hub, loaded using `AutoModelForCausalLM` and `AutoTokenizer` with `torch.bfloat16` precision and automatically mapped to the GPU (`device_map="auto"`).

### 2. Prompting Strategies

Five distinct Zero-shot prompts and five distinct Few-shot prompts were defined and tested. The Few-shot prompts use 6 manually selected examples from the FarsTail training set (examples shown in the notebook output).

**Labels and Definitions (in Farsi):**
- **e (Entailment):** The hypothesis can be logically inferred from the premise.
- **c (Contradiction):** The hypothesis logically contradicts the premise.
- **n (Neutral):** The hypothesis is neither necessarily true nor necessarily false based on the premise.

### 3. Evaluation

A custom function `precision_recall_f1` was used to calculate standard metrics (Precision, Recall, F1-Score) for each class ('e', 'c', 'n') and the overall **Macro F1 score**. Evaluation was performed on a small sample (`n=5`) of the validation dataset (`val`) for initial prompt tuning.
