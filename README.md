# Farsi Textual Entailment (FarsTail) with Gemma-3-270m

This project explores Textual Entailment (TE) on the low-resource Farsi (Persian) language using the `google/gemma-3-270m` model. We evaluate the model's performance using three distinct approaches: Prompt Engineering (Zero-shot and Few-shot), traditional Sequence Classification Fine-tuning, and Causal Language Modeling (CL) Fine-tuning augmented with generated rationales (Chain-of-Thought style).

The goal is to analyze how different training paradigms and prompting strategies impact the TE task in a low-resource setting.

## Key Components

*   **Dataset:** [FarsTail](https://github.com/dml-qom/FarsTail.git) (A Persian Textual Entailment dataset).
*   **Base Model:** `google/gemma-3-270m`.
*   **Evaluation Metrics:** Precision, Recall, and Macro F1-score for the three-way classification (`e`=Entailment, `c`=Contradiction, `n`=Neutral).

## 🗃️ Project Structure

| File | Description | Method |
| :--- | :--- | :--- |
| `APE.ipynb` | **Prompt Engineering:** Evaluates the *base* `gemma-3-270m` model using various Zero-shot and Few-shot prompts on the validation set. | APE/Prompting |
| `Fine_tuning.ipynb` | **Sequence Classification Fine-tuning:** Sets up a traditional classification head (`AutoModelForSequenceClassification`) over the Gemma backbone. It freezes the base layers and trains only the new classification head. | Seq. Classification |
| `data_augmentation.py` | **Data Augmentation:** A Python script using a powerful external LLM (`google/gemma-3-27b-it` via OpenRouter) to generate detailed Persian *reasons* (rationales) for a subset of the training data. | Data Augmentation |
| `APE_Fine_tuning_augmented_data.ipynb` | **CL Fine-tuning with Rationale:** Fine-tunes the `gemma-3-270m` for Causal Language Modeling (Instruction Tuning) using a prompt format that includes the generated reason before the final label: `[Input] + [Reason] + [LABEL] + [Label]`. | Causal LM / APE Fine-tuning |

## 🛠️ Setup and Installation

The project uses standard Python libraries, primarily within a Google Colab environment with GPU access (recommended).

1.  **Clone the repository:**
    ```bash
    git clone [Your-Repo-Link]
    cd [Your-Repo-Name]
    ```

2.  **Install dependencies:**
    ```bash
    !pip install -q transformers accelerate datasets pandas scikit-learn matplotlib
    # For data_augmentation.py, you will also need:
    # !pip install -q openai
    ```

3.  **Hugging Face Login:**
    To download the model weights and to run fine-tuning, log in to Hugging Face:
    ```python
    from huggingface_hub import notebook_login
    notebook_login()
    ```
    *(Requires a Hugging Face token with Read access.)*

4.  **OpenRouter API Key (for Augmentation):**
    The `data_augmentation.py` script requires an API key from [OpenRouter](https://openrouter.ai) to generate rationales using the large Gemma model.

##  Experimental Results Summary

### 1. Prompt Engineering (APE.ipynb)

The base Gemma-3-270m model, even with detailed Zero-shot and Few-shot prompting, showed extremely limited ability to perform Textual Entailment on the FarsTail dataset, indicating the need for fine-tuning or a larger base model.

| Prompt Type | Best Macro F1 |
| :--- | :--- |
| Zero-shot | **0.1905** (`zero_4`, `zero_5`) |
| Few-shot (6 examples) | **0.2500** (`few_2`) |

### 2. Sequence Classification Fine-tuning (`Fine_tuning.ipynb`)

This experiment followed a traditional fine-tuning approach by replacing the language modeling head with a sequence classification head (`num_labels=3`). To minimize training time, only the newly added classification head was trained (i.e., freezing all base layers).

| Metric | Result (Epoch 5) |
| :--- | :--- |
| Validation Accuracy | 0.4157 |
| **Macro F1-score** | **0.4127** |
| Trainable Params | 1,920 / 268,100,096 |

The low performance suggests that freezing all base layers and only training the randomly initialized classification head is insufficient for adapting the model to the TE task, even on a large dataset like FarsTail.

### 3. Causal LM Fine-tuning with Rationale (`APE_Fine_tuning_augmented_data.ipynb`)

This experiment used a Chain-of-Thought style prompt format, augmented with generated rationales (`[Reasoning] [LABEL] [Label]`), and treated the task as a Causal Language Modeling (CLM) task.

| Strategy | Trainable Parameters | Final Validation Loss |
| :--- | :--- | :--- |
| Last Layer Only (1 layer) | 5.57M (2.08%) | **2.761** |
| Half Model (4 layers) | 22.29M (8.32%) | 3.057 |

*Note: The model used is `google/gemma-3-270m` for CLM. The evaluation requires manual text generation and extraction, which is not fully automated in the provided notebook, but the validation loss suggests the *Last Layer Only* CLM approach performed slightly better in terms of predicting the sequence tokens.*
