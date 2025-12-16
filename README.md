# Farsi Textual Entailment (FarsTail) with Gemma-3-270m

This project explores Textual Entailment (TE) on the low-resource Farsi (Persian) language using the `google/gemma-3-270m` model. We evaluate the model's performance using three distinct approaches: Prompt Engineering (Zero-shot and Few-shot), traditional Sequence Classification Fine-tuning, and Causal Language Modeling (CL) Fine-tuning augmented with generated rationales (Chain-of-Thought style).

The goal is to analyze how different training paradigms and prompting strategies impact the TE task in a low-resource setting.

## Key Components

*   **Dataset:** [FarsTail](https://github.com/dml-qom/FarsTail.git) (A Persian Textual Entailment dataset).
*   **Base Model:** `google/gemma-3-270m`.
*   **Evaluation Metrics:** Precision, Recall, and Macro F1-score for the three-way classification (`e`=Entailment, `c`=Contradiction, `n`=Neutral).

##  Project Structure

| File | Description | Method |
| :--- | :--- | :--- |
| `APE.ipynb` | **Prompt Engineering:** Evaluates the *base* `gemma-3-270m` model using various Zero-shot and Few-shot prompts on the validation set. | APE/Prompting |
| `Fine_tuning.ipynb` | **Sequence Classification Fine-tuning:** Sets up a traditional classification head (`AutoModelForSequenceClassification`) over the Gemma backbone. It freezes the base layers and trains only the new classification head. | Seq. Classification |
| `data_augmentation.py` | **Data Augmentation:** A Python script using a powerful external LLM (`google/gemma-3-27b-it` via OpenRouter) to generate detailed Persian *reasons* (rationales) for a subset of the training data. | Data Augmentation |
| `APE_Fine_tuning_augmented_data.ipynb` | **CL Fine-tuning with Rationale:** Fine-tunes the `gemma-3-270m` for Causal Language Modeling (Instruction Tuning) using a prompt format that includes the generated reason before the final label: `[Input] + [Reason] + [LABEL] + [Label]`. | Causal LM / APE Fine-tuning |

##  Setup and Installation

The project uses standard Python libraries, primarily within a Google Colab environment with GPU access (recommended).

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mohammadentezari2001/LLM-Fine-Tuning-Textual-Entailment.git
    cd LLM-Fine-Tuning-Texual-Entailment
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
