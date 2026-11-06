# A BERT-Based QA System for Pressure Ulcer Management and Healthcare Practitioner Training

An extractive question-answering system that answers clinical questions about **pressure ulcers**: definition, causes and risk factors, stages 1–4, prevention and treatment. A **DistilBERT** model is fine-tuned on a custom **SQuAD v2-style** dataset that I built for this project. The dataset includes unanswerable questions, so the model learns to abstain when the context does not contain the answer.

*Coursework for the Advanced Topics in Deep Learning module, MSc Artificial Intelligence, Liverpool John Moores University.*

## Dataset

- **Built manually:** I collected clinical information on pressure ulcer management and organised it into context passages, one per clinical topic.
- **Annotated in SQuAD v2.0 format,** with several question–answer pairs per context.
- **Size:** 4,727 question–answer pairs over 270 contexts (3,676 answerable, 1,051 unanswerable).
- **Expanded with paraphrased questions** to add variety in how questions are asked.

The notebook builds the dataset and saves it as `Pressure_Ulcer_dataset.json`. The JSON file is not included in this repository.

## Approach

All steps are in [bert_pressure_ulcer_qa.ipynb](bert_pressure_ulcer_qa.ipynb):

1. **Dataset construction:** build the contexts and QA pairs, expand them, and save them in SQuAD JSON format.
2. **Data loading:** load and validate the JSON.
3. **Pre-processing:**
   - flatten the dataset to one question per row;
   - tokenize it (max length 384, stride 128, sliding window for long contexts);
   - map each answer to start and end token positions, with unanswerable questions mapped to `[CLS]`.
4. **Training:** fine-tune `distilbert-base-uncased` with the Hugging Face `Trainer`.
   - 3 epochs, learning rate 2e-5, batch size 16, weight decay 0.01.
   - Early stopping on validation loss.
   - About 66 minutes of training.
5. **Evaluation:** measure Exact Match and F1 on a held-out test set.
6. **Results discussion:** strengths, limitations and possible improvements.

## Results

| Epoch | Train loss (avg) | Validation loss |
| --- | --- | --- |
| 1 | ~2.90 | 2.578 |
| 2 | ~2.52 | **2.534** (best, kept) |
| 3 | ~2.38 | 2.546 |

| Test set (473 samples) | Score |
| --- | --- |
| Exact Match (EM) | **50.11%** |
| F1 | **50.79%** |

**Key findings**

- **Abstention works:** the model often predicts `[CLS]` (no answer) for unanswerable questions, which matters in a clinical setting.
- **Validation loss was lowest at epoch 2:** it rose slightly at epoch 3, an early sign of overfitting, so early stopping kept the epoch 2 checkpoint.
- **EM and F1 are almost equal,** so predictions tend to be either fully right or fully wrong.

## Limitations

- **Small, single-author dataset.** It covers one clinical topic and has not been reviewed by clinicians.
- **Test results may be optimistic.** The paraphrased questions were added before the train/test split, so a test question can be a reworded version of a training question about the same context.
- **The model sometimes over-abstains:** it can predict `[CLS]` when an answer does exist. Correct "no answer" predictions on unanswerable questions also raise both EM and F1.
- **Answer spans are sometimes too long** and include extra surrounding text.
- **Not for clinical use.** This is a learning project, not a validated medical tool.

**Next steps:**

- split the data by context before adding paraphrases;
- grow the dataset and have clinicians review it;
- report scores separately for answerable and unanswerable questions;
- try larger or biomedical models such as BioBERT.

## How to run

1. Install the libraries:
   ```bash
   pip install -r requirements.txt
   ```
2. Open the notebook and run the cells in order. It builds the dataset, trains the model and evaluates it.

A GPU makes training much faster.

## Tools

Python · PyTorch · Hugging Face Transformers · Datasets · DistilBERT · NumPy · pandas · Jupyter
