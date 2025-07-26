# ML_Final_Project

This project applies modern Natural Language Processing (NLP) techniques to classify news articles as **real** or **fake** using transformer-based models such as **BERT**, **RoBERTa**, and **DeBERTa**.

---

## 🚀 Project Overview

- Fine-tuned large transformer models (BERT, RoBERTa, DeBERTa) for binary text classification
- Improved macro F1 score from baseline **0.65** to **0.91** with advanced models and tuning
- Built end-to-end pipeline for preprocessing, model training, and evaluation
- Addressed class imbalance using threshold tuning, stratified k-fold CV, and active learning

---

## 📁 How to Run

### 1. Clone the repo

```bash
git clone https://github.com/colecperry/ML_Final_Project.git
cd ML_Final_Project

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python data_processing.py # Run data preprocessing on fake.csv, spits out test/train/val splits
python logistic_regression.py # Run Logistic Regression/TF IDF Model
python roberta.py # Run Roberta Model -> requires large GPU (used Google Colab)
python deberta.py # Run Deberta Model -> requires large GPU (used Google Colab)

📊 Evaluation Metrics
    - Macro F1 Score: Primary metric due to class imbalance
    - Confusion Matrix: Used to visualize true/false positives/negatives
    - Threshold Optimization: Applied to improve minority class recall

🧪 Models Tested
Model	  Vectorizer	      Macro F1	  Notes
Logistic  Regression/TF-IDF	  ~0.65	      Baseline
BERT	  Transformer	      ~0.87	      HuggingFace bert-base-uncased
RoBERTa	  Transformer	      ~0.89	      roberta-base
DeBERTa	  Transformer	       0.91	      Best-performing model




