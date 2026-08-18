# ML Assignment 2: Handwritten Digit Classification

## a. Problem Statement

Build and evaluate multiple machine-learning classification models on one public classification dataset and expose the trained models through an interactive Streamlit application.

## b. Dataset Description

**Dataset:** Optical Recognition of Handwritten Digits

**Source:** UCI Machine Learning Repository

The UCI dataset is a multivariate classification dataset containing 64 integer features derived from handwritten digit images. The UCI repository reports 5,620 instances and 64 features. This implementation uses `sklearn.datasets.load_digits()`, which scikit-learn documents as a copy of the test set of the UCI handwritten-digits dataset. The copy used here contains **1,797 instances, 64 features and 10 classes (digits 0-9)**.

Each sample is an 8x8 handwritten digit image flattened into 64 pixel-intensity features. Feature values range from 0 to 16.

This satisfies the assignment minimum of 12 features and 500 instances.

## c. Github Repository Link

**https://github.com/rajarshikgupta/ML_Assignment_2/**

## d. Models Used

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)

Because this is a multiclass classification problem, AUC uses one-vs-rest (OvR) with weighted averaging. Precision, Recall and F1 use weighted averaging.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9722 | 0.9991 | 0.9724 | 0.9722 | 0.9722 | 0.9692 |\n| Decision Tree | 0.8250 | 0.9028 | 0.8241 | 0.8250 | 0.8237 | 0.8057 |\n| kNN | 0.9639 | 0.9951 | 0.9648 | 0.9639 | 0.9636 | 0.9600 |\n| Naive Bayes | 0.8111 | 0.9707 | 0.8480 | 0.8111 | 0.8151 | 0.7940 |\n| Random Forest | 0.9694 | 0.9992 | 0.9701 | 0.9694 | 0.9692 | 0.9662 |\n
### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Provides a strong multiclass baseline after feature standardization and performs well on the pixel features. |
| Decision Tree | Captures nonlinear relationships but is more sensitive to individual tree structure than the ensemble model. |
| kNN | Performs strongly because similar handwritten digits tend to have similar pixel patterns in feature space. |
| Naive Bayes | Provides a fast probabilistic baseline, but its conditional-independence assumption limits its ability to model relationships between pixels. |
| Random Forest (Ensemble) | Combines multiple decision trees and provides strong overall performance on the digit classification task. |
| **Overall Winner** | **Logistic Regression**, based primarily on the highest weighted F1 score, with MCC and AUC considered as supporting metrics. |

## Streamlit Application Features

- Dataset upload option using CSV
- Model selection dropdown
- Accuracy, AUC, Precision, Recall, F1 and MCC
- Confusion matrix
- Classification report
- Prediction preview

## Project Files

- `app.py`
- `model_training.py`
- `requirements.txt`
- `README.md`
- `test_data.csv`
- `model_metrics.csv`
- `feature_columns.csv`
- `model/` containing all five saved models

## Run Locally

```bash
pip install -r requirements.txt
python model_training.py
streamlit run app.py
```

## Streamlit Community Cloud Link

**https://mlassignment2-rajarshi.streamlit.app/**

## Dataset References

UCI Machine Learning Repository, Optical Recognition of Handwritten Digits, DOI: 10.24432/C50P49.

Scikit-learn `load_digits` documentation: this is a copy of the test set of the UCI handwritten-digits dataset.
