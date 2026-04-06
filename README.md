# Spam Detection using Naive Bayes (Custom + Sklearn)

## 📌 Overview
This project implements a spam detection system using the Enron Email Dataset.

Two implementations are provided:
1. Sklearn-based pipeline (baseline)
2. Custom Multinomial Naive Bayes (from scratch)

---

## ⚙️ Approach

### Data Processing
- Combined Subject + Message
- Cleaned text (lowercase, removed special characters)
- Handled missing values

### Feature Extraction
- TF-IDF Vectorization

### Models
- Sklearn MultinomialNB (baseline)
- Custom Multinomial Naive Bayes

---

## 📊 Results

Accuracy: ~98%

| Metric | Value |
|------|------|
| Precision | ~0.98–0.99 |
| Recall | ~0.98–0.99 |
| F1-score | ~0.98 |

---

## 🧠 Key Learning

- Implemented Naive Bayes from scratch
- Understood probability calculations and smoothing
- Learned difference between precision, recall, and F1-score
- Worked with real-world noisy dataset

---

## 📁 Files

- train_sklearn_nb.py → sklearn implementation
- custom_main.py → custom Naive Bayes implementation

---

## 🚀 Future Work

- Compare with Logistic Regression, SVM
- Add time-based evaluation (concept drift)
- Implement BERT-based classifier

---

## 💻 Tech Stack
Python, NumPy, Pandas, Scikit-learn