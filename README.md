# 📰 Fake News Detection using Machine Learning

This project implements a machine learning-based Fake News Detection system using two classification models: **Multinomial Naive Bayes** and **Logistic Regression**. The goal is to predict whether a given news article is fake or real based on its textual content.

---


## 📖 Project Overview

With the widespread distribution of information online, fake news has become a significant issue. This project applies NLP and supervised learning to classify news as real or fake using:
- **TF-IDF Vectorization**
- **Naive Bayes Classifier**
- **Logistic Regression**

---

## 🛠️ Tech Stack

- **Python**
- **Scikit-learn**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Joblib / Pickle**

---

## 📊 Dataset

- **Source:** Kaggle / open-source fake news dataset
- **Columns:** `text`, `label`
  - `text`: The news content
  - `label`: 0 = Fake, 1 = Real

---

## ⚙️ Preprocessing

1. Remove null values
2. Convert to lowercase
3. Remove punctuation and stopwords
4. Apply **TF-IDF Vectorization**

---

## 🤖 Models Used

1. **Multinomial Naive Bayes**
2. **Logistic Regression**

---

## 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix

---

## 🧪 Confusion Matrices

Plots comparing the true vs predicted labels for both models.

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Example:
cm_nb = confusion_matrix(y_test, y_pred_nb)
disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb, display_labels=nb_model.classes_)
disp_nb.plot()

cm_lr = confusion_matrix(y_test, y_pred_lr)
disp_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr, display_labels=lr_model.classes_)
disp_lr.plot()
