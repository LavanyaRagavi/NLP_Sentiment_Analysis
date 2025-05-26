
# 🏥 Sentiment Analysis on Healthcare Reviews

## 📌 Project Overview
This project performs **sentiment analysis** on hospital patient feedback to classify whether a review is **positive** or **negative**. This is a **Natural Language Processing (NLP)** project using **Logistic Regression** after **SMOTE oversampling**, resulting in **93% accuracy**.

## 🎯 Objective
To build a **machine learning model** that can:
- Understand patient reviews
- Predict the sentiment behind each review (Positive or Negative)
- Help hospitals gain insights into patient satisfaction

## 🧠 Skills Applied
- Data Cleaning & Preprocessing
- Text Vectorization (TF-IDF)
- Class Imbalance Handling (SMOTE)
- Machine Learning Model Building
- Model Tuning (Grid Search)
- Evaluation Metrics & Visualization
- Confusion Matrix and ROC-AUC Curve

## 🗃️ Dataset
- **Source**: [Hospital Reviews Dataset - Kaggle](https://www.kaggle.com)
- **Shape**: `996 rows × 4 columns`
- **Features**:
  - `Feedback`: Text review
  - `Sentiment Label`: 1 (Positive), 0 (Negative)
  - `Ratings`: Rating out of 5
  - `Unnamed: 3`: Dropped due to nulls

## 🔧 Data Preprocessing
- Removed null and unnecessary columns
- Cleaned and tokenized text
- Converted text to lowercase, removed punctuations and stopwords
- Used **TF-IDF Vectorizer** to convert text to numerical format

## ⚖️ Handling Class Imbalance
- Positive Reviews: 728
- Negative Reviews: 268
- Applied **SMOTE (Synthetic Minority Oversampling Technique)** to balance the dataset before model training

## 🔍 Model Building
- **Model Used**: Logistic Regression
- Also tried Random Forest & Grid Search, but best result with Logistic Regression
- Final Accuracy: **93%**

## 📊 Evaluation Metrics

**Confusion Matrix**:
```
[[ 49   9]
 [ 16 126]]
```

**Performance Metrics**:
- Accuracy: `93%`
- Precision: `93.3%`
- Recall: `88.7%`
- F1 Score: `90.9%`
- ROC-AUC Score: (calculated in model)

## 📈 Visualizations
- Word cloud for frequent words
- Bar plot for sentiment distribution
- Confusion matrix heatmap
- ROC-AUC curve

## 📦 Project Structure
```
├── hospital.csv
├── Sentiment_Analysis_Model.ipynb
├── sentiment_model.pkl
└── README.md
```

   Open `Sentiment_Analysis_Model_nlp.ipynb` in Jupyter Notebook or Google Colab
