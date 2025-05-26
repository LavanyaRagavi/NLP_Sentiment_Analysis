# NLP_Sentiment_Analysis
## 🧠 Sentiment Analysis on Healthcare Reviews

### 📌 Problem Statement
Classify patient feedback as Positive (1) or Negative (0) based on text reviews using NLP techniques.

### 📊 Dataset
- Source: Hospital Reviews Dataset from Kaggle
- Total Records: 996
- Features: Feedback, Sentiment Label, Ratings

### 🔍 Techniques Used
- Text preprocessing (cleaning, stopwords, lemmatization)
- TF-IDF Vectorization
- SMOTE for class imbalance
- Model building with Logistic Regression and Random Forest
- Hyperparameter tuning using GridSearchCV

### 📈 Results
- **Best Model:** Logistic Regression
- **Accuracy:** 93%
- **Evaluation:** Precision, Recall, F1-score, ROC-AUC
- **Tools:** Python, scikit-learn, matplotlib, seaborn, joblib

### 🗂️ Files
- `sentiment_analysis.py`: Main code
- `logistic_regression_sentiment_model.pkl`: Final saved model
- `README.md`: Project overview
- `requirements.txt`: Python dependencies

### 📎 Future Improvements
- Expand dataset size
- Try advanced models like BERT for better accuracy
