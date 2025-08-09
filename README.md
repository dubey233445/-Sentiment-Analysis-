# Sentiment Analysis Project

## 📌 Overview
This project implements a **Sentiment Analysis** model to classify text reviews as **Positive** or **Negative**.  
The model was trained and tested on a balanced dataset and achieved **89.05% accuracy**.

---

## 📊 Model Performance
| Metric           | Negative Class | Positive Class | Macro Avg | Weighted Avg |
|------------------|---------------|---------------|-----------|--------------|
| Precision        | 0.90          | 0.88          | 0.89      | 0.89         |
| Recall           | 0.88          | 0.90          | 0.89      | 0.89         |
| F1-Score         | 0.89          | 0.89          | 0.89      | 0.89         |
| **Accuracy**     | -             | -             | **0.89**  | **0.89**     |

---

## ⚙️ Technologies Used
- **Python** 🐍
- **Pandas** for data manipulation
- **NumPy** for numerical computations
- **Scikit-learn** for machine learning model & evaluation
- **NLTK / spaCy** for text preprocessing (tokenization, stopword removal, etc.)

---

## 📂 Project Structure
├── notebooks/ # Jupyter notebooks for EDA & training

├── README.md # Project documentation

└── sentiment_nlp.py # Main script


---

## 🔍 How It Works
1. **Data Preprocessing**
   - Text cleaning (lowercasing, punctuation removal)
   - Tokenization
   - Stopword removal
   - Lemmatization

2. **Feature Extraction**
   - TF-IDF Vectorization

3. **Model Training**
   - Logistic Regression / Naive Bayes (customize based on your model)
   - Train-Test split (80%-20%)

4. **Evaluation**
   - Accuracy, Precision, Recall, F1-score
   - Classification Report

---

## 🚀 How to Run
1. **Clone the repository**
   ```bash
   git clone https://github.com/dubey233445/sentiment-analysis.git
   cd sentiment-analysis
📈 Future Improvements

Experiment with deep learning models (LSTM, BERT)

Add more preprocessing steps for slang & emojis

Deploy as a web application (Flask/Streamlit)

📜 License

This project is licensed under the MIT License.


