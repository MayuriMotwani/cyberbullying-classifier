# 🧠 Cyberbullying Comment Classifier

A **machine learning project** designed to detect and classify **cyberbullying comments** on social media posts using **TF-IDF features** and two optimized models — **Logistic Regression** and **Random Forest**.  

Cyberbullying is a growing concern in digital communication, affecting mental health and social behavior. This project demonstrates an **automated NLP pipeline** that can classify comments into bullying categories such as **age, gender, religion, ethnicity**, and more, achieving a **best accuracy of ~65%**.

---

## 🌍 Dataset Source

This project uses a combination of two publicly available datasets:

- `cyberbullying_tweets.csv` (~5000 samples)  
- `cyberbullying_dataset.csv` (~3000 samples)  

**Preprocessing and treatment:**

- Merged both datasets for a balanced representation  
- Removed duplicates and missing entries  
- Text normalization (lowercasing, punctuation removal)  
- Stopword removal using NLTK  
- TF-IDF vectorization for feature extraction  

*Links to original datasets:*  
- [Cyberbullying Tweets Dataset](https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification)  
- [Cyberbullying Dataset](https://www.kaggle.com/datasets/saurabhshahane/cyberbullying-dataset) 
---

## ⚙️ Methods

The overall workflow:

**Text preprocessing → TF-IDF Vectorization → Model Training → Evaluation**

**Models Used:**

| Model | Description | Reason for Choice |
|-------|-------------|-----------------|
| Logistic Regression | Linear model for classification | Fast, interpretable, good baseline |
| Random Forest | Ensemble of decision trees | Handles non-linear patterns, robust to overfitting |

**Alternative approaches considered:** SVM, Naive Bayes, deep learning (LSTM/BERT embeddings) — Logistic Regression + Random Forest chosen for **speed, interpretability, and simplicity**.

<p align="center">
  <img src="images/model_dark.png" width="85%" alt="Model Workflow Diagram">
</p>

---

## 📊 Exploratory Data Analysis

Class distribution and basic EDA insights:

<p align="center">
  <img src="images/eda_dark.png" width="85%" alt="EDA Visualization">
</p>

- Classes are somewhat imbalanced, requiring careful evaluation of minority classes.  
- Text length and keyword frequency were analyzed to guide preprocessing.

---

## 📈 Experiments and Results

**Model performance comparison:**

| Model | Accuracy | Strengths |
|:------|:--------|:-----------|
| Logistic Regression | **64.73%** | Fast training, good generalization |
| Random Forest | **58.07%** | Handles complex patterns, better recall for minority classes |

<p align="center">
  <img src="images/confusion_dark.png" width="80%" alt="Confusion Matrix Visualization">
</p>

**Key Observations:**

- Logistic Regression performed best overall.  
- Random Forest struggled with unbalanced labels but showed strong recall in some minority classes.  
- Text normalization and stopword removal significantly improved accuracy.  
- TF-IDF vectorization captured the most informative features for classification.

---

## 🚀 Steps to Run the Code

1. **Run the main pipeline:**

```bash
python main_pipeline.py
Or regenerate visuals and README documentation:

bash
Copy code
python generate_images_and_readme.py
Outputs:

artifacts/ → trained models

images/ → visualization set

README.md → updated project summary

📁 Project Structure
sql
Copy code
cyberbullying-classifier/
├── artifacts/
│   ├── LogisticRegression_fast.joblib
│   ├── RandomForest_fast.joblib
│   └── vectorizer_fast.joblib
├── images/
│   ├── eda_dark.png
│   ├── model_dark.png
│   └── confusion_dark.png
├── cyberbullying_tweets.csv
├── cyberbullying_dataset.csv
├── generate_images_and_readme.py
├── main_pipeline.py
└── README.md

🧩 Tech Stack
Python 3.x
Pandas, NumPy, Scikit-learn
Matplotlib / Seaborn
NLTK
Joblib

🧠 Author
Mayuri Motwani
B.Tech, Computer Science Engineering — Data Science Lab
✨ Passionate about AI, NLP, and social good applications

🏁 Conclusion
Logistic Regression achieved the best accuracy (~65%) for cyberbullying classification.
Random Forest shows promise for complex patterns but is sensitive to class imbalance.
Proper preprocessing (text normalization, stopword removal) and TF-IDF features are crucial.
Future improvements include:
BERT or LSTM-based embeddings
Real-time Streamlit dashboard for moderation
Bias and fairness analysis in NLP models

📚 References
Cyberbullying Tweets Dataset
Cyberbullying Dataset
Scikit-learn documentation: https://scikit-learn.org/
NLTK library: https://www.nltk.org/
Matplotlib: https://matplotlib.org/
Seaborn: https://seaborn.pydata.org/
