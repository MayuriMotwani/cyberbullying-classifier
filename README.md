#  Cyberbullying Comment Classifier

A **Machine Learning web application** built with **Streamlit** that detects and classifies cyberbullying comments on social media posts.  
This project uses Logistic Regression and Random Forest models trained on real-world cyberbullying datasets.

---

##  Features
- Predicts if a given text contains **cyberbullying or not**.
- Classifies the type of cyberbullying (e.g., Religion, Age, Gender, Ethnicity, etc.).
- Built using **Python**, **Scikit-learn**, and **Streamlit**.
- Real-time web interface for instant predictions.

---

##  Project Structure
cyberbullying-classifier/
├── app.py # Streamlit app file
├── artifacts/ # Saved model files
│ ├── tweets_LR.joblib
│ ├── tweets_RF.joblib
│ ├── tweets_vectorizer.joblib
│ ├── vectorizer_fast.joblib
│ ├── LogisticRegression_fast.joblib
│ ├── RandomForest_fast.joblib
├── cyberbullying_tweets.csv # Dataset 1
├── cyberbullying_data.csv # Dataset 2
├── requirements.txt # Python dependencies
└── README.md # Project description (this file)



---

##  Models Used
- **Logistic Regression** — Fast and accurate text classifier  
- **Random Forest** — Robust ensemble learning model  
- **TfidfVectorizer** — Converts comments into numerical features  

---

##  Dataset
The datasets used are publicly available:
- [Cyberbullying Tweets Dataset (Kaggle)](https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification)
- Custom preprocessed dataset: `cyberbullying_data.csv`

---

## ⚙️ Installation & Usage

### 1️⃣ Clone the repository
```bash
git clone https://github.com/MayuriMotwani/cyberbullying-classifier.git
cd cyberbullying-classifier
2️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
3️⃣ Run the Streamlit app
bash
Copy code
streamlit run app.py
Now open your browser and go to:
👉 http://localhost:8501

☁️ Deployment (Optional)
You can deploy this project for free on Streamlit Cloud:

Go to https://share.streamlit.io

Connect your GitHub account

Select this repository

Click Deploy

Your live app will look like:
https://cyberbullying-classifier.streamlit.app

🛠️ Requirements
All required Python libraries are listed in requirements.txt, including:

streamlit

pandas

scikit-learn

joblib

nltk

👩‍💻 Author
Mayuri Motwani
B.Tech in Computer Science Engineering
Machine Learning Project
📧 Project: Cyberbullying Comment Classifier

📜 License
This project is licensed under the MIT License —
you are free to use, modify, and share it with proper attribution.


