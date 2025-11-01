# ============================================================
# ⚡ CYBERBULLYING DETECTION APP (Dual Model Families)
# ============================================================

import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords

# ------------------------------------------------------------
# 📦 NLTK setup
# ------------------------------------------------------------
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

# ------------------------------------------------------------
# 🧹 Text cleaning
# ------------------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\w+|#", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return " ".join([w for w in text.split() if w not in stop_words])

# ------------------------------------------------------------
# ⚙️ Load model family
# ------------------------------------------------------------
@st.cache_resource
def load_models(family):
    if family == "Tweets Models":
        vectorizer = joblib.load("artifacts/tweets_vectorizer.joblib")
        lr_model = joblib.load("artifacts/tweets_LR.joblib")
        rf_model = joblib.load("artifacts/tweets_RF.joblib")
    else:  # Fast Models
        vectorizer = joblib.load("artifacts/vectorizer_fast.joblib")
        lr_model = joblib.load("artifacts/LogisticRegression_fast.joblib")
        rf_model = joblib.load("artifacts/RandomForest_fast.joblib")

    return vectorizer, lr_model, rf_model

# ------------------------------------------------------------
# 🎨 Streamlit UI
# ------------------------------------------------------------
st.set_page_config(page_title="Cyberbullying Detector", layout="centered")

st.title("Cyberbullying Detection System")
st.write("Detect cyberbullying type using two families of ML models trained on different datasets.")

# Model family selection
model_family = st.selectbox(
    "Choose Model Family:",
    ("Tweets Models", "Fast Models")
)

# Model selection within family
model_choice = st.radio(
    "Select Model:",
    ("Logistic Regression", "Random Forest"),
    horizontal=True
)

# Load models dynamically
vectorizer, lr_model, rf_model = load_models(model_family)

# Input text
text_input = st.text_area(" Enter a tweet or post:", height=150)

# Predict button
if st.button("🔍 Predict"):
    if not text_input.strip():
        st.warning("Please enter some text first!")
    else:
        clean = clean_text(text_input)
        vec = vectorizer.transform([clean])

        if model_choice == "Logistic Regression":
            pred = lr_model.predict(vec)[0]
        else:
            pred = rf_model.predict(vec)[0]

        st.success(f"🔹 Predicted Cyberbullying Type: **{pred}**")

# Footer
st.markdown("---")
st.caption("Developed using Streamlit and Scikit-learn.")
