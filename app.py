# ============================================================
# 🌐 STREAMLIT APP FOR CYBERBULLYING DETECTION (2 MODELS)
# ============================================================
import streamlit as st
import joblib
import re, string
import nltk
from nltk.corpus import stopwords

# ------------------------------------------------------------
# 🧹 Text cleaning setup
# ------------------------------------------------------------
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\w+|#", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return " ".join([w for w in text.split() if w not in stop_words])

# ------------------------------------------------------------
# 💾 Load saved models & vectorizer
# ------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    vectorizer = joblib.load("artifacts/vectorizer_fast.joblib")
    log_model = joblib.load("artifacts/LogisticRegression_fast.joblib")
    rf_model = joblib.load("artifacts/RandomForest_fast.joblib")
    return vectorizer, log_model, rf_model

vectorizer, log_model, rf_model = load_artifacts()

# ------------------------------------------------------------
# 🎨 Streamlit UI
# ------------------------------------------------------------
st.set_page_config(page_title="Cyberbullying Detector", page_icon="🧠", layout="centered")

st.title("🧠 Cyberbullying Detection App")
st.write("Enter a social media comment or tweet below to predict whether it contains cyberbullying content.")

# Model selection
model_choice = st.selectbox(
    "Select Model:",
    ("Logistic Regression", "Random Forest")
)

# Input text box
user_input = st.text_area("✏️ Enter text here:", height=150, placeholder="Type or paste a comment...")

# Prediction
if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text before predicting.")
    else:
        # Clean and vectorize
        cleaned = clean_text(user_input)
        X_input = vectorizer.transform([cleaned])

        # Choose model
        if model_choice == "Logistic Regression":
            pred = log_model.predict(X_input)[0]
        else:
            pred = rf_model.predict(X_input)[0]

        # Display result
        if pred.lower() == "not_cyberbullying":
            st.success("✅ This comment is **NOT cyberbullying**.")
        else:
            st.error(f"⚠️ This comment is classified as **{pred.upper()}**.")
