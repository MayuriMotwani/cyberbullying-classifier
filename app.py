import streamlit as st
import pandas as pd, joblib

st.set_page_config(page_title="Cyberbullying Classifier", layout="centered")
st.title("🧠 Cyberbullying Type Classifier")

# Load models
vectorizer = joblib.load("artifacts/vectorizer.joblib")
lr_model = joblib.load("artifacts/LogisticRegression.joblib")
rf_model = joblib.load("artifacts/RandomForest.joblib")

models = {"Logistic Regression": lr_model, "Random Forest": rf_model}
model_name = st.sidebar.selectbox("Select Model", list(models.keys()))
model = models[model_name]

st.subheader("🔹 Single Tweet Prediction")
text = st.text_area("Enter tweet text:")
if st.button("Predict"):
    if text.strip():
        X = vectorizer.transform([text])
        pred = model.predict(X)[0]
        st.success(f"Prediction: {pred}")
    else:
        st.warning("Please enter text!")

st.markdown("---")
st.subheader("🔹 Batch CSV Prediction")
file = st.file_uploader("Upload CSV (must have 'tweet_text' column)", type="csv")
if file:
    df = pd.read_csv(file)
    if "tweet_text" in df.columns:
        X = vectorizer.transform(df["tweet_text"].fillna(""))
        df["predicted_cyberbullying_type"] = model.predict(X)
        st.dataframe(df.head())
        st.download_button("Download Predictions", df.to_csv(index=False), "predictions.csv")
    else:
        st.error("No column named 'tweet_text' found.")
