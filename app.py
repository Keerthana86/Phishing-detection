import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# --- Configuration ---
MODEL_PATH = 'model.h5'
TOKENIZER_PATH = 'tokenizer.pkl'
MAX_LEN = 100

# --- Page settings ---
st.set_page_config(page_title="Phishing Detection System", layout="wide")
st.title("📁 Phishing Detection From Uploaded Dataset")
st.markdown("Upload your dataset")

# --- Load model ---
if not os.path.exists(MODEL_PATH):
    st.error("❌ model.h5 not found.")
    st.stop()

model = load_model(MODEL_PATH)

# --- Load tokenizer ---
if not os.path.exists(TOKENIZER_PATH):
    st.error("❌ tokenizer.pkl not found.")
    st.stop()

with open(TOKENIZER_PATH, 'rb') as handle:
    tokenizer = pickle.load(handle)

# --- File upload ---
st.subheader("📤 Upload Dataset File")
uploaded_file = st.file_uploader("Upload a .txt file (Format: label[TAB]text)", type="txt")

if uploaded_file is not None:
    try:
        # Read and parse lines
        lines = uploaded_file.read().decode("utf-8").splitlines()
        data = [line.strip() for line in lines[1:] if "\t" in line]
        texts = [line.split("\t")[1] for line in data]
        labels = [line.split("\t")[0] for line in data]

        # Tokenize and pad
        sequences = tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
        preds = model.predict(padded)
        preds_label = (preds > 0.5).astype(int).flatten()

        # Label conversion
        label_map = {'phishing': 1, 'legitimate': 0}
        y_true = np.array([label_map.get(lbl.lower(), 0) for lbl in labels])

        # Calculate accuracy
        accuracy = np.mean(preds_label == y_true)
        st.success(f"✅ Model Accuracy on Uploaded Data: **{accuracy:.4f}**")

        # Display prediction results
        results_df = pd.DataFrame({
            'Text': texts,
            'Actual Label': ['Phishing' if val == 1 else 'Legitimate' for val in y_true],
            'Predicted Label': ['Phishing' if val == 1 else 'Legitimate' for val in preds_label],
            'Confidence': [round(p[0], 2) for p in preds]
        })

        st.write("### 📊 Prediction Results")
        st.dataframe(results_df, use_container_width=True)

        # Confusion Matrix
        cm = confusion_matrix(y_true, preds_label)
        st.write("### 🔁 Confusion Matrix")
        cm_fig = px.imshow(
            cm,
            x=["Legitimate", "Phishing"],
            y=["Legitimate", "Phishing"],
            labels=dict(x="Predicted", y="Actual", color="Count"),
            text_auto=True,
            color_continuous_scale="Blues"
        )
        st.plotly_chart(cm_fig)

        # Classification Report
        st.write("### 📄 Classification Report")
        report_dict = classification_report(y_true, preds_label, target_names=["Legitimate", "Phishing"], output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(report_df)

        # ROC Curve
        st.write("### 📈 ROC Curve")
        fpr, tpr, _ = roc_curve(y_true, preds)
        roc_auc = auc(fpr, tpr)
        roc_fig = go.Figure()
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
        roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
        roc_fig.update_layout(title=f"ROC Curve (AUC = {roc_auc:.2f})",
                              xaxis_title='False Positive Rate',
                              yaxis_title='True Positive Rate')
        st.plotly_chart(roc_fig)

        # Confidence Distribution
        st.write("### 🔬 Confidence Score Distribution")
        confidence_scores = [p[0] for p in preds]
        confidence_fig = px.histogram(
            x=confidence_scores,
            nbins=20,
            labels={'x': 'Confidence Score'},
            title='Prediction Confidence Distribution'
        )
        confidence_fig.update_layout(yaxis_title='Count')
        st.plotly_chart(confidence_fig)

    except Exception as e:
        st.error(f"❌ Error while processing the file: {e}")
