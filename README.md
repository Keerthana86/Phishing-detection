# 🔐 Phishing Detection using CNN and XGBoost

A machine learning-based web application that detects phishing URLs using a hybrid model combining **Convolutional Neural Networks (CNN)** and **Extreme Gradient Boosting (XGBoost)**. The app provides prediction results, evaluation metrics, and visualizations through an interactive **Streamlit** interface.

---

## 🚀 Features

- Detects phishing URLs with high accuracy.
- Dual-model architecture: CNN for sequence-based feature learning and XGBoost for boosting performance.
- User-friendly Streamlit web interface.
- Upload `.txt` files with URLs for batch evaluation.
- Displays prediction results, confusion matrix, and performance metrics.

---

## 🧠 Models Used

- **CNN**: Used to extract patterns from URL character sequences.
- **XGBoost**: Trained on handcrafted and statistical features for powerful ensemble learning.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **ML Libraries**: TensorFlow / Keras, XGBoost, Scikit-learn, Pandas, Numpy, Matplotlib

---

## 📂 Project Structure

Phishing-Detection/
├── app.py # Streamlit app
├── cnn_model.h5 # Trained CNN model
├── tokenizer.pickle # Tokenizer for URLs
├── xgboost_model.json # Trained XGBoost model
├── urls_dataset.txt # Sample dataset for testing
├── utils.py # Preprocessing and helper functions
├── requirements.txt # Python dependencies
└── README.md
