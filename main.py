import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from xgboost import XGBClassifier

# --- Paths ---
train_path = 'C:/Users/CRESCENT/Desktop/Phishing Detection/input/DL Dataset/train.txt'
test_path = 'C:/Users/CRESCENT/Desktop/Phishing Detection/input/DL Dataset/test.txt'

# --- Load dataset ---
def load_data(path):
    data = [line.strip() for line in open(path, "r").readlines()[1:]]
    texts = [line.split("\t")[1] for line in data]
    labels = [line.split("\t")[0] for line in data]
    return texts, labels

x_train_raw, y_train_raw = load_data(train_path)
x_test_raw, y_test_raw = load_data(test_path)

# --- Encode labels ---
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_raw)
y_test = label_encoder.transform(y_test_raw)

# --- Tokenize texts ---
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(x_train_raw)
x_train_seq = tokenizer.texts_to_sequences(x_train_raw)
x_test_seq = tokenizer.texts_to_sequences(x_test_raw)

# --- Pad sequences ---
MAX_LEN = 100
x_train_pad = pad_sequences(x_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
x_test_pad = pad_sequences(x_test_seq, maxlen=MAX_LEN, padding='post', truncating='post')

# --- Save tokenizer ---
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# ================================
# CNN MODEL
# ================================
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=MAX_LEN),
    Conv1D(filters=64, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.5),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

model.fit(
    x_train_pad, y_train,
    epochs=12,
    batch_size=64,
    validation_data=(x_test_pad, y_test),
    callbacks=[early_stop],
    verbose=1
)

# --- CNN Evaluation ---
y_pred_cnn = (model.predict(x_test_pad) > 0.5).astype(int)
cnn_accuracy = accuracy_score(y_test, y_pred_cnn)
print(f"\nCNN Model Accuracy: {cnn_accuracy:.4f}")
print("CNN Classification Report:")
print(classification_report(y_test, y_pred_cnn))

# --- Save CNN Model ---
model.save('model.h5')
print("CNN model saved as model.h5")

# ================================
# XGBOOST MODEL
# ================================
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100, max_depth=5)
xgb_model.fit(x_train_pad, y_train)

# --- XGBoost Evaluation ---
y_pred_xgb = xgb_model.predict(x_test_pad)
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
print(f"\nXGBoost Model Accuracy: {xgb_accuracy:.4f}")
print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))

# --- Save XGBoost Model ---
with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
print("XGBoost model saved as xgb_model.pkl")
