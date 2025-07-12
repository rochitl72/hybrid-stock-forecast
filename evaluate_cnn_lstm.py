import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import load_model
from utils.preprocessing import load_and_preprocess_data

# === CONFIG ===
MODEL_PATH = 'models/cnn_lstm_model.h5'
OUT_DIR = 'visualise/cnn_lstm'
os.makedirs(OUT_DIR, exist_ok=True)

# === LOAD DATA ===
X, y, scaler = load_and_preprocess_data('data/MSFT_stock_data.csv')
split_index = int(0.8 * len(X))
X_test = X[split_index:]
y_test = y[split_index:]

# === LOAD MODEL ===
model = load_model(MODEL_PATH)
y_pred = model.predict(X_test).flatten()

# === INVERSE TRANSFORM ===
zeros_template = np.zeros((len(y_test), 5))
zeros_template[:, 3] = y_test
y_actual = scaler.inverse_transform(zeros_template)[:, 3]

zeros_pred = np.zeros((len(y_pred), 5))
zeros_pred[:, 3] = y_pred
y_pred_actual = scaler.inverse_transform(zeros_pred)[:, 3]

# === ERROR THRESHOLD BASED LABELS ===
threshold = 10
true_diff = np.diff(y_actual, prepend=y_actual[0])
pred_diff = np.diff(y_pred_actual, prepend=y_pred_actual[0])

y_true_class = np.select([true_diff < -threshold, abs(true_diff) <= threshold, true_diff > threshold], [0, 1, 2])
y_pred_class = np.select([pred_diff < -threshold, abs(pred_diff) <= threshold, pred_diff > threshold], [0, 1, 2])


# === METRICS ===
acc = accuracy_score(y_true_class, y_pred_class)
prec = precision_score(y_true_class, y_pred_class, average='weighted', zero_division=0)
recall = recall_score(y_true_class, y_pred_class, average='weighted', zero_division=0)
f1 = f1_score(y_true_class, y_pred_class, average='weighted', zero_division=0)

mae = mean_absolute_error(y_actual, y_pred_actual)
mse = mean_squared_error(y_actual, y_pred_actual)
rmse = np.sqrt(mse)
r2 = r2_score(y_actual, y_pred_actual)

# === SAVE METRICS TO TXT FILE ===
with open(f"{OUT_DIR}/metrics.txt", "w") as f:
    f.write("📈 CNN + LSTM Model Evaluation\n")
    f.write(f"Accuracy:      {acc:.4f}\n")
    f.write(f"Precision:     {prec:.4f}\n")
    f.write(f"Recall:        {recall:.4f}\n")
    f.write(f"F1-Score:      {f1:.4f}\n\n")
    f.write(f"MAE:           {mae:.4f}\n")
    f.write(f"MSE:           {mse:.4f}\n")
    f.write(f"RMSE:          {rmse:.4f}\n")
    f.write(f"R² Score:      {r2:.4f}\n")

print(f"✅ Metrics saved to {OUT_DIR}/metrics.txt")
