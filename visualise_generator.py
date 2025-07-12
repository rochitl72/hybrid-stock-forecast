import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler, label_binarize
from tensorflow.keras.models import load_model
from utils.preprocessing import load_and_preprocess_data

# 🔧 Constants
MODEL_PATH = 'models/cnn_lstm_model.h5'
OUT_DIR = 'visualise/cnn_lstm'
os.makedirs(OUT_DIR, exist_ok=True)

# 📦 Load data
X, y, scaler = load_and_preprocess_data('data/MSFT_stock_data.csv')
split_index = int(0.8 * len(X))
X_test = X[split_index:]
y_test = y[split_index:]

# Load model and predict
model = load_model(MODEL_PATH)
y_pred = model.predict(X_test).flatten()

# ✅ Inverse transform
zeros_template = np.zeros((len(y_test), 5))
zeros_template[:, 3] = y_test
y_actual = scaler.inverse_transform(zeros_template)[:, 3]

zeros_pred = np.zeros((len(y_pred), 5))
zeros_pred[:, 3] = y_pred
y_pred_actual = scaler.inverse_transform(zeros_pred)[:, 3]

# ===============================
# 1. 🔥 HEATMAP of Prediction Error
# ===============================
errors = y_actual - y_pred_actual
df_heat = pd.DataFrame({'Actual': y_actual, 'Predicted': y_pred_actual, 'Error': errors})
plt.figure(figsize=(6, 4))
sns.heatmap(df_heat.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap (CNN + LSTM)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/heatmap.png")
plt.close()

# ===============================
# 2. 📈 ROC CURVE (via binning)
# ===============================
# Binning continuous values into 3 classes
threshold = 10
diff = y_pred_actual - y_actual
labels = np.select([diff < -threshold, abs(diff) <= threshold, diff > threshold], [0, 1, 2])

# Binarize for ROC (multi-class)
y_true_bin = label_binarize(labels, classes=[0,1,2])
y_scores_bin = label_binarize(np.round(y_pred_actual - y_actual + 1), classes=[0,1,2])

fpr = dict(); tpr = dict(); roc_auc = dict()
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC
plt.figure(figsize=(6,4))
for i in range(3):
    plt.plot(fpr[i], tpr[i], label=f"Class {i} AUC={roc_auc[i]:.2f}")
plt.plot([0,1],[0,1],'--',color='gray')
plt.title("ROC Curve (CNN + LSTM)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/roc_curve.png")
plt.close()

# ===============================
# 3. 🟣 SCATTER PLOT
# ===============================
plt.figure(figsize=(6,4))
plt.scatter(y_actual, y_pred_actual, alpha=0.5, s=10)
plt.title("Scatter Plot: Actual vs Predicted")
plt.xlabel("Actual Close Price")
plt.ylabel("Predicted Close Price")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/scatter_plot.png")
plt.close()

# ===============================
# 4. 🔲 CONFUSION MATRIX
# ===============================
pred_labels = np.select([diff < -threshold, abs(diff) <= threshold, diff > threshold], [0, 1, 2])
cm = confusion_matrix(labels, pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Under", "Correct", "Over"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix (CNN + LSTM)")
plt.savefig(f"{OUT_DIR}/confusion_matrix.png")
plt.close()

# ===============================
# 5. 📦 BOX PLOT of Errors
# ===============================
plt.figure(figsize=(6,4))
sns.boxplot(y=errors)
plt.title("Boxplot of Prediction Errors (CNN + LSTM)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/box_plot.png")
plt.close()

print("✅ Visuals saved to visualise/cnn_lstm/")
