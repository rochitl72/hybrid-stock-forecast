import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import load_model
from utils.preprocessing import load_and_preprocess_data
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

# === Custom Attention Layer ===
class Attention(Layer):
    def __init__(self):
        super(Attention, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

# === CONFIG ===
MODEL_PATH = 'models/bilstm_attention_model.h5'
OUT_DIR = 'visualise/bilstm_attention'
os.makedirs(OUT_DIR, exist_ok=True)

# === LOAD DATA ===
X, y, scaler = load_and_preprocess_data('data/MSFT_stock_data.csv')
split_index = int(0.8 * len(X))
X_test = X[split_index:]
y_test = y[split_index:]

# === LOAD MODEL & PREDICT ===
model = load_model(MODEL_PATH, custom_objects={'Attention': Attention})
y_pred = model.predict(X_test).flatten()

# === INVERSE TRANSFORM ===
zeros_template = np.zeros((len(y_test), 5))
zeros_template[:, 3] = y_test
y_actual = scaler.inverse_transform(zeros_template)[:, 3]

zeros_pred = np.zeros((len(y_pred), 5))
zeros_pred[:, 3] = y_pred
y_pred_actual = scaler.inverse_transform(zeros_pred)[:, 3]

# === HEATMAP ===
errors = y_actual - y_pred_actual
df_heat = pd.DataFrame({'Actual': y_actual, 'Predicted': y_pred_actual, 'Error': errors})
plt.figure(figsize=(6, 4))
sns.heatmap(df_heat.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap (BiLSTM + Attention)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/heatmap.png")
plt.close()

# === ROC CURVE (simulated) ===
threshold = 10  # price difference tolerance
diff = y_pred_actual - y_actual
labels = np.select([diff < -threshold, abs(diff) <= threshold, diff > threshold], [0, 1, 2])

# Binarize true and predicted
y_true_bin = label_binarize(labels, classes=[0, 1, 2])
y_pred_bin = label_binarize(np.select([diff < -threshold, abs(diff) <= threshold, diff > threshold], [0, 1, 2]), classes=[0, 1, 2])

fpr, tpr, roc_auc = {}, {}, {}
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(6, 4))
for i in range(3):
    plt.plot(fpr[i], tpr[i], label=f"Class {i} AUC={roc_auc[i]:.2f}")
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.title("ROC Curve (BiLSTM + Attention)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/roc_curve.png")
plt.close()

# === SCATTER PLOT ===
plt.figure(figsize=(6, 4))
plt.scatter(y_actual, y_pred_actual, alpha=0.5, s=10, c='green')
plt.title("Scatter Plot: Actual vs Predicted")
plt.xlabel("Actual Close Price")
plt.ylabel("Predicted Close Price")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/scatter_plot.png")
plt.close()

# === CONFUSION MATRIX ===
pred_labels = np.select([diff < -threshold, abs(diff) <= threshold, diff > threshold], [0, 1, 2])
cm = confusion_matrix(labels, pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Under", "Correct", "Over"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix (BiLSTM + Attention)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix.png")
plt.close()

# === BOX PLOT ===
plt.figure(figsize=(6, 4))
sns.boxplot(y=errors, color='teal')
plt.title("Boxplot of Prediction Errors (BiLSTM + Attention)")
plt.ylabel("Error")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/box_plot.png")
plt.close()

print("✅ Visuals saved to visualise/bilstm_attention/")
