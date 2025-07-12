import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

# === CONFIG ===
OUT_DIR = "visualise/comparison"
os.makedirs(OUT_DIR, exist_ok=True)

# === MODEL METRICS ===
metrics = {
    "CNN + LSTM": {"RMSE": 9.7394, "MAE": 6.6931, "R2": 0.9929, "Accuracy": 0.85},
    "BiLSTM + Attention": {"RMSE": 15.8657, "MAE": 10.7196, "R2": 0.9812, "Accuracy": 0.75},
}

# Extract metric names
metric_names = list(next(iter(metrics.values())).keys())

# === RADAR CHART ===
labels = metric_names
num_vars = len(labels)

angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
angles += angles[:1]

plt.figure(figsize=(6, 6))
ax = plt.subplot(111, polar=True)

for model, vals in metrics.items():
    values = list(vals.values())
    values += values[:1]
    ax.plot(angles, values, linewidth=2, label=model)
    ax.fill(angles, values, alpha=0.2)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.set_title("Radar Chart: Model Performance")
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/radar_chart.png")
plt.close()

# === BOX PLOT OF ALL METRICS ===
df_metrics = []
for model, vals in metrics.items():
    for metric, val in vals.items():
        df_metrics.append((model, metric, val))

import pandas as pd
df = pd.DataFrame(df_metrics, columns=["Model", "Metric", "Value"])
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Metric", y="Value", hue="Model")
plt.title("Combined Boxplot of All Metrics")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/combined_boxplot.png")
plt.close()

# === HEATMAPS OF MODEL METRICS ===
for model, vals in metrics.items():
    df_heat = pd.DataFrame(vals, index=[0])
    plt.figure(figsize=(4, 3))
    sns.heatmap(df_heat, annot=True, cmap="YlGnBu")
    plt.title(f"Metric Heatmap - {model}")
    plt.tight_layout()
    model_slug = model.lower().replace(" ", "_").replace("+", "plus")
    plt.savefig(f"{OUT_DIR}/heatmap_{model_slug}.png")
    plt.close()

# === BAR PLOT FOR RMSE, MAE, R2 ===
metric_plot = ["RMSE", "MAE", "R2"]
plt.figure(figsize=(6, 4))
for m in metric_plot:
    sns.barplot(x=list(metrics.keys()), y=[metrics[model][m] for model in metrics], label=m)
plt.title("Model Performance Bar Plot (RMSE, MAE, R2)")
plt.xlabel("Model")
plt.ylabel("Score")
plt.legend(metric_plot)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/performance_barplot.png")
plt.close()

# === ACCURACY BAR PLOT ===
plt.figure(figsize=(6, 4))
sns.barplot(x=list(metrics.keys()), y=[metrics[model]["Accuracy"] for model in metrics])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/accuracy_bar.png")
plt.close()

print("✅ All comparison visuals saved to visualise/comparison/")
