# 📈 Hybrid Stock Forecasting with BiLSTM + Attention & CNN-LSTM

This project explores advanced deep learning architectures to forecast Microsoft (MSFT) stock prices using time series data. We implement and compare two powerful models:

- 🔁 **BiLSTM with Attention**
- 🧠 **CNN + LSTM Hybrid**

---

## 🔍 Objective

To predict future closing prices of MSFT stock and classify the prediction error into 3 classes (under-predicted, accurate, over-predicted) using an error-threshold-based strategy.

---

## 💡 Model Architectures

### 1. BiLSTM + Attention  
- Bidirectional LSTM for capturing forward and backward dependencies  
- Custom Attention layer for focusing on important time steps  

### 2. CNN + LSTM  
- 1D Convolution to extract local temporal features  
- LSTM to capture long-term dependencies  

---

## 📊 Results

| Metric       | BiLSTM + Attention | CNN + LSTM |
|--------------|--------------------|------------|
| Accuracy     | 1.0000             | 1.0000     |
| Precision    | 1.0000             | 1.0000     |
| Recall       | 1.0000             | 1.0000     |
| F1-Score     | 1.0000             | 1.0000     |
| MAE          | 10.7196            | 6.6931     |
| MSE          | 251.72             | 94.86      |
| RMSE         | 15.8657            | 9.7394     |
| R² Score     | 0.9812             | 0.9929     |

> 📌 *Despite equal classification performance, CNN+LSTM had a significantly lower MAE/RMSE.*

---

## ⚙️ How to Run

### Step 1: Install Requirements
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
Step 2: Train Models
bash
Copy
Edit
python train_model2.py   # BiLSTM + Attention
python train_model1.py   # CNN + LSTM
Step 3: Evaluate Models
bash
Copy
Edit
python evaluate_bilstm.py
python evaluate_cnnlstm.py
📁 Folder Structure
kotlin
Copy
Edit
hybrid-stock-forecast/
│
├── data/
│   └── MSFT_stock_data.csv
│
├── models/
│   └── bilstm_attention_model.h5
│   └── cnn_lstm_model.h5
│
├── utils/
│   └── preprocessing.py
│
├── visualise/
│   └── bilstm_attention/
│       └── metrics.txt
│   └── cnn_lstm/
│       └── metrics.txt
│
├── train_model1.py
├── train_model2.py
├── evaluate_cnnlstm.py
├── evaluate_bilstm.py
└── requirements.txt
