import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model
from utils.preprocessing import load_and_preprocess_data
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

# ✅ Custom Attention Layer (same as in train_model2.py)
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

# ✅ Load preprocessed data
X, y, scaler = load_and_preprocess_data('data/MSFT_stock_data.csv')
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# ✅ Load trained models
model1 = load_model('models/cnn_lstm_model.h5')
model2 = load_model('models/bilstm_attention_model.h5', custom_objects={'Attention': Attention})

# ✅ Generate predictions
pred1 = model1.predict(X_test)
pred2 = model2.predict(X_test)

# ✅ Inverse scale the predictions & actuals
scaled_full = np.zeros((len(y_test), 5))  # shape must match scaler input
scaled_full[:, 3] = y_test
y_actual = scaler.inverse_transform(scaled_full)[:, 3]

scaled_pred1 = np.zeros((len(pred1), 5))
scaled_pred1[:, 3] = pred1.flatten()
y_pred1 = scaler.inverse_transform(scaled_pred1)[:, 3]

scaled_pred2 = np.zeros((len(pred2), 5))
scaled_pred2[:, 3] = pred2.flatten()
y_pred2 = scaler.inverse_transform(scaled_pred2)[:, 3]

# ✅ Metrics Function
def print_metrics(y_true, y_pred, model_name):
    print(f"\n📊 {model_name} Metrics:")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
    print(f"MAE:  {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"R²:   {r2_score(y_true, y_pred):.4f}")

# ✅ Display metrics
print_metrics(y_actual, y_pred1, "CNN + LSTM")
print_metrics(y_actual, y_pred2, "BiLSTM + Attention")

# ✅ Plot actual vs predicted
plt.figure(figsize=(12, 5))
plt.plot(y_actual, label='Actual Price', color='black')
plt.plot(y_pred1, label='CNN+LSTM', linestyle='--')
plt.plot(y_pred2, label='BiLSTM+Attention', linestyle=':')
plt.title("Microsoft Stock Price Prediction")
plt.xlabel("Days")
plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()
plt.savefig("results/prediction_comparison.png")
plt.show()
