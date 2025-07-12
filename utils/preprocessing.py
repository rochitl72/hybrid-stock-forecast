import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(file_path, sequence_length=60):
    df = pd.read_csv(file_path)
    
    # Sort by date
    df = df.sort_values('Date')

    # Drop NA if any
    df.dropna(inplace=True)

    # Reorder columns if needed (Open, High, Low, Close, Volume)
    data = df[['Open', 'High', 'Low', 'Close', 'Volume']].values

    # Normalize
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, 3])  # 3 = Close

    X = np.array(X)
    y = np.array(y)
    return X, y, scaler
