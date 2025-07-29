# model_core.py

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import torch.optim as optim

# RamanNet definition
class RamanNet(nn.Module):
    def __init__(self, input_length=1000, window_size=50, step=25, n1=32, n2=256, embedding_dim=128, num_classes=2):
        super(RamanNet, self).__init__()
        self.window_size = window_size
        self.step = step
        self.num_windows = (input_length - window_size) // step + 1

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(window_size, n1),
                nn.BatchNorm1d(n1),
                nn.LeakyReLU()
            ) for _ in range(self.num_windows)
        ])

        self.dropout1 = nn.Dropout(0.4)
        self.summary_dense = nn.Sequential(
            nn.Linear(n1 * self.num_windows, n2),
            nn.BatchNorm1d(n2),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )

        self.embedding = nn.Sequential(
            nn.Linear(n2, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        windows = []
        for i in range(self.num_windows):
            start = i * self.step
            end = start + self.window_size
            window = x[:, start:end]
            windows.append(self.blocks[i](window))
        x = torch.cat(windows, dim=1)
        x = self.dropout1(x)
        x = self.summary_dense(x)
        emb = nn.functional.normalize(self.embedding(x), p=2, dim=1)
        out = self.classifier(emb)
        return out, emb

def preprocess_spectrum(df, target_len=1000):
    # Safely convert all columns to float
    df = df.apply(lambda col: pd.to_numeric(col, errors='coerce')).dropna()

    # Extract x and y
    x = df.iloc[:, 0].values
    y = df.iloc[:, 1].values

    # Interpolate and normalize
    x_uniform = np.linspace(x.min(), x.max(), target_len)
    y_interp = interp1d(x, y, kind='linear', fill_value="extrapolate")(x_uniform)
    y_norm = StandardScaler().fit_transform(y_interp.reshape(-1, 1)).flatten()

    return y_norm

# Training function with live progress
def train_model(X_np, y_np, epochs=100, progress_callback=None):
    X_train_tensor = torch.tensor(X_np, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_np, dtype=torch.long)

    model = RamanNet(input_length=1000)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs, _ = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if progress_callback and (epoch + 1) % 10 == 0:
            with torch.no_grad():
                preds = torch.argmax(outputs, dim=1)
                acc = (preds == y_train_tensor).float().mean().item()
                progress_callback(epoch + 1, acc)

    return model

def preprocess_spectrum(df, target_len=1000):
    import pandas as pd
    df = df.apply(lambda col: pd.to_numeric(col, errors='coerce')).dropna()

    x = df.iloc[:, 0].values
    y = df.iloc[:, 1].values

    x_uniform = np.linspace(x.min(), x.max(), target_len)
    y_interp = interp1d(x, y, kind='linear', fill_value="extrapolate")(x_uniform)
    y_norm = StandardScaler().fit_transform(y_interp.reshape(-1, 1)).flatten()

    return y_norm
def predict_spectrum(model, df):
    spectrum = preprocess_spectrum(df)
    input_tensor = torch.tensor(spectrum.reshape(1, -1), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        outputs, _ = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
    return pred, confidence
