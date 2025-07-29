import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel,
    QFileDialog, QVBoxLayout, QWidget
)
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d

# --------------- RamanNet Model Definition ---------------
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

# --------------- Main App Class ---------------
class IL3DetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧪 IL-3 Detection from Raman Spectrum")
        self.setGeometry(100, 100, 600, 500)

        # Load model
        self.model = RamanNet(input_length=1000)
        self.model.load_state_dict(torch.load("ramannet_model.pt", map_location=torch.device("cpu")))
        self.model.eval()

        # UI layout
        layout = QVBoxLayout()

        self.upload_btn = QPushButton("📂 Upload Spectrum")
        self.upload_btn.setFont(QFont("Arial", 12))
        self.upload_btn.clicked.connect(self.load_file)

        self.predict_btn = QPushButton("🧠 Predict IL-3")
        self.predict_btn.setFont(QFont("Arial", 12))
        self.predict_btn.clicked.connect(self.predict)
        self.predict_btn.setEnabled(False)

        self.result_label = QLabel("")
        self.result_label.setFont(QFont("Arial", 14))

        # Plot canvas
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)

        layout.addWidget(self.upload_btn)
        layout.addWidget(self.predict_btn)
        layout.addWidget(self.result_label)
        layout.addWidget(self.canvas)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.spectrum = None  # placeholder for input

    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Spectrum File", "", "CSV or TXT Files (*.csv *.txt)")
        if path:
            df = pd.read_csv(path, header=None)
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            x = df.iloc[:, 0].values
            y = df.iloc[:, 1].values

            # Plot
            self.ax.clear()
            self.ax.plot(x, y)
            self.ax.set_title("Raman Spectrum")
            self.ax.set_xlabel("Wavenumber (cm⁻¹)")
            self.ax.set_ylabel("Intensity")
            self.canvas.draw()

            self.spectrum = (x, y)
            self.predict_btn.setEnabled(True)
            self.result_label.setText("")

    def preprocess(self, x, y):
        x_uniform = np.linspace(x.min(), x.max(), 1000)
        y_interp = interp1d(x, y, kind='linear', fill_value="extrapolate")(x_uniform)
        y_norm = StandardScaler().fit_transform(y_interp.reshape(-1, 1)).flatten()

        # Segment
        window_size, step = 50, 25
        segments = []
        for i in range(0, len(y_norm) - window_size + 1, step):
            segments.append(y_norm[i:i + window_size])
        input_tensor = torch.tensor(np.concatenate(segments)).float().unsqueeze(0)
        return input_tensor

    def predict(self):
        if self.spectrum is None:
            return
        x, y = self.spectrum
        input_tensor = self.preprocess(x, y)
        with torch.no_grad():
            output, _ = self.model(input_tensor)
            pred = torch.argmax(output, dim=1).item()
            probs = torch.softmax(output, dim=1)[0].numpy()
            confidence = probs[pred]

        if confidence < 0.65:
            self.result_label.setStyleSheet("color: orange;")
            msg = f"⚠️ Uncertain – Retest recommended (Confidence: {confidence:.2%})"
        elif pred == 1:
            self.result_label.setStyleSheet("color: green;")
            msg = f"✅ IL-3 Present (Confidence: {confidence:.2%})"
        else:
            self.result_label.setStyleSheet("color: red;")
            msg = f"❌ IL-3 Absent (Confidence: {confidence:.2%})"

        self.result_label.setText(msg)

# --------------- Run the App ---------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = IL3DetectorApp()
    window.show()
    sys.exit(app.exec_())
