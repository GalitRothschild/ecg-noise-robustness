"""
Shallow 1D CNN for ECG arrhythmia classification.
Based on Rasti et al. (2024), reproduced for Stage 1.
"""

import torch
import torch.nn as nn


class ShallowCNN1D(nn.Module):
    def __init__(self, num_classes=5, input_length=216):
        super(ShallowCNN1D, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=11, padding=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Block 2
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Block 3
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # ── Auto-calculate the flattened size ──────────────────────────────
        # Run a dummy tensor through features to get the real output size
        # This way it never mismatches, regardless of input_length
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_length)
            flat_size = self.features(dummy).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = ShallowCNN1D(num_classes=5, input_length=216)
    dummy = torch.randn(8, 1, 216)
    out = model(dummy)
    print("Output shape:", out.shape)  # should be [8, 5]
    print("Model ready!")