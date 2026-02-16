import torch
import torch.nn as nn


class MultiTaskCRNN(nn.Module):
    def __init__(self, num_scenes, num_devices):
        super().__init__()

        # -------- CNN Backbone --------
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        # After pooling:
        # Input: [B, 1, 128, 320]
        # → [B, 32, 32, 80]

        self.rnn = nn.LSTM(
            input_size=32 * 32,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # -------- Multi-task Heads --------
        self.scene_head = nn.Linear(128 * 2, num_scenes)
        self.device_head = nn.Linear(128 * 2, num_devices)

    def forward(self, x):
        # x: [B, 1, 128, 320]
        x = self.cnn(x)               # [B, 32, 32, 80]

        B, C, H, T = x.shape
        x = x.permute(0, 3, 1, 2)     # [B, T, C, H]
        x = x.contiguous().view(B, T, C * H)  # [B, T, 1024]

        _, (hn, _) = self.rnn(x)

        # Concatenate last forward & backward hidden states
        features = torch.cat((hn[-2], hn[-1]), dim=1)

        scene_out = self.scene_head(features)
        device_out = self.device_head(features)

        return scene_out, device_out
