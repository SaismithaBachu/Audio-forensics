import pandas as pd
import torch
from torch.utils.data import Dataset
import librosa
import os
import numpy as np
import torch.nn.functional as F

MAX_FRAMES = 320


class MultiTaskAudioDataset(Dataset):
    def __init__(self, csv_path, base_dir, sr=44100):
        self.df = pd.read_csv(csv_path)

        # 🔹 Keep only real devices
        self.df = self.df[self.df["source_label"].isin(['a', 'b', 'c'])].reset_index(drop=True)

        self.base_dir = base_dir.rstrip("/") + "/"
        self.sr = sr

        # Build full paths
        self.df["full_path"] = self.df["filename"].apply(
            lambda x: os.path.join(self.base_dir, x)
        )

        # Keep only existing files (for partial dataset)
        self.df = self.df[self.df["full_path"].apply(os.path.exists)].reset_index(drop=True)

        print(f"Usable audio files: {len(self.df)}")

        self.scene_to_idx = {s: i for i, s in enumerate(sorted(self.df["scene_label"].unique()))}
        self.device_to_idx = {d: i for i, d in enumerate(sorted(self.df["source_label"].unique()))}

    def __len__(self):
        return len(self.df)

    def pad_or_crop(self, spec):
        """
        spec: Tensor [1, 128, T]
        """
        T = spec.shape[-1]

        if T < MAX_FRAMES:
            pad_amt = MAX_FRAMES - T
            spec = F.pad(spec, (0, pad_amt))
        else:
            spec = spec[:, :, :MAX_FRAMES]

        return spec

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row["full_path"]

        y, sr = librosa.load(file_path, sr=self.sr)

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=128,
            hop_length=512
        )
        mel_db = librosa.power_to_db(mel)

        mel_tensor = torch.tensor(mel_db).unsqueeze(0).float()
        mel_tensor = self.pad_or_crop(mel_tensor)

        scene_label = torch.tensor(self.scene_to_idx[row["scene_label"]])
        device_label = torch.tensor(self.device_to_idx[row["source_label"]])

        return mel_tensor, scene_label, device_label
