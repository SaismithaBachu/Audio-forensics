from audio_dataset import MultiTaskAudioDataset
from torch.utils.data import DataLoader

dataset = MultiTaskAudioDataset(
    csv_path="splits/train.csv",
    base_dir="datasets/dcase_tau/TAU-urban-acoustic-scenes-2020-mobile-development"
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0  # keep 0 on Windows
)

x, scene, device = next(iter(loader))

print("Batch spectrogram shape:", x.shape)
print("Scene labels:", scene)
print("Device labels:", device)
