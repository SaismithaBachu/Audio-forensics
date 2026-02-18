import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from training.audio_dataset import MultiTaskAudioDataset
from models.multitask_crnn import MultiTaskCRNN

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_ROOT = r"D:\MajorProject\data\raw\TAU_2020"
CHECKPOINT_PATH = "artifacts/checkpoint_epoch7.pth"
# ----------------------------------------

def main():

    # Load validation dataset
    val_dataset = MultiTaskAudioDataset(
        csv_path="../splits/val.csv",
        base_dir=DATASET_ROOT
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False
    )

    num_scenes = len(val_dataset.scene_to_idx)
    num_devices = len(val_dataset.device_to_idx)

    model = MultiTaskCRNN(num_scenes, num_devices).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    all_scene_preds = []
    all_scene_labels = []

    all_device_preds = []
    all_device_labels = []

    with torch.no_grad():
        for x, scene, device in val_loader:
            x = x.to(DEVICE)

            scene_logits, device_logits = model(x)

            pred_scene = torch.argmax(scene_logits, dim=1).cpu().numpy()
            pred_device = torch.argmax(device_logits, dim=1).cpu().numpy()

            all_scene_preds.extend(pred_scene)
            all_scene_labels.extend(scene.numpy())

            all_device_preds.extend(pred_device)
            all_device_labels.extend(device.numpy())

    # -------- Confusion Matrices --------
    scene_cm = confusion_matrix(all_scene_labels, all_scene_preds)
    device_cm = confusion_matrix(all_device_labels, all_device_preds)

    plot_confusion_matrix(
        scene_cm,
        classes=list(val_dataset.scene_to_idx.keys()),
        title="Scene Confusion Matrix"
    )

    plot_confusion_matrix(
        device_cm,
        classes=list(val_dataset.device_to_idx.keys()),
        title="Device Confusion Matrix"
    )


def plot_confusion_matrix(cm, classes, title):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
