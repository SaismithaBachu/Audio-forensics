import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.audio_dataset import MultiTaskAudioDataset
from models.multitask_crnn import MultiTaskCRNN


# ---------------- CONFIG ----------------
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_ROOT = r"D:\MajorProject\data\raw\TAU_2020"

# ----------------------------------------


def main():
    # -------- Dataset & Loader --------
    train_dataset = MultiTaskAudioDataset(
        csv_path="../splits/train.csv",
        base_dir=DATASET_ROOT
    )

    val_dataset = MultiTaskAudioDataset(
        csv_path="../splits/val.csv",
        base_dir=DATASET_ROOT
    )

    print("\nTrain size:", len(train_dataset))
    print("Val size:", len(val_dataset))
    print("Scenes:", train_dataset.scene_to_idx)
    print("Devices:", train_dataset.device_to_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    num_scenes = len(train_dataset.scene_to_idx)
    num_devices = len(train_dataset.device_to_idx)

    # -------- Model --------
    model = MultiTaskCRNN(num_scenes, num_devices).to(DEVICE)

    # -------- Loss & Optimizer --------
    scene_criterion = nn.CrossEntropyLoss()
    device_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # -------- Training Loop --------
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for x, scene, device in train_loader:
            x = x.to(DEVICE)
            scene = scene.to(DEVICE)
            device = device.to(DEVICE)

            optimizer.zero_grad()

            scene_logits, device_logits = model(x)

            loss_scene = scene_criterion(scene_logits, scene)
            loss_device = device_criterion(device_logits, device)

            loss = loss_scene + loss_device
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        print(f"\nEpoch [{epoch+1}/{EPOCHS}] - Train Loss: {avg_loss:.4f}")

        # Run validation and get validation loss
        validate(model, val_loader, scene_criterion, device_criterion)

        # -------- Save checkpoint --------
        os.makedirs("artifacts", exist_ok=True)
        ckpt_path = os.path.join("artifacts", f"checkpoint_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)

        print(f"Saved checkpoint: {ckpt_path}")


def validate(model, loader, scene_criterion, device_criterion):
    model.eval()
    correct_scene = 0
    correct_device = 0
    total = 0

    with torch.no_grad():
        for x, scene, device in loader:
            x = x.to(DEVICE)
            scene = scene.to(DEVICE)
            device = device.to(DEVICE)

            scene_logits, device_logits = model(x)

            pred_scene = torch.argmax(scene_logits, dim=1)
            pred_device = torch.argmax(device_logits, dim=1)

            correct_scene += (pred_scene == scene).sum().item()
            correct_device += (pred_device == device).sum().item()
            total += scene.size(0)

    scene_acc = correct_scene / total * 100
    device_acc = correct_device / total * 100

    print(
        f"Validation | Scene Acc: {scene_acc:.2f}% | "
        f"Device Acc: {device_acc:.2f}%"
    )



if __name__ == "__main__":
    main()