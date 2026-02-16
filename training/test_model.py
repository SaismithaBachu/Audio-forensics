import sys
from pathlib import Path

# Add parent directory to path so we can import models
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from models.multitask_crnn import MultiTaskCRNN

num_scenes = 10   # adjust if needed
num_devices = 9   # adjust to your dataset

model = MultiTaskCRNN(num_scenes, num_devices)

x = torch.randn(8, 1, 128, 320)

scene_logits, device_logits = model(x)

print("Scene output shape:", scene_logits.shape)
print("Device output shape:", device_logits.shape)
