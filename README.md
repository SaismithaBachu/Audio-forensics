# Audio Forensics — Multi-Task Scene & Device Classification

Classify audio into **acoustic scenes** (urban, bus, etc.) and **sound devices** (mobile, tablet, etc.) using a CRNN model trained on [DCASE 2020 TAU dataset](https://zenodo.org/record/3819968).

---

## Setup

### 1. Clone & Create Virtual Environment
```bash
git clone <your-repo-url>
cd audio_forensics
python -m venv .venv
```

### 2. Activate Virtual Environment
**Windows (PowerShell)**:
```powershell
.venv\Scripts\Activate.ps1
```

**Linux/Mac**:
```bash
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Dataset

The DCASE 2020 TAU dataset is **not included** in the repo. Download manually:

1. Download from [Zenodo](https://zenodo.org/record/3819968)
2. Extract to:
   ```
   datasets/dcase_tau/TAU-urban-acoustic-scenes-2020-mobile-development/
   ```

Expected structure:
```
datasets/
└── dcase_tau/
    └── TAU-urban-acoustic-scenes-2020-mobile-development/
        ├── meta.csv
        ├── audio/
        └── evaluation_setup/
```

---

## Usage

### Split Dataset (train/val/test)
```bash
cd training
python split_dataset.py
```
Creates: `splits/train.csv`, `splits/val.csv`, `splits/test.csv`

### Train Model
```bash
python train.py
```
Saves checkpoints to `artifacts/checkpoint_epoch*.pth` after each epoch.

### Load Trained Model
```python
import torch
from models.multitask_crnn import MultiTaskCRNN

# Load checkpoint
device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load("artifacts/checkpoint_epoch5.pth", map_location=device)

# Initialize model
model = MultiTaskCRNN(num_scenes=10, num_devices=9).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Inference
x = torch.randn(1, 1, 128, 320).to(device)
with torch.no_grad():
    scene_logits, device_logits = model(x)
    scene_pred = torch.argmax(scene_logits, dim=1)
    device_pred = torch.argmax(device_logits, dim=1)
```

### Export Model (Optional)

**TorchScript**:
```python
example = torch.randn(1, 1, 128, 320).to(device)
traced = torch.jit.trace(model.eval(), example)
traced.save("artifacts/model.pt")
```

**ONNX**:
```python
torch.onnx.export(
    model.eval(),
    example,
    "artifacts/model.onnx",
    input_names=["input"],
    output_names=["scene_out", "device_out"]
)
```

---

## Results

**Training completed (5 epochs):**
- Best Scene Accuracy: ~17%
- Best Device Accuracy: ~97%
- Final Train Loss: 0.9758

*Note: Low scene accuracy suggests the model may need more tuning (data augmentation, class weighting, or architecture changes).*

---

## Project Structure

```
audio_forensics/
├── models/
│   ├── __init__.py
│   └── multitask_crnn.py          # CRNN architecture
├── preprocessing/
│   └── __init__.py
├── training/
│   ├── __init__.py
│   ├── audio_dataset.py           # PyTorch Dataset class
│   ├── split_dataset.py           # Create train/val/test splits
│   ├── test_loader.py             # Mini test (optional)
│   ├── test_model.py              # Model sanity check (optional)
│   └── train.py                   # Main training script
├── artifacts/                     # Saved checkpoints (gitignored)
├── datasets/                      # Audio data (gitignored)
├── splits/                        # CSV split files
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Dependencies

- **torch**: Deep learning framework
- **pandas**: Data manipulation
- **scikit-learn**: Train/test splitting
- **librosa**: Audio processing
- **numpy**: Numerical computing

See `requirements.txt` for pinned versions.


