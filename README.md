# ResNeXt-50 Animal Classifier

Classifying 50 animal species using ResNeXt-50 and transfer learning.

## Results

**Test Accuracy: 79.11%** | **Top-5: 95.11%**

| Phase | What Trained  | Epochs | Accuracy |
|-------|-------------  |--------|----------|
| 1     | Layer4 + FC   | 15     | 53.69%   |
| 2     | Layer3-4 + FC | 25     | 72.46%   |
| 3     | Layer2-4 + FC | 25     | 76.60%   |
| 4     | All Layers    | 30     | 79.11%   |
 

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Train
python train.py --phase all

# Test
python train.py --eval_only --resume models/checkpoints/phase4_best_model.pth

# Predict
python predict.py --image path/to/animal.jpg --checkpoint models/checkpoints/phase4_best_model.pth

---

## Dataset Setup

Put images in `data/raw/` organized by class:
```
data/raw/
├── antelope/
├── bear/
└── ... (50 folders)
```

---

## What It Does

- ResNeXt-50 pretrained on ImageNet
- 4-phase progressive transfer learning
- Layer-wise learning rates (1e-6 to 3e-4)
- AdamW optimizer + Cosine annealing scheduler
- Mixed precision training for Tesla T4

---

## Training

```bash
# All phases
python train.py --phase all

# Individual phases
python train.py --phase 1
python train.py --phase 2
python train.py --phase 3
python train.py --phase 4

# Fresh start
python train.py --phase 1 --fresh-start
```

Auto-resumes from checkpoint if interrupted.

---

## Config

Edit `config/config.py` to change:
- Batch size
- Learning rates
- Number of epochs
- Augmentation


## Why 79%?

Aimed for 90-92% but hit 79% because:
- 50 classes (some animals look similar)
- Class imbalance
- Hardware limits on Colab

To get 90%+: more data, ensemble models, or Vision Transformer.

---
## References

- [ResNeXt Paper](https://arxiv.org/abs/1611.05431)
- [AwA2 Dataset](https://cvml.ist.ac.at/AwA2/)
---
