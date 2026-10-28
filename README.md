# FR-MobileTransformer

Official implementation of **"FR-MobileTransformer: an ultra-lightweight hybrid architecture for robust plant disease recognition in edge-constrained environments"** (submitted to the *Journal of Electronic Imaging*).

## Highlights

| Metric | Value |
|---|---|
| Top-1 accuracy (PlantVillage, 39 classes) | **98.45% ± 0.26%** (3 seeds) |
| Parameters | **0.447M** (with aux head, training-time) |
| MACs | **0.128G** (thop) |
| Checkpoint size | **1.8 MB** (FP32) |
| Inference | **2,880 images/s** (RTX 2080 Ti); **3.4 ms/image** (4-thread x86 CPU) |

## Architecture

Three complementary components, orthogonal by design:

1. **Reparameterized stem** — multi-branch training merged into a single 3×3 conv at inference (structural dimension).
2. **FGFM (Frequency-Gated Feature Modulation)** — frequency-domain channel gate modulating per-channel spectral responses (spectral dimension), ~parameter-free.
3. **CLAG (Cross-Level Attention Gate)** — deep semantics refine shallow spatial features with interpretable attention maps (spatial dimension).

## Requirements

```
Python 3.8+, PyTorch >= 1.13, torchvision, numpy, thop
```

## Dataset

Download PlantVillage from the official repository:

```
https://github.com/spMohanty/PlantVillage-Dataset
```

Prepare the 39-class color subset in `data_clean/` with an 85/15 train/validation split (the script splits automatically via `--val_ratio 0.15`).

## Training

Full model (any seed):

```
python exp_frmt_experiments.py --data_path data_clean --seed 42   --ablate full
python exp_frmt_experiments.py --data_path data_clean --seed 2026 --ablate full
python exp_frmt_experiments.py --data_path data_clean --seed 1024 --ablate full
```

Ablations:

```
python exp_frmt_experiments.py --data_path data_clean --seed 42 --ablate baseline   # backbone only
python exp_frmt_experiments.py --data_path data_clean --seed 42 --ablate reparam    # + reparameterized stem
python exp_frmt_experiments.py --data_path data_clean --seed 42 --ablate fgfm       # + FGFM
python exp_frmt_experiments.py --data_path data_clean --seed 42 --ablate clag       # + CLAG
python exp_frmt_experiments.py --data_path data_clean --seed 42 --ablate aux        # + aux head
```

Transformer depth study:

```
python exp_frmt_experiments.py --data_path data_clean --seed 42 --layers 1
python exp_frmt_experiments.py --data_path data_clean --seed 42 --layers 2
```

Comparisons:

```
python exp_frmt_experiments.py --data_path data_clean --seed 42 --backbone mobilenet_v3_small
python exp_frmt_experiments.py --data_path data_clean --seed 42 --backbone mobilenet_v2
python exp_frmt_experiments.py --data_path data_clean --seed 42 --backbone shufflenet_v2_x1_0
python exp_frmt_experiments.py --data_path data_clean --seed 42 --backbone efficientnet_b0
```

Outputs: `experiments/<tag>/best_model_full.pth`, `summary.json`, `history.json`.

## Evaluation

```
python exp_eval.py --data_path data_clean --ckpt experiments/frmt_s42/best_model_full.pth
```

## Notes

- Data augmentation: RandomResizedCrop + HFlip/VFlip/Rot/ColorJitter + RandAugment + Mixup.
- Loss: CE (label smoothing 0.1) + 0.5·Focal + 0.3·aux CE.
- Optimizer: AdamW, warmup 5 epochs + cosine decay 100 epochs, EMA (decay 0.999), early stopping.

## Citation

```
D. Mei and H. Wang, "FR-MobileTransformer: an ultra-lightweight hybrid architecture for robust plant disease recognition in edge-constrained environments," Journal of Electronic Imaging (submitted).
```
