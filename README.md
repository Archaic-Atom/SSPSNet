# SSPGNet — Sparse Self-Prompt Guided Stereo Matching

[![License MIT](https://img.shields.io/badge/license-MIT-green.svg?style=plastic)](./LICENSE)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-green.svg?style=plastic)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=plastic)

Official PyTorch implementation of the paper:

> **Sparse Self-Prompt Guided Stereo Matching for Real-World Generalization**
> *Sensors*, 2026.

SSPGNet is a domain-generalized stereo-matching network whose core mechanism is a *sparse self-prompt*: a confidence-thresholded sparse disparity map, self-estimated from vision-foundation-model (DINOv2 / DepthAnything v2) features via cost aggregation, is then refined into a dense disparity map through cross-attention-based sparse-to-dense propagation. Under the SceneFlow → {KITTI, Middlebury, ETH3D} zero-shot protocol, SSPGNet attains bad-pixel rates of **3.6 % / 4.4 % / 7.6 % / 2.1 %**, ranking first on three of the four benchmarks.

> **Note.** The internal code name of this model is `StereoA` (kept for backward compatibility with the JackFramework training scripts).

---

## Highlights

- **Sparse self-prompt mechanism** — confidence-thresholded sparse disparity, self-estimated from VFM features, used as a guidance prompt rather than the final output.
- **Sparse-to-dense propagation** — cross-attention-based stereo feature interaction (`Source/UserModelImplementation/Models/StereoA/Networks/_prompt.py`) progressively refines the prompt into a dense disparity.
- **Frozen foundation backbone** — the DINOv2 / DepthAnything v2 ViT-L/14 backbone is *frozen* throughout training; only **9.01 M** trainable parameters are updated.
- **Strong cross-domain generalization** — best peak performance on KITTI 2012 / KITTI 2015 / ETH3D (rank 1) and second-best on Middlebury (rank 2) under SceneFlow-only training.
- **In-the-wild evaluation** — qualitative results on real-world stereo pairs captured with a ZED 2 camera.

---

## Repository Layout

```
SSPGNet/
├── Source/
│   ├── main.py                                # entry point
│   ├── UserModelImplementation/
│   │   ├── Models/StereoA/                    # SSPGNet model code
│   │   │   ├── Networks/
│   │   │   │   ├── model.py                   # top-level network
│   │   │   │   ├── _feature_extraction.py     # VFM features (Stage 1) + transform (Stage 2)
│   │   │   │   ├── _cost_volume.py            # cost volume (Stage 3)
│   │   │   │   ├── _feature_matching.py       # 3D-hourglass aggregation (Stage 4)
│   │   │   │   └── _prompt.py                 # sparse prompt module (Stage 5, the novelty)
│   │   │   ├── _loss.py                       # smooth-L1 + multi-modal cross-entropy
│   │   │   └── _accuracy.py
│   │   └── Dataloaders/                       # SceneFlow / CreStereo / KITTI / MB / ETH3D loaders
│   ├── Libs/                                  # GANet & sync_bn CUDA extensions
│   └── Tools/                                 # dataset-list generation, evaluation
├── Scripts/                                   # train / test bash scripts
├── Datasets/                                  # CSV training / testing lists (generated)
├── Weights/                                   # released checkpoints (download separately)
├── profile_sspgnet.py                         # parameter / FLOPs / latency profiler
├── LICENSE
└── README.md
```

---

## Environment

```
OS:       Ubuntu 18.04 / 20.04
Python:   3.8.5+
PyTorch:  1.15.0+
CUDA:     compatible with the installed PyTorch
```

Install [JackFramework](https://github.com/Archaic-Atom/JackFramework) (training/eval harness):

```bash
git clone https://github.com/Archaic-Atom/JackFramework.git
cd JackFramework && ./install.sh
```

Compile the GANet / SyncBN CUDA extensions:

```bash
cd Source/Libs/GANet  && python setup.py build && cp -r build/lib* build/lib
cd ../sync_bn         && python setup.py build && cp -r build/lib* build/lib
```

---

## Datasets

Generate per-dataset CSV lists into `./Datasets/`:

```bash
DatasetListGenerator --dataset SceneFlow   --dataset_folder_path /path/to/SceneFlow/   --save_folder_path ./Datasets/
DatasetListGenerator --dataset CreStereo   --dataset_folder_path /path/to/CreStereo/   --save_folder_path ./Datasets/
DatasetListGenerator --dataset KITTI2012   --dataset_folder_path /path/to/Kitti2012/   --save_folder_path ./Datasets/
DatasetListGenerator --dataset KITTI2015   --dataset_folder_path /path/to/Kitti2015/   --save_folder_path ./Datasets/
DatasetListGenerator --dataset Middlebury  --dataset_folder_path /path/to/Middlebury/  --save_folder_path ./Datasets/
DatasetListGenerator --dataset ETH3D       --dataset_folder_path /path/to/ETH3D/       --save_folder_path ./Datasets/
```

---

## Training

Pre-training on SceneFlow or CreStereo:

```bash
# SceneFlow
./Scripts/start_train_dataset_model.sh

# CreStereo (recommended for cross-domain results in Table 5)
./Scripts/start_pre_train_dataset_model.sh
```

Hyperparameters (see Section 3.2 + Supplementary A of the paper):

| Hyperparameter | Value |
| --- | --- |
| Maximum disparity D | 196 |
| Foundation-model layer indices | {4, 11, 17, 23} |
| Patch size p (ViT-L/14) | 14 |
| Confidence threshold t (train / test) | 0.10 / 0.15 |
| Number of Laplacian components K | default of [Xu et al. 2024] |
| Affinity matrix channels | 8 (= 3×3 − 1, CSPN-style) |
| Optimizer | Adam (β₁ = 0.9, β₂ = 0.999) |
| Learning rate | 1 × 10⁻³ for 50 epochs, then 1 × 10⁻⁴ for 10 epochs |
| Batch size | 3 per GPU (×6 GPUs = 18) |
| Random crop | 518 × 266 |

---

## Cross-Domain Evaluation (Zero-Shot)

The pre-trained model is evaluated directly on the four target benchmarks without any fine-tuning:

```bash
./Scripts/start_test_kitti12_dataset_model.sh
./Scripts/start_test_kitti15_dataset_model.sh
./Scripts/start_test_middlebury_dataset_model.sh
./Scripts/start_test_eth3d_dataset_model.sh
```

Reproducing the numbers reported in **Table 5** of the paper (CreStereo-pre-trained):

| Benchmark | Threshold | SSPGNet | Rank |
| --- | --- | --- | --- |
| KITTI 2012 | > 3 px | **3.6 %** | 1 |
| KITTI 2015 | > 3 px | **4.4 %** | 1 |
| Middlebury | > 2 px |   7.6 %   | 2 |
| ETH3D      | > 1 px | **2.1 %** | 1 |

---

## Computational Profile

`profile_sspgnet.py` profiles parameter count, FLOPs, and wall-clock latency:

```bash
pip install thop
python profile_sspgnet.py
```

On a single NVIDIA RTX 3090 at the KITTI input envelope (378 × 1246):

| Metric | Value |
| --- | --- |
| Total parameters | 313.38 M |
| ↳ frozen (DINOv2 ViT-L/14) | 304.37 M |
| ↳ trainable | 9.01 M |
| FLOPs / forward | 2.92 T |
| Latency / pair (avg of 10 runs, 3 warm-up) | 0.609 s |

---

## Pre-trained Weights

Pre-trained SSPGNet checkpoints (SceneFlow and CreStereo variants) used to obtain the cross-domain numbers above will be released alongside this repository. See the [Releases](https://github.com/Archaic-Atom/SSPSNet/releases) page.

---

## Citation

If this work is useful for your research, please cite:

```bibtex
@article{li2026sspgnet,
  title   = {Sparse Self-Prompt Guided Stereo Matching for Real-World Generalization},
  author  = {Li, Hangbiao and Mo, Haojun and Li, Xing and Fang, Tao and Liu, Sikun and Yu, Shuzhen and Rao, Zhibo},
  journal = {Sensors},
  year    = {2026},
}
```

---

## License

MIT — see [LICENSE](./LICENSE).

---

## Acknowledgements

This codebase builds on [JackFramework](https://github.com/Archaic-Atom/JackFramework) and uses the DINOv2 / DepthAnything v2 vision foundation models from Meta AI as frozen feature extractors. We thank the authors of CFNet, GA-Net, NMRF, and Mask-CFNet for releasing reference implementations that informed our design.
