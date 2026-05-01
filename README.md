# LARY — A Latent Action Representation Yielding Benchmark for Generalizable Vision-to-Action Alignment

<p align="center">
  <img src="assets/lary.jpg" alt="LARYBench" width="100%">
</p>

<p align="center">
  <a href="https://meituan-longcat.github.io/LARYBench/"><img src="https://img.shields.io/badge/Project-Page-blue?style=flat-square&logo=github" alt="Project Page"></a>
  &nbsp;
  <a href="https://arxiv.org/abs/2604.11689"><img src="https://img.shields.io/badge/arXiv-Paper-red?style=flat-square&logo=arxiv" alt="arXiv"></a>
  &nbsp;
  <a href="https://huggingface.co/datasets/meituan-longcat/LARYBench"><img src="https://img.shields.io/badge/🤗-HuggingFace-yellow?style=flat-square" alt="HuggingFace"></a>
  &nbsp;
  <a href="https://modelscope.cn/datasets/meituan-longcat/LARYBench"><img src="https://img.shields.io/badge/ModelScope-ModelHub-blue" alt="ModelScope"></a>
  &nbsp;
  <a href="https://discord.gg/EXsG52D8SW"><img src="https://img.shields.io/badge/Discord-Join%20Chat-5865F2?style=flat-square&logo=discord&logoColor=white"></a>
  &nbsp;
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="MIT License"></a>
</p>

**LARY** is a unified evaluation framework for **latent action representations**.
Given any model that produces latent action representations (LAMs or visual encoders), LARY provides three complementary evaluation pipelines:

| Pipeline | Task |
|---|---|
| **`get_latent_action`** | Extract latent action representations from videos or image pairs |
| **`classification`** | Probe how well latent actions capture *action semantics* (action-type recognition) |
| **`regression`** | Probe how well latent actions can *decode physical robot actions* (action regression) |

---

## News
- **[2026-05-01]** LARYBench now supports SigLIP2, relative-action regression evaluation (`target = action_tgt - action_src`), and a fast dataset integrity checker. Happy Labor Day!
- **[2026-04-27]** We have open-sourced all datasets on [HuggingFace](https://huggingface.co/datasets/meituan-longcat/LARYBench).
- **[2026-04-21]** We release the general LAMs trained in ablation studies, [LAPA-DINOv3](https://huggingface.co/AGI-Eval/LAPA-DINOv3) and  [LAPA-DINOv2](https://huggingface.co/AGI-Eval/LAPA-DINOv2). Even though these models are still rough experimental prototypes, with clear flaws in both training data and methods, we’re sharing them anyway to help push latent action research forward together. Have fun~
- **[2026-04-15]** We release partial training datasets due to the license limitation.
- **[2026-04-13]** We release the code, text annotations, and partial validation datasets. Training datasets are coming soon.

## Release Checklist

- [x] Code
- [x] Text annotations [link](https://github.com/meituan-longcat/LARYBench/tree/main/data)
- [x] Partial Validation datasets 
- [x] Partial Training datasets
- [x] Full datasets

---

## Table of Contents

1. [Overview](#overview)
2. [Contributions](#contributions)
3. [Environment Setup](#environment-setup)
4. [Data Preparation](#data-preparation)
5. [Quick Start](#quick-start)
6. [Relative-Action Regression](#relative-action-regression)
7. [Supported Models](#supported-models)
8. [Adding a Custom Model](#adding-a-custom-model)
9. [Supported Datasets](#supported-datasets)
10. [Evaluation Outputs](#evaluation-outputs)

---

## Overview

While the shortage of explicit action data limits Vision-Language-Action (VLA) models, human action videos offer a scalable yet unlabeled data source. A critical challenge in utilizing large-scale human video datasets lies in transforming visual signals into ontology-independent representations, known as latent actions. However, the capacity of latent action representation to derive robust control from visual observations has yet to be rigorously evaluated.

We introduce the Latent Action Representation Yielding (LARY) Benchmark, a unified framework for evaluating latent action representations on both high-level semantic actions (*what to do*) and low-level robotic control (*how to do*). The comprehensively curated dataset encompasses over one million videos (1,000 hours) spanning 151 action categories, alongside 620K image pairs and 595K motion trajectories across diverse embodiments and environments. Our experiments reveal two crucial insights: (i) General visual foundation models, trained without any action supervision, consistently outperform specialized embodied LAMs. (ii) Latent-based visual space is fundamentally better aligned to physical action space than pixel-based space. These results suggest that general visual representations inherently encode action-relevant knowledge for physical control, and that semantic-level abstraction serves as a fundamentally more effective pathway from vision to action than pixel-level reconstruction.

<p align="center">
  <img src="assets/framework.png" alt="LARYBench Overview" width="100%">
</p>

## Contributions

- **LARYBench**: We introduce LARYBench, a comprehensive benchmark that first decouples the evaluation of latent action representations from downstream policy performance. LARYBench probes representations along two complementary dimensions — high-level semantic action (*what to do*) encoding and the low-level physical dynamics required for robotic control (*how to do it*) — enabling direct, standardized measurement of representation quality itself.

- **Large-Scale Data Engine**: To support rigorous evaluation, we develop an automated data engine to re-segment and re-annotate a large-scale corpus, yielding 1.2M videos, 620K image pairs, and 595K trajectories across 151 action categories and 11 robotic embodiments, covering both human and robotic agents from egocentric and exocentric perspectives in simulated and real-world environments.

- **Key Findings**: Through systematic evaluation of 11 models, we reveal two consistent findings: (i) action-relevant features can emerge from large-scale visual pre-training without explicit action supervision, and (ii) latent-based feature spaces tend to align with robotic control better than pixel-based ones. These results suggest that future VLA systems may benefit more from leveraging general visual representations than from learning action spaces solely on scarce robotic data.

---

## Environment Setup

Use `larybench` as the base environment.

```bash
conda create -n larybench python=3.10 -y
conda activate larybench
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Some model families keep their original dependencies and should be configured from their upstream projects when you evaluate them:

| Model family | Environment guidance |
|---|---|
| `dinov2`, `dinov3`, `siglip2`, `dinov2-origin`, `dinov3-origin`, `siglip2-origin`, `lapa`, `magvit2`, `univla`, `flux2` | Use `larybench` |
| `vjepa2` | Follow [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) and activate your `vjepa2` env |
| `wan2-2` | Follow [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) and activate your `wan` env |
| `villa-x` | Follow [microsoft/villa-x](https://github.com/microsoft/villa-x) and set `VILLA_X_DIR` |

Configure paths in `env.sh`, then source it before running commands. Example:

```bash
LARY_ROOT=/your_name/code/LARYBench
LARY_LOG_DIR=/your_data_disk/LARYBench/logs
DATA_DIR=/your_data_disk/LARYBench/data
MODEL_DIR=/your_data_disk/LARYBench/models
LARY_LA_DIR=/your_data_disk/LARYBench/latent_actions
DINO_V2_PATH=/your_data_disk/LARYBench/models/DINOv2
DINO_V3_PATH=/your_data_disk/LARYBench/models/DINOv3
SIGLIP2_PATH=/your_data_disk/LARYBench/models/SigLIP2
source env.sh
```

## Data Preparation

The dataset root should be `DATA_DIR`:

```text
/your_data_disk/LARYBench/data/
├── classification/
│   ├── AgiBotWorld-Beta/
│   ├── Ego4D/
│   ├── EgoDex/
│   ├── EPIC-KITCHENS/
│   ├── HoloAssist/
│   ├── SSv2/
│   └── TACO/
├── regression/
│   ├── agibot_45/
│   ├── calvin/{train_stride5,val_stride5}/
│   ├── robocoin_10/
│   └── vlabench/
└── regression_relative/        # optional; generated for relative-action regression
```

Metadata CSVs are committed in this repository under `data/`. They store relative paths and are resolved against `DATA_DIR` at runtime.

Data setup flow:

1. Download the LARYBench archives from [HuggingFace](https://huggingface.co/datasets/meituan-longcat/LARYBench) or [ModelScope](https://modelscope.cn/datasets/meituan-longcat/LARYBench).
2. Extract them so the folder layout matches the example above.
3. Download SSv2 and EgoDex separately, then generate their clipped videos. This is required for the `Human_1st` classification task.

```bash
python utils/prepare_ssv2_egodex.py \
  --ssv2-root /path/to/20bn-something-something-v2 \
  --egodex-root /path/to/EgoDex \
  --output-dir $DATA_DIR/classification \
  --workers 16
```

To check whether images, videos, and regression `.npy` files exist and can be opened, run the integrity checker. Use `--groups` to scan only selected datasets.

```bash
python utils/check_dataset_integrity.py \
  --data-root $DATA_DIR \
  --output dataset_integrity_report.txt \
  --workers 64 \
  --timeout 3

python utils/check_dataset_integrity.py \
  --data-root $DATA_DIR \
  --groups CALVIN \
  --output dataset_integrity_calvin.txt \
  --workers 64 \
  --timeout 3
```

## Quick Start

The evaluation has two steps:

1. Extract latent actions from image pairs or videos with the model you want to evaluate.
2. Train a lightweight probe on the extracted latent actions: classification for action semantics, or regression for physical actions.

GPU defaults are explicit in the examples below. `extract` is single-GPU by default (`--gpus 0`); pass a comma-separated list (`--gpus 0,1,2,3,4,5,6,7`) to start one extraction partition per GPU and merge partition CSVs automatically. `classify` defaults to `--gpus 0,1,2,3,4,5,6,7`, so pass `--gpus 0` for single-card probing. `regress` follows `CUDA_VISIBLE_DEVICES`; if it is unset, the CLI assumes `0,1,2,3,4,5,6,7`, so set it explicitly for single-card runs.

### Example A: Regression on CALVIN with LAPA-DINOv2

```bash
conda activate larybench
source env.sh

# Step 1: extract latent actions from image pairs.
# --model: dinov3, siglip2, magvit2, lapa, univla, villa-x, dinov2-origin, dinov3-origin, siglip2-origin.
# --stride: calvin=5, vlabench=5, agibotbeta=45, robocoin=10.
# --split: calvin/vlabench use train,val; agibotbeta/robocoin use seen_train,seen_val.
# Single GPU by default; use --gpus 0,1,2,3 for multi-GPU partitioned extraction.
python -m lary.cli extract \
  --model dinov2 \
  --dataset calvin \
  --split train \
  --mode image \
  --stride 5 \
  --gpus 0,1,2,3

python -m lary.cli extract \
  --model dinov2 \
  --dataset calvin \
  --split val \
  --mode image \
  --stride 5 \
  --gpus 0,1,2,3

# Step 2: train the regression probe.
# Uses CUDA_VISIBLE_DEVICES; one visible GPU means single-card training, multiple visible GPUs use accelerate.
# Keep --dataset and --stride consistent with the extracted CSV names.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m lary.cli regress \
  --model dinov2 \
  --dataset calvin \
  --stride 5 \
  --model-type mlp
```

The extraction step writes latent-action `.npz` files under `$LARY_LA_DIR` and CSVs such as `data/train_la_calvin_5_dinov2.csv`. Regression logs and metrics are written under `$LARY_LOG_DIR/regression/`.

### Example B: Classification on Robot_1st with LAPA-DINOv2

```bash
conda activate larybench
source env.sh

# Step 1: extract latent actions from videos.
# --model: dinov3, siglip2, magvit2, lapa, univla, villa-x, dinov2-origin, dinov3-origin, siglip2-origin.
# --dataset: robot_1st, human_1st

python -m lary.cli extract \
  --model dinov2 \
  --dataset robot_1st \
  --split train \
  --mode video \
  --gpus 0,1,2,3

python -m lary.cli extract \
  --model dinov2 \
  --dataset robot_1st \
  --split val \
  --mode video \
  --gpus 0,1,2,3

# Step 2: train the classification probe.
# Robot_1st has 54 classes; Human_1st has 123 classes.
# --dim: dinov2=1024, dinov3=1024, siglip2=768, magvit2=18, 
#        lapa=1024, univla=128, villa-x:32,
#        dinov2-origin=1024, dinov3-origin=1024, vjepa2=1024,
#        siglip2-origin=768, wan2-2=48, flux2=128
python -m lary.cli classify \
  --model dinov2 \
  --dataset robot_1st \
  --dim 1024 \
  --classes 54 \
  --gpus 0,1,2,3
```

Classification outputs are written under `$LARY_LOG_DIR/classification/`.

## Relative-Action Regression

Absolute regression predicts the absolute action chunk. Relative regression predicts relative motion between two frames. Generate non-overwriting relative-action files first:

```bash
python utils/prepare_relative_actions.py \
  --dataset calvin \
  --input-root $DATA_DIR \
  --output-root $DATA_DIR \
  --csv data/train_la_calvin_5_dinov2.csv \
  --csv data/val_la_calvin_5_dinov2.csv \
  --workers 32
```

This creates `DATA_DIR/regression_relative/...` and writes relative-action mean/std statistics, for example `DATA_DIR/regression_relative/calvin/relative_action_stats_calvin.json`.

Run relative regression with the same latent-action CSVs:

```bash
CUDA_VISIBLE_DEVICES=0 python -m lary.cli regress \
  --model dinov2 \
  --dataset calvin \
  --stride 5 \
  --model-type mlp \
  --action-mode relative
```

## Supported Models

| Model key | What it extracts | Environment |
|---|---|---|
| `dinov2` | LAPA-DINOv2 latent actions | `larybench` |
| `dinov3` | LAPA-DINOv3 latent actions | `larybench` |
| `siglip2` | LAPA-SigLIP2 latent actions | `larybench` |
| `magvit2` | Open-MAGVIT2 based latent actions | `larybench`; set `MAGVIT2_CONFIG_PATH` and `MAGVIT2_TOKENIZER_PATH` |
| `dinov2-origin` | Raw DINOv2 visual features | `larybench` |
| `dinov3-origin` | Raw DINOv3 visual features | `larybench` |
| `siglip2-origin` | Raw SigLIP2 visual features | `larybench` |
| `lapa` | LAPA / LAQ latent actions | `larybench` |
| `univla` | UniVLA latent actions | `larybench`; set `UNIVLA_CKPT_PATH` |
| `villa-x` | villa-X latent actions | upstream villa-X env |
| `flux2` | FLUX.2 VAE features | `larybench`; set `AE_MODEL_PATH` |
| `vjepa2` | V-JEPA2 video features | upstream `vjepa2` env |
| `wan2-2` | Wan2.2 VAE features | upstream `wan` env |

## Adding a Custom Model

LARYBench only needs your model to convert a video or image pair into a numeric `tokens` array saved in each latent-action `.npz` file.

1. Add model-specific imports in [get_latent_action/dynamics.py](get_latent_action/dynamics.py), guarded by `USE_MODEL` if the dependency is optional.

```python
env_model = os.environ.get("USE_MODEL")
if env_model == "my-model":
    from my_project import MyModel
```

2. Register the model loader in `get_dynamic_tokenizer(model)`.

```python
elif model == "my-model":
    dynamics = MyModel.from_pretrained(os.environ["MY_MODEL_CKPT"]).cuda()
```

3. Add the forward branch in `get_latent_action(x, tokenizer, model_name)` and return either `(tokens, indices)` or `tokens`. Classification and regression use `tokens`; `tokens.shape[-1]` is the `--dim` value for classification.

```python
elif model_name == "my-model":
    tokens = tokenizer(x)          # expected shape: (B, ..., D)
    indices = np.array([])
```

4. If the model needs a different input format, add a matching branch in [lary/extract.py](lary/extract.py) for dataset preprocessing and batch execution. Reuse existing branches such as `dinov2-origin`, `vjepa2`, or `wan2-2` as templates.

5. Set any required environment variables in `env.sh`, then run:

```bash
python -m lary.cli extract \
  --model my-model \
  --dataset calvin \
  --split train/val \
  --mode image \
  --stride 5 \
  --gpus 0
```

After extraction creates `data/val_la_<dataset>_<stride>_my-model.csv` or `data/val_la_<dataset>_my-model.csv`, the existing `classify` and `regress` commands can evaluate it without model-specific changes.

## Supported Datasets

### Classification Datasets

| Dataset key | Splits | Input mode | Notes |
|---|---|---|---|
| `human_1st` | `train`, `val` | video | 123-class. Including EgoDex, SSv2, Ego4D, HoloAssist, EPIC-KITCHENS, TACO |
| `robot_1st` | `train`, `val` | video | 54-class. Made by AgiBotWorld-Beta |

### Regression Datasets

| Dataset key | Splits | Stride |
|---|---|---|
| `calvin` | `train`, `val` | 5 |
| `vlabench` | `train`, `val` | 5 |
| `vlabench_15` | `train`, `val` | 15 |
| `vlabench_30` | `train`, `val` | 30 |
| `agibotbeta` | `seen_train`, `seen_val` | 45 |
| `robocoin` | `seen_train`, `seen_val` | 10 |

## Evaluation Outputs

Extraction creates `.npz` latent actions under `$LARY_LA_DIR` and a metadata CSV under this repository's `data/` directory. Classification writes checkpoints, logs, confusion matrices, and class metrics under `$LARY_LOG_DIR/classification/`. Regression writes checkpoints, best-result CSVs, and trajectory visualizations under `$LARY_LOG_DIR/regression/`.

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{nie2026larylatentactionrepresentation,
      title={LARY: A Latent Action Representation Yielding Benchmark for Generalizable Vision-to-Action Alignment}, 
      author={Dujun Nie and Fengjiao Chen and Qi Lv and Jun Kuang and Xiaoyu Li and Xuezhi Cao and Xunliang Cai},
      year={2026},
      eprint={2604.11689},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.11689}, 
}
```

---

## Data Statements

LARYBench is built upon the following publicly available datasets. We gratefully acknowledge the efforts of their creators and ask users to comply with each dataset's respective license and terms of use.

| Dataset | Link |
|---|---|
| EgoDex | [github.com/apple/ml-egodex](https://github.com/apple/ml-egodex) |
| Something-Something V2 | [something-something-v2](https://www.qualcomm.com/developer/software/something-something-v-2-dataset) |
| Ego4D | [github.com/facebookresearch/Ego4d](https://github.com/facebookresearch/Ego4d) |
| HoloAssist | [holoassist.github.io](https://holoassist.github.io/) |
| EPIC-KITCHENS | [epic-kitchens.github.io](https://epic-kitchens.github.io/) |
| TACO | [taco2024.github.io](https://taco2024.github.io/) |
| AgiBotWorld-Beta | [github.com/OpenDriveLab/AgiBot-World](https://github.com/OpenDriveLab/AgiBot-World) |
| LIBERO | [github.com/Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) |
| RoboCOIN | [github.com/FlagOpen/RoboCOIN](https://github.com/FlagOpen/RoboCOIN) |
| VLABench | [github.com/OpenMOSS/VLABench](https://github.com/OpenMOSS/VLABench) |
| CALVIN | [github.com/mees/calvin](https://github.com/mees/calvin) |


## License

The code and tools in this repository are released under the [MIT License](LICENSE).

However, this dataset is derived from multiple third-party datasets, each governed by its own license. **The overall dataset is subject to the most restrictive terms among all included sources.** Users must comply with the respective licenses for each subset.

### Dataset License Summary

| Dataset | License |
|---|---|
| EPIC-KITCHENS | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) |
| TACO | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| AgiBotWorld-Beta | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |
| Ego4D, HoloAssist, LIBERO, RoboCOIN, VLABench, CALVIN | [MIT](https://opensource.org/licenses/MIT) |

### Important Notices

- **Non-commercial use only**: Subsets derived from EPIC-KITCHENS, and AgiBotWorld-Beta are restricted to **non-commercial research and educational purposes only**, due to the NC (NonCommercial) clauses in their respective licenses.

- **ShareAlike**: The AgiBotWorld-Beta-derived subset is subject to the **SA (ShareAlike)** clause. Any redistribution of this subset must be made available under the same [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

- **Attribution required**: All subsets derived from Creative Commons-licensed sources require proper attribution to the original dataset authors.

### Usage Recommendation

If you intend to use this dataset for **commercial purposes**, please use only the subsets released under MIT or CC BY 4.0 licenses (i.e., TACO and other datasets). The remaining subsets are **strictly non-commercial**.

For any questions regarding licensing, please refer to the original dataset sources or contact the respective dataset authors.

---

## Acknowledgements

We thank the following open-source projects for their contributions:

- [V-JEPA2](https://github.com/facebookresearch/vjepa2)
- [UniVLA](https://github.com/OpenDriveLab/UniVLA)
- [Wan2.2](https://github.com/Wan-Video/Wan2.2)
- [flux2](https://github.com/black-forest-labs/flux2)
- [villa-x](https://github.com/microsoft/villa-x)
- [Open-MAGVIT2](https://github.com/tencentarc/seed-voken)
- [SigLIP2](https://github.com/google-research/big_vision/tree/main)
- [DINOv2](https://github.com/facebookresearch/dinov2)
- [DINOv3](https://github.com/facebookresearch/dinov3)

## Support
Please contact us at <a href="mailto:longcat-team@meituan.com">longcat-team@meituan.com</a> or join our WeChat Group if you have any questions.

#### WeChat Group
<img src="assets/Wechat.png" width = "200" height = "200"  />
