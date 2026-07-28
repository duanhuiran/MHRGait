# MHRGait and MHRGait++

Official implementation of **MHRGait** for MHR-based gait recognition and
**MHRGait++** for end-to-end silhouette and MHR recognition, built with
[OpenGait](https://github.com/ShiqiYu/OpenGait).

## Results

All values are percentages. MHRGait++ uses original silhouettes, joint
training from scratch, and modality-balanced retrieval.

| Dataset | Method | Results |
| --- | --- | --- |
| SUSTech1K | MHRGait | Rank-1 **67.81**, Rank-5 **86.89** |
| SUSTech1K | MHRGait++ | Rank-1 **90.58**, Rank-5 **96.63** |
| CCPG | MHRGait | Gait Rank-1 mean **68.19**, ReID Rank-1 mean **83.47** |
| CCPG | MHRGait++ | Gait Rank-1 mean **90.01**, ReID Rank-1 mean **96.22** |
| CCGR-MINI | MHRGait | Rank-1 **10.86**, Rank-5 **24.86**, mAP **11.75**, mINP **4.77** |
| CCGR-MINI | MHRGait++ | Rank-1 **41.61**, Rank-5 **64.00**, mAP **37.99**, mINP **23.90** |
| CASIA-B* | MHRGait | NM **97.24**, BG **84.31**, CL **81.01**, Mean **87.52** |
| CASIA-B* | MHRGait++ | NM **98.68**, BG **95.56**, CL **89.35**, Mean **94.53** |

## Installation

Set up OpenGait, then install the additional requirements:

```bash
pip install -r requirements_mhrgait.txt
```

## Data Preparation

Convert raw SAM-3D-Body MHR dictionaries to `[T, 389]` arrays:

```bash
python datasets/MHRGait/prepare_mhr389.py \
  --input-root /path/to/raw_mhr_root \
  --output-root /path/to/mhr389-pkl \
  --workers 8
```

The script concatenates `global_rot`, `body_pose_params`, `hand_pose_params`,
`scale_params`, `shape_params`, and `expr_params` in this order. Use the
generated root directly for MHRGait.

### MHR dimensions and semantic grouping

Each extracted frame stores the complete **389-D MHR vector**. MHRGait does
not encode all 389 values: it selects **184 dimensions**, consisting of
76 body-pose dimensions and 108 hand-pose dimensions. MHRGait++ uses the
same selection in its MHR branch.

The selected parameters are organized into six body groups and two hand
groups. The indices below refer to the original 389-D vector and use Python's
half-open `range(start, end)` convention:

```python
BODY76_PARTS = (
    ("torso_head", tuple(range(3, 27))),
    ("right_arm", tuple(range(27, 37))),
    ("left_arm", tuple(range(37, 47))),
    ("right_leg_foot", tuple(range(47, 56)) + tuple(range(123, 127))),
    ("left_leg_foot", tuple(range(56, 65)) + tuple(range(119, 123))),
    ("body_flexibility", tuple(range(127, 133))),
)
HAND_PARTS = (
    ("left_hand", tuple(range(136, 190))),
    ("right_hand", tuple(range(190, 244))),
)
```

For MHRGait++, pair the original silhouettes with MHR389:

```bash
python datasets/MHRGait/build_multimodal_root.py \
  --silhouette-root /path/to/original-silhouette-pkl \
  --mhr-root /path/to/mhr389-pkl \
  --output-root /path/to/silhouette-mhr389-pkl
```

Both roots must follow `<subject>/<sequence>/<view>/*.pkl`. The script checks
sequence and frame-count consistency and creates symbolic links without
duplicating the data.

## Training and Evaluation

Set `data_cfg.dataset_root` in a config under `configs/mhrgait/` or
`configs/mhrgaitpp/`, then run:

```bash
# Train
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 opengait/main.py \
  --cfgs <CONFIG> --phase train --log_to_file

# Evaluate
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 opengait/main.py \
  --cfgs <CONFIG> --phase test --iter <CHECKPOINT_ITERATION> --log_to_file
```

The GPU count must match `evaluator_cfg.sampler.batch_size`.

## Acknowledgement

This code is based on [OpenGait](https://github.com/ShiqiYu/OpenGait) and is
intended for academic research.
