# csiNet — Competitive Scene Interpretation

Core experimental code for single-frame, point-supervised infrared small target detection. CSI passes the image through an encoder and, after encoding, alternates between updating the background state and the target queries, carrying cross-round feedback through a differentiable rendering. The encoder and the CSI branch are trained jointly from random initialization.

## Datasets

The loader reads the original public splits, the original images and the existing point annotations directly. It generates no pseudo-labels, re-splits no data and rewrites no annotation. SIRST4 uses a 512x512 input and SIRST-UAVB uses a 640x640 input. The paths below are examples and can be replaced with your own data root.

The recommended directory layout is as follows.

```text
SIRST4/                         # the same layout applies to UAVB
├── images/
├── annotations/
│   └── annotations.json
└── img_idx/
    ├── train.txt
    └── test.txt
```

The JSON uses COCO-style fields.

```json
{
  "images": [{"id": "000001", "file_name": "000001.png", "width": 320, "height": 256}],
  "annotations": [{"image_id": "000001", "keypoints": [123.0, 87.0, 2]}]
}
```

`keypoints` is a flat list of `[x, y, visibility]` triples and only the points with `visibility > 0` take part in training. An image may carry several points or none at all. Image identifiers must keep the same type throughout an annotation file. Each line of `train.txt` and `test.txt` holds an image identifier rather than a path with an extension, and string identifiers must keep their leading zeros. The width and height in the JSON must match the actual image. Where a UAVB annotation names a `.jpg` file and the disk holds a `.png` file of the same name, the reader falls back automatically and leaves the original file untouched.

The original per-split layout is also supported, namely `train/train_images/`, `train/annotations/train.json` and the matching `test/test_images/`, `test/annotations/test.json`. No segmentation mask is required.

Preprocessing keeps the original affine geometry and the valid-region mask. The grey levels are normalized to [0, 1], training applies a random scale in [1, 1.15] and evaluation uses a fixed scale factor of 1. The encoder replicates the single input channel to three channels internally.

## Code structure

```text
train.py                          training, F1 model selection, checkpoint resume, logging
evaluate.py                       decoding, matching on the original image, per-image evaluation

model/
  competitive_scene_net.py        target and background states, competition, differentiable rendering
  scene_spire_encoder.py          HRPE to CSI feature interface
  spire_net.py                    reused SPIRE network implementation
  modules/                        the two channel-weighting modules that HRPE depends on

utils/
  scene_data.py                   raw point annotation reading and the valid region
  scene_criterion.py              Hungarian matching, per-round loss, TV
  scene_optimization.py           scheduling and the optional EMA
  scene_training_record.py        logging and F1 model selection helpers
  dataset.py                      file name extension compatibility
  transforms.py                   affine transforms
  metrics.py                      data format and point matching
```
