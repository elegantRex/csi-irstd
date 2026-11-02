"""Read unchanged public point labels; use the baseline affine geometry only."""
import copy
import random
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.dataset import resolve_image_path
from utils.metrics import detect_dataset_format
from utils.transforms import AffineTransform, affine_points


def horizontal_flip_geometry(image, points, valid, inverse):
    """Flip an input-space sample and preserve its mapping to original pixels."""
    width = image.shape[1]
    flipped_points = points.copy()
    flipped_points[:, 0] = width - 1 - flipped_points[:, 0]
    flip = np.array([[-1., 0., width - 1], [0., 1., 0.], [0., 0., 1.]])
    inverse3 = np.vstack([inverse, [0., 0., 1.]])
    return image[:, ::-1].copy(), flipped_points, valid[:, ::-1].copy(), (inverse3 @ flip)[:2].astype(inverse.dtype)


class ScenePoints(Dataset):
    def __init__(self, root, split='train', fixed_size=(512, 512), training=False, hflip_probability=0.):
        if not 0 <= hflip_probability <= 1:
            raise ValueError("hflip_probability must be in [0,1]")
        if hflip_probability and not training:
            raise ValueError("Random flip is only allowed for training")
        self.hflip_probability = hflip_probability
        root = Path(root)
        if detect_dataset_format(str(root)) == 'format_b':
            image_root = root / 'images'
            annotations = root / 'annotations/annotations.json'
            split_file = root / f'img_idx/{split}.txt'
            selected = set(split_file.read_text().split())
        else:
            image_root = root / split / f'{split}_images'
            annotations = root / split / f'annotations/{split}.json'
            split_file, selected = None, None
        doc = json.loads(annotations.read_text())
        by_image = {}
        for ann in doc['annotations']:
            by_image.setdefault(ann['image_id'], []).extend(ann.get('keypoints', []))
        self.records = []
        for rec in sorted(doc['images'], key=lambda r: r['id']):
            if selected is not None and str(rec['id']) not in selected:
                continue
            kp = np.array(by_image.get(rec['id'], []), dtype=np.float32).reshape(-1, 3)
            points = kp[kp[:, 2] > 0, :2].copy()
            self.records.append(dict(image_id=rec['id'], path=resolve_image_path(str(image_root), rec['file_name']),
                                     points=points, width=rec['width'], height=rec['height']))
        if selected is not None and {str(r['image_id']) for r in self.records} != selected:
            raise ValueError('Some public split IDs are missing from annotation metadata')
        if not self.records:
            raise ValueError('Empty public split')
        self.fixed_size = tuple(fixed_size)
        self.affine = AffineTransform(scale=(1, 1.15) if training else (1, 1), fixed_size=self.fixed_size)
        self.receipt = dict(root=str(root.resolve()), split=split, images=len(self),
                            annotations=str(annotations), annotations_sha256=hashlib.sha256(annotations.read_bytes()).hexdigest(),
                            split_sha256=hashlib.sha256(split_file.read_bytes()).hexdigest() if split_file else None,
                            supervision='existing visible points only', preprocessing='baseline affine; grayscale /255; no PRPS')

        if hflip_probability:
            self.receipt['hflip_probability'] = hflip_probability
            self.receipt['preprocessing'] += '; synchronized horizontal flip'

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        rec = copy.deepcopy(self.records[index])
        image = cv2.imread(rec['path'], cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(rec['path'])
        if image.shape != (rec['height'], rec['width']):
            raise ValueError(f"Image/metadata dimensions differ: {rec['path']}")
        original_shape = image.shape
        image, geometry = self.affine(image, {})
        points = affine_points(rec['points'], geometry['trans']).astype(np.float32)
        h, w = self.fixed_size
        valid_points = (points[:, 0] >= 0) & (points[:, 0] < w) & (points[:, 1] >= 0) & (points[:, 1] < h)
        valid_image = cv2.warpAffine(np.ones(original_shape, np.uint8), geometry['trans'], (w, h), flags=cv2.INTER_NEAREST)
        inverse = cv2.invertAffineTransform(geometry['trans'])
        points = points[valid_points]
        if self.hflip_probability and random.random() < self.hflip_probability:
            image, points, valid_image, inverse = horizontal_flip_geometry(image, points, valid_image, inverse)
        target = dict(image_id=rec['image_id'], points=torch.from_numpy(points),
                      original_points=rec['points'], original_size=original_shape,
                      inverse=inverse,
                      valid=torch.from_numpy(valid_image.astype(np.float32))[None])
        return torch.from_numpy(image.copy()).float()[None] / 255., target

    @staticmethod
    def collate(batch):
        images, targets = zip(*batch)
        return torch.stack(images), list(targets)
