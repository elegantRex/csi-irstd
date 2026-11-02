"""Original affine geometry used by CSI; no heatmap generation."""
import math
import random
from typing import Tuple
import cv2
import numpy as np

def affine_points(pt, t):
    ones = np.ones((pt.shape[0], 1), dtype=float)
    pt = np.concatenate([pt, ones], axis=1).T
    new_pt = np.dot(t, pt)
    return new_pt.T

class AffineTransform(object):
    """保持宽高比的缩放 + padding；可选随机缩放与旋转。"""

    def __init__(
        self,
        scale: Tuple[float, float] = None,
        rotation: Tuple[int, int] = None,
        fixed_size: Tuple[int, int] = (512, 512),
    ):
        self.scale = scale
        self.rotation = rotation
        self.fixed_size = fixed_size

    def __call__(self, img, target):
        h, w = img.shape[0], img.shape[1]
        dst_h, dst_w = self.fixed_size[0], self.fixed_size[1]

        scale_ratio = min(dst_h / h, dst_w / w)
        if self.scale is not None:
            scale_factor = random.uniform(*self.scale)
            scale_ratio *= scale_factor

        new_h = h * scale_ratio
        new_w = w * scale_ratio
        pad_h = (dst_h - new_h) / 2
        pad_w = (dst_w - new_w) / 2

        src = np.array([[0, 0], [w - 1, 0], [0, h - 1]], dtype=np.float32)
        dst = np.array(
            [
                [pad_w, pad_h],
                [dst_w - pad_w - 1, pad_h],
                [pad_w, dst_h - pad_h - 1],
            ],
            dtype=np.float32,
        )

        if self.rotation is not None:
            angle = random.randint(*self.rotation)
            angle_rad = angle / 180 * math.pi
            center = np.array([(dst_w - 1) / 2, (dst_h - 1) / 2])
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            for i in range(3):
                offset = dst[i] - center
                rotated_offset = np.array([
                    offset[0] * cos_a - offset[1] * sin_a,
                    offset[0] * sin_a + offset[1] * cos_a,
                ])
                dst[i] = center + rotated_offset

        trans = cv2.getAffineTransform(src, dst)
        dst_heatmap = dst / 4
        reverse_trans = cv2.getAffineTransform(dst_heatmap, src)

        resize_img = cv2.warpAffine(
            img,
            trans,
            (dst_w, dst_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        if "keypoints" in target:
            kps = target["keypoints"]
            mask = np.logical_and(kps[:, 0] != 0, kps[:, 1] != 0)
            kps[mask] = affine_points(kps[mask], trans)
            for i in range(len(kps)):
                if mask[i]:
                    x, y = kps[i]
                    if x < 0 or x >= dst_w or y < 0 or y >= dst_h:
                        kps[i] = [0, 0]
            target["keypoints"] = kps

        target["trans"] = trans
        target["reverse_trans"] = reverse_trans
        return resize_img, target
