"""CSI points evaluated."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.competitive_scene_net import CompetitiveSceneNet
from utils.scene_data import ScenePoints
from utils.metrics import calculate_metrics, compute_prf1
from utils.transforms import affine_points


def decode_scene(stage, targets, threshold=.35, value_range=.35, max_targets=8):
    """Baseline 3x3 grid NMS / relative score gate; use learned coordinates.

    No quarter-pixel response-gradient offset: coordinates are directly learned.
    Predictions outside the original image are excluded before the output cap.
    """
    probability = stage['logits'].float().sigmoid()
    peaks = probability * (probability == F.max_pool2d(probability, 3, 1, 1))
    predictions = []
    for i, target in enumerate(targets):
        score = peaks[i].flatten().cpu().numpy()
        coords = stage['coords'][i].float().flatten(1).T.cpu().numpy()
        coords = affine_points(coords, target['inverse'])
        h, w = target['original_size']
        valid = (score > threshold) & (coords[:, 0] >= 0) & (coords[:, 0] < w) & (coords[:, 1] >= 0) & (coords[:, 1] < h)
        idx = np.flatnonzero(valid)
        idx = idx[np.argsort(-score[idx], kind='stable')]
        selected = []
        for j in idx:
            if any(abs(score[j] - score[k]) > value_range * max(score[j], score[k]) for k in selected):
                continue
            selected.append(j)
            if len(selected) == max_targets:
                break
        predictions.append([(float(coords[j, 0]), float(coords[j, 1]), float(score[j])) for j in selected])
    return predictions


@torch.no_grad()
def monitor_model(model, loader, device, split, threshold=.35, value_range=.35,
                  max_targets=8, tp_distance=5., max_images=0):
    """Read-only epoch monitoring, using exactly the standalone decoder/matcher.

    Train split scores measure fitting, not generalization. No head/threshold fit,
    optimizer change, or checkpoint selection is performed here.
    """
    was_training = model.training
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
           torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    model.eval()
    tp = fp = fn = images = pixels = 0
    progress = tqdm(loader, desc=f'Eval [{split}]', dynamic_ncols=True)
    try:
        for image, targets in progress:
            stage = model(image.to(device, non_blocking=True))[-1]
            predictions = decode_scene(stage, targets, threshold, value_range, max_targets)
            for target, pred in zip(targets, predictions):
                if max_images and images >= max_images:
                    break
                xy = np.asarray(pred, dtype=np.float32).reshape(-1, 3)[:, :2]
                a, b, c = calculate_metrics(xy, target['original_points'].reshape(-1, 2), tp_distance)
                tp += a; fp += b; fn += c
                images += 1
                pixels += int(np.prod(target['original_size']))
            precision, recall, f1 = compute_prf1(tp, fp, fn)
            progress.set_postfix(P=f'{precision:.2%}', Pd=f'{recall:.2%}', F1=f'{f1:.2%}')
            if max_images and images >= max_images:
                break
    finally:
        progress.close()
        model.train(was_training)
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state_all(rng[3])
    return dict(split=split, images=images, partial=bool(max_images), threshold=threshold,
                value_range=value_range, max_targets=max_targets, tp_distance=tp_distance,
                TP=tp, FP=fp, FN=fn, precision=precision, Pd=recall, F1=f1,
                Fa_point=fp / max(pixels, 1), FP_per_image=fp / max(images, 1),
                scope='training fit diagnostic' if split == 'train' else 'official test evaluation',
                inference_precision='FP32')


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data-path', '--data_path', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--split', choices=['train', 'test'], default='test')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', default='cuda')
    p.add_argument('--threshold', type=float, default=.35)
    p.add_argument('--value-range', type=float, default=.35)
    p.add_argument('--max-num-targets', type=int, default=8)
    p.add_argument('--tp-distance', type=float, default=5.)
    p.add_argument('--max-images', type=int, default=0, help='Smoke check only; marks result as partial')
    p.add_argument('--thresholds', nargs='+', type=float, help='Descriptive threshold sweep; never selects a best threshold')
    args = p.parse_args()
    if args.max_num_targets < 1:
        raise ValueError('Positive output cap required')
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    # Match FP32 monitoring in train_competitive_scene.py.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = CompetitiveSceneNet(**checkpoint['model_config']).to(args.device).eval()
    model.load_state_dict(checkpoint['model'])
    data = ScenePoints(args.data_path, split=args.split, fixed_size=checkpoint['args']['fixed_size'])
    loader = DataLoader(data, batch_size=args.batch_size, num_workers=args.workers, collate_fn=ScenePoints.collate)
    tp = fp = fn = images = pixels = 0
    rows = []
    sweep = {float(t): np.zeros(3, dtype=np.int64) for t in (args.thresholds or [])}
    torch.set_num_threads(4)
    for image, targets in tqdm(loader, desc='Evaluating', dynamic_ncols=True):
        stage = model(image.to(args.device))[-1]
        preds = decode_scene(stage, targets, args.threshold, args.value_range, args.max_num_targets)
        remaining = min(len(targets), args.max_images - images) if args.max_images else len(targets)
        for threshold, counts in sweep.items():
            alternative = decode_scene(stage, targets, threshold, args.value_range, args.max_num_targets)
            for target, pred in list(zip(targets, alternative))[:remaining]:
                counts += calculate_metrics(np.asarray(pred, dtype=np.float32).reshape(-1, 3)[:, :2],
                                            target['original_points'].reshape(-1, 2), args.tp_distance)
        for target, pred in zip(targets, preds):
            if args.max_images and images >= args.max_images:
                break
            xy = np.asarray(pred, dtype=np.float32).reshape(-1, 3)[:, :2]
            a, b, c = calculate_metrics(xy, target['original_points'].reshape(-1, 2), args.tp_distance)
            tp += a; fp += b; fn += c
            images += 1
            pixels += int(np.prod(target['original_size']))
            rows.append(dict(image_id=target['image_id'], detections=pred, TP=a, FP=b, FN=c))
        if args.max_images and images >= args.max_images:
            break
    precision, recall, f1 = compute_prf1(tp, fp, fn)
    report = dict(TP=tp, FP=fp, FN=fn, precision=precision, Pd=recall, F1=f1,
                  Fa_point=fp / pixels, images=images, dataset=data.receipt,
                  protocol='SPIRE utils.metrics.calculate_metrics GT-greedy matching; visible labels',
                  decoder='3x3 query-grid NMS, native relative gate, learned coordinates; in-image cap',
                  partial=bool(args.max_images), checkpoint_epoch=checkpoint['epoch'],
                  checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                  inference_precision='FP32', matmul_tf32=False, cudnn_tf32=False,
                  args=vars(args), per_image=rows)
    report['threshold_sweep'] = []
    for threshold, counts in sorted(sweep.items()):
        p_, r_, f_ = compute_prf1(*counts)
        report['threshold_sweep'].append(dict(threshold=threshold, TP=int(counts[0]), FP=int(counts[1]),
                                              FN=int(counts[2]), precision=float(p_), Pd=float(r_), F1=float(f_),
                                              Fa_point=float(counts[1] / pixels)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ['TP', 'FP', 'FN', 'precision', 'Pd', 'F1', 'Fa_point', 'images', 'partial']}, indent=2))


if __name__ == '__main__':
    main()
