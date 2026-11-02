"""Explicit F1 checkpoint selection and flat, inspectable epoch records."""
import csv
import json
import os
from pathlib import Path


def improves_f1(evaluation, best):
    # Strictly greater: equal F1 retains the earlier evaluated checkpoint.
    return best is None or evaluation['F1'] > best['metrics']['F1']


def atomic_json(value, path):
    path = Path(path)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, path)


def append_epoch_csv(path, row, best):
    evaluation = row.get('evaluation', {})
    flat = {k: row.get(k) for k in ['epoch', 'lr', 'loss', 'existence', 'location', 'reconstruction',
                                  'background_tv', 'reconstruction_weight', 'samples', 'seconds']}
    flat.update({f'eval_{k}': evaluation.get(k) for k in
                 ['split', 'images', 'partial', 'precision', 'Pd', 'F1', 'Fa_point', 'TP', 'FP', 'FN']})
    flat['best_epoch'] = best['epoch'] if best else None
    flat['best_F1'] = best['metrics']['F1'] if best else None
    path = Path(path)
    header = not path.exists() or path.stat().st_size == 0
    with path.open('a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(flat))
        if header:
            writer.writeheader()
        writer.writerow(flat)
