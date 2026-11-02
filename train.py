"""Train CSI on unchanged public splits, with explicit F1 checkpoint selection."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import shutil
import platform
import time
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.competitive_scene_net import CompetitiveSceneNet
from utils.scene_optimization import ModelEMA, lr_multiplier
from utils.scene_data import ScenePoints
from utils.scene_criterion import scene_loss
from evaluate import monitor_model
from utils.scene_training_record import improves_f1, atomic_json, append_epoch_csv


def worker_init(_):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    cv2.setNumThreads(0)


def atomic_save(obj, path):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(obj, temporary)
    os.replace(temporary, path)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-path', '--data_path', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--fixed-size', '--fixed_size', nargs=2, type=int, default=[512, 512])
    p.add_argument('--epochs', '--nEpochs', type=int, default=230)
    p.add_argument('--batch-size', '--batchSize', type=int, default=8)
    p.add_argument('--lr', type=float, default=0.0003)
    p.add_argument('--lr-warmup-epochs', type=int, default=0)
    p.add_argument('--train-hflip', type=float, default=0., help='Training-only synchronized horizontal flip probability')
    p.add_argument('--disable-target-feedback', action='store_true', help='A7: use raw image in background updates; keep background residual and competitor')
    p.add_argument('--disable-background-competition', action='store_true', help='Only set background competing logit to zero; retain both residual paths')
    p.add_argument('--disable-background-residual', action='store_true', help='Only use I instead of I-B for target updates; retain background and competition')
    p.add_argument('--reset-background-state', action='store_true', help='Reset background latent state to its initialization each round; retain all interaction paths')
    p.add_argument('--detach-render-feedback', action='store_true', help='Detach entire rendered target only on its feedback edge; keep the forward computation')
    p.add_argument('--detach-render-confidence', action='store_true', help='Experimental: stop score gradients through rendering; forward scores unchanged')
    p.add_argument('--ema-decay', type=float, default=0., help='0 disables EMA; otherwise evaluate and select EMA, also log raw metrics')
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--min-lr-ratio', type=float, default=.01)
    p.add_argument('--grad-clip', type=float, default=5.)
    p.add_argument('--location-weight', type=float, default=2.)
    p.add_argument('--background-weight', type=float, default=.001)
    p.add_argument('--focal-alpha', type=float, default=.25)
    p.add_argument('--focal-gamma', type=float, default=2.)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--encoder', choices=['compact', 'spire'], default='spire')
    p.add_argument('--channels', type=int, default=48)
    p.add_argument('--iterations', type=int, default=3)
    p.add_argument('--mode', choices=['competitive', 'independent', 'direct'], default='competitive')
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=2027)
    p.add_argument('--amp', choices=['off', 'bf16'], default='bf16')
    p.add_argument('--reconstruction-weight', type=float, default=0.0)
    p.add_argument('--reconstruction-warmup', type=int, default=10)
    p.add_argument('--save-every', '--save_interval', type=int, default=10)
    p.add_argument('--resume', help='Resume all optimizer/scheduler/RNG states from an epoch checkpoint')
    p.add_argument('--max-batches', type=int, default=0, help='Debug only; marks output as smoke training')
    p.add_argument('--eval-interval', '--eval_interval', type=int, default=1,
                   help='Monitor every N epochs and at final epoch; 0 disables')
    p.add_argument('--eval-split', '--eval_split', choices=['train', 'test'], default='test',
                   help='Default test follows requested original-protocol selection; train scores are fitting diagnostics')
    p.add_argument('--select-best-by', choices=['f1', 'none'], default='f1',
                   help='Select best evaluated F1, preserving the earlier epoch on ties')
    p.add_argument('--log-batches', type=int, default=50, help='Save batch losses every N steps; 0 disables')
    p.add_argument('--eval-max-images', type=int, default=0, help='0 evaluates entire split; positive is labelled partial')
    p.add_argument('--eval-threshold', type=float, default=.35)
    p.add_argument('--allow-monitoring-resume', action='store_true',
                   help='Explicitly allow trainer/monitor-only source changes; model/data/loss must match checkpoint hashes')
    return p.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.save_every < 1 or any(x % 8 for x in args.fixed_size):
        raise ValueError('Positive epochs/batch/save interval; input sides divisible by 8 required')
    if args.eval_interval < 0 or args.eval_max_images < 0:
        raise ValueError('Evaluation interval/limit cannot be negative')
    if args.select_best_by == 'f1' and not args.eval_interval:
        raise ValueError('F1 selection requires evaluation; use --select-best-by none to disable')
    if not (0 < args.focal_alpha < 1) or args.focal_gamma < 0:
        raise ValueError('Require 0 < focal-alpha < 1 and focal-gamma >= 0')
    if args.lr <= 0 or args.grad_clip <= 0 or not 0 <= args.min_lr_ratio <= 1:
        raise ValueError('Invalid learning rate, clipping or minimum LR ratio')
    if min(args.weight_decay, args.location_weight, args.background_weight, args.reconstruction_weight, args.reconstruction_warmup) < 0:
        raise ValueError('Loss weights, decay and warmup must be nonnegative')
    if args.channels < 1 or args.iterations < 1:
        raise ValueError('Channels and iterations must be positive')
    if not 0 <= args.lr_warmup_epochs < args.epochs or not 0 <= args.ema_decay < 1:
        raise ValueError('Require 0 <= warmup < epochs and 0 <= ema-decay < 1')
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f'Nonempty output: {output}; use a fresh run or --resume')
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if args.amp == 'bf16' and device.type == 'cuda' and not torch.cuda.is_bf16_supported():
        raise ValueError('BF16 unsupported; use --amp off')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cv2.setNumThreads(0)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dataset = ScenePoints(args.data_path, fixed_size=args.fixed_size, training=True, hflip_probability=args.train_hflip)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        pin_memory=device.type == 'cuda', worker_init_fn=worker_init,
                        generator=generator, collate_fn=ScenePoints.collate, persistent_workers=False)
    monitor_loader = None
    if args.eval_interval:
        monitor_data = ScenePoints(args.data_path, split=args.eval_split, fixed_size=args.fixed_size, training=False)
        monitor_loader = DataLoader(monitor_data, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.workers, pin_memory=device.type == 'cuda',
                                    worker_init_fn=worker_init, collate_fn=ScenePoints.collate,
                                    generator=torch.Generator().manual_seed(args.seed + 100000), persistent_workers=False)
    model = CompetitiveSceneNet(args.channels, args.iterations, args.mode, detach_render_confidence=args.detach_render_confidence, encoder=args.encoder, disable_target_feedback=args.disable_target_feedback,
                                disable_background_competition=args.disable_background_competition,
                                disable_background_residual=args.disable_background_residual,
                                reset_background_state=args.reset_background_state,
                                detach_render_feedback=args.detach_render_feedback).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * args.min_lr_ratio)
    if args.lr_warmup_epochs:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: lr_multiplier(epoch, args.epochs, args.lr_warmup_epochs, args.min_lr_ratio))
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay else None
    start = 0
    best = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        immutable = ['data_path', 'fixed_size', 'epochs', 'batch_size', 'lr', 'workers', 'seed', 'amp',
                     'mode', 'channels', 'iterations', 'max_batches', 'reconstruction_weight', 'reconstruction_warmup']
        new_defaults = dict(disable_target_feedback=False, encoder='compact', train_hflip=0., detach_render_confidence=False, lr_warmup_epochs=0, ema_decay=0., weight_decay=1e-4, min_lr_ratio=.01, grad_clip=5., location_weight=2., background_weight=.001, focal_alpha=.25, focal_gamma=2.)
        new_defaults.update(disable_background_competition=False, disable_background_residual=False,
                            reset_background_state=False, detach_render_feedback=False)
        for key, default in new_defaults.items():
            if checkpoint['args'].get(key, default) != vars(args)[key]:
                raise ValueError(f'Resume changes {key}; start a new experiment')
        for key in immutable:
            if checkpoint['args'][key] != vars(args)[key]:
                raise ValueError(f'Resume changes {key}; start a separately named experiment')
        if 'select_best_by' in checkpoint['args']:
            for key in ['select_best_by', 'eval_split', 'eval_interval', 'eval_threshold', 'eval_max_images']:
                if checkpoint['args'][key] != vars(args)[key]:
                    raise ValueError(f'Resume changes checkpoint-selection setting {key}')
        if checkpoint['dataset'] != dataset.receipt:
            raise ValueError('Dataset receipt changed since checkpoint')
        model.load_state_dict(checkpoint.get('raw_model', checkpoint['model']))
        if ema is not None:
            ema.model.load_state_dict(checkpoint['ema_model'])
            ema.updates = checkpoint['ema_updates']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start = checkpoint['epoch']
        torch.set_rng_state(checkpoint['rng']['torch'])
        random.setstate(checkpoint['rng']['python'])
        np.random.set_state(checkpoint['rng']['numpy'])
        generator.set_state(checkpoint['rng']['loader'])
        if device.type == 'cuda' and checkpoint['rng']['cuda'] is not None:
            torch.cuda.set_rng_state_all(checkpoint['rng']['cuda'])
        best = checkpoint.get('best_selection')
        if best is not None:
            saved_best = output / 'best_f1.pth'
            if not saved_best.exists():
                candidate = Path(args.resume).resolve().parent / 'best_f1.pth'
                if not candidate.exists():
                    raise FileNotFoundError('Resume tracks an earlier best epoch but best_f1.pth is missing')
                shutil.copy2(candidate, saved_best)
            best_checkpoint = torch.load(saved_best, map_location='cpu', weights_only=False)
            if best_checkpoint.get('best_selection') != best or best_checkpoint['epoch'] != best['epoch']:
                raise ValueError('Saved best checkpoint does not match resume selection history')
    source_paths = [Path(__file__), Path(__file__).parent / 'evaluate.py', Path(__file__).parent / 'model/competitive_scene_net.py',
                    Path(__file__).parent / 'utils/scene_data.py', Path(__file__).parent / 'utils/scene_criterion.py',
                    Path(__file__).parent / 'utils/scene_training_record.py', Path(__file__).parent / 'utils/scene_optimization.py']
    encoder_sources = ['model/scene_spire_encoder.py', 'model/spire_net.py', 'modules/ConditionalChannelWeighting.py', 'modules/LA_OCCW.py'] if args.encoder == 'spire' else []
    source_paths.extend(Path(__file__).parent / name for name in encoder_sources)
    sources = {str(p.relative_to(Path(__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    if args.resume and checkpoint['sources'] != sources:
        protected = ['utils/scene_optimization.py', 'model/competitive_scene_net.py', 'utils/scene_data.py', 'utils/scene_criterion.py'] + encoder_sources
        if not args.allow_monitoring_resume or any(checkpoint['sources'].get(k) != sources[k] for k in protected):
            raise ValueError('Source changed. Only trainer/monitor changes may be explicitly acknowledged with --allow-monitoring-resume; model/data/loss must stay identical')
    receipt = dict(args=vars(args), dataset=dataset.receipt, model=model.config,
                   parameters=sum(p.numel() for p in model.parameters()), sources=sources,
                   selection=f'best evaluated F1 on {args.eval_split}' if args.select_best_by == 'f1' else 'fixed final epoch',
                   protocol='official-test-selected reference' if args.select_best_by == 'f1' and args.eval_split == 'test' else 'training-fit diagnostic or fixed-final protocol',
                   monitoring=monitor_data.receipt if monitor_loader is not None else None,
                   resumed_source_hashes=checkpoint['sources'] if args.resume else None,
                   status='smoke only' if args.max_batches else 'training',
                   optimizer=dict(name='AdamW', weight_decay=args.weight_decay, gradient_clip=args.grad_clip),
                   scheduler=dict(name='WarmupCosine' if args.lr_warmup_epochs else 'CosineAnnealingLR', warmup=args.lr_warmup_epochs, T_max=args.epochs, eta_min=args.lr*args.min_lr_ratio),
                   matching=dict(distance=5, threshold=args.eval_threshold, value_range=.35, max_targets=8),
                   hardware=dict(device=str(device), gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None),
                   python_version=platform.python_version(), amp=args.amp, torch_version=torch.__version__)
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2))
    atomic_json(receipt, output / 'training_config.json')
    source_snapshot = output / 'source_snapshot'
    source_snapshot.mkdir(exist_ok=True)
    for path in source_paths:
        destination = source_snapshot / path.relative_to(Path(__file__).parent)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    print(json.dumps(receipt, indent=2), flush=True)
    for epoch in range(start, args.epochs):
        model.train()
        totals, samples = {}, 0
        epoch_start = time.perf_counter()
        rw = args.reconstruction_weight * min(1., (epoch + 1) / max(args.reconstruction_warmup, 1))
        length = min(len(loader), args.max_batches) if args.max_batches else len(loader)
        progress = tqdm(itertools.islice(loader, length), total=length, desc=f'Epoch {epoch+1}/{args.epochs}', dynamic_ncols=True)
        for step, (image, targets) in enumerate(progress):
            image = image.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp == 'bf16' and device.type == 'cuda'):
                stages = model(image)
            loss, parts = scene_loss(stages, image, targets, rw, background_weight=args.background_weight, mode=args.mode, location_weight=args.location_weight, focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite loss; checkpoint not overwritten')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            count = len(targets)
            samples += count
            for key, value in dict(loss=loss, **parts).items():
                totals[key] = totals.get(key, 0.) + float(value.detach()) * count
            progress.set_postfix(loss=f'{totals["loss"]/samples:.4f}', point=f'{totals["location"]/samples:.4f}', lr=f'{optimizer.param_groups[0]["lr"]:.2g}')
            if args.log_batches > 0 and (step == 0 or (step+1) % args.log_batches == 0 or step+1 == length):
                with (output / 'batch_metrics.jsonl').open('a') as file:
                    file.write(json.dumps(dict(epoch=epoch+1, batch=step+1, lr=optimizer.param_groups[0]['lr'],
                                              **{k:float(v.detach()) for k,v in dict(loss=loss,**parts).items()}))+'\n')
        progress.close()
        row = dict(epoch=epoch+1, **{k: v/samples for k, v in totals.items()},
                   lr=optimizer.param_groups[0]['lr'], seconds=time.perf_counter()-epoch_start,
                   reconstruction_weight=rw, samples=samples)
        scheduler.step()
        state = dict(model=model.state_dict(), model_config=model.config, optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(), epoch=epoch+1, args=vars(args), dataset=dataset.receipt,
                     sources=sources, rng=dict(torch=torch.get_rng_state(), python=random.getstate(),
                     numpy=np.random.get_state(), loader=generator.get_state(),
                     cuda=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None))
        if ema is not None:
            state.update(ema_model=ema.model.state_dict(), ema_updates=ema.updates)
        state['best_selection'] = best
        atomic_save(state, output / 'last.pth')
        if monitor_loader is not None and ((epoch + 1) % args.eval_interval == 0 or epoch + 1 == args.epochs):
            evaluation = monitor_model(model, monitor_loader, device, args.eval_split,
                                       threshold=args.eval_threshold, max_images=args.eval_max_images)
            if ema is not None:
                row['raw_evaluation'] = evaluation
                evaluation = monitor_model(ema.model, monitor_loader, device, args.eval_split, threshold=args.eval_threshold, max_images=args.eval_max_images)
                evaluation['weight_source'] = 'ema'
                print(f"[RAW F1] {row['raw_evaluation']['F1']:.4%}; [EMA F1] {evaluation['F1']:.4%}", flush=True)
            row['evaluation'] = evaluation
            if args.select_best_by == 'f1' and improves_f1(evaluation, best):
                best = dict(epoch=epoch+1, metric='F1', metrics=evaluation,
                            protocol=receipt['protocol'], evaluated_interval=args.eval_interval)
                state['best_selection'] = best
                selected_state = dict(state)
                if ema is not None:
                    selected_state.update(raw_model=state['model'], model=ema.model.state_dict(), weight_source='ema')
                atomic_save(selected_state, output / 'best_f1.pth')
                atomic_json(best, output / 'best_metrics.json')
                print(f"[BEST F1] epoch={epoch+1} F1={evaluation['F1']:.4%} -> {output / 'best_f1.pth'}", flush=True)
            with (output / 'evaluation.jsonl').open('a') as file:
                file.write(json.dumps(dict(epoch=epoch+1, **evaluation)) + '\n')
            print(f"[{args.eval_split.upper()} MONITOR] Epoch {epoch+1}: "
                  f"P={evaluation['precision']:.2%} Pd={evaluation['Pd']:.2%} F1={evaluation['F1']:.2%} "
                  f"TP={evaluation['TP']} FP={evaluation['FP']} FN={evaluation['FN']} "
                  f"images={evaluation['images']} partial={evaluation['partial']}", flush=True)
        if ema is not None:
            state.update(ema_model=ema.model.state_dict(), ema_updates=ema.updates)
        state['best_selection'] = best
        atomic_save(state, output / 'last.pth')
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            atomic_save(state, output / f'epoch{epoch+1:03d}.pth')
        row['best_epoch'] = best['epoch'] if best else None
        row['best_F1'] = best['metrics']['F1'] if best else None
        append_epoch_csv(output / 'training_log.csv', row, best)
        with (output / 'metrics.jsonl').open('a') as file:
            file.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
        with (output / 'training.log').open('a') as file:
            file.write(json.dumps(row)+'\n')
    atomic_json(dict(completed_epoch=args.epochs, best=best,
                     recommended_checkpoint='best_f1.pth' if best else 'last.pth',
                     protocol=receipt['protocol']), output / 'training_summary.json')
    if best:
        print(f"Training complete: best epoch={best['epoch']}, F1={best['metrics']['F1']:.4%}; evaluate best_f1.pth", flush=True)


if __name__ == '__main__':
    main()
