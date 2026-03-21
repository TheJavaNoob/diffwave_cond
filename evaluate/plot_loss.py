#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def moving_average(values, window):
  if window <= 1 or window > len(values):
    return values
  out = []
  running = 0.0
  for i, v in enumerate(values):
    running += v
    if i >= window:
      running -= values[i - window]
    out.append(running / min(i + 1, window))
  return out


def load_loss_csv(path):
  steps = []
  losses = []
  with open(path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
      if 'step' not in row or 'loss' not in row:
        raise ValueError("CSV must contain 'step' and 'loss' columns")
      steps.append(int(row['step']))
      losses.append(float(row['loss']))
  if not steps:
    raise ValueError(f'No rows found in {path}')
  return steps, losses


def main():
  parser = argparse.ArgumentParser(description='Plot training and optional dev loss curves')
  parser.add_argument('--csv', default='model/train_loss.csv', help='path to training loss CSV file')
  parser.add_argument('--dev_csv', default='model/dev_loss.csv',
      help='path to dev loss CSV file (omit with --no_dev to skip)')
  parser.add_argument('--no_dev', action='store_true',
      help='skip plotting dev loss even if a dev CSV exists')
  parser.add_argument('--out', default='model/loss_curve.png', help='path to output image')
  parser.add_argument('--smooth', type=int, default=1, help='moving average window (default: 1 = no smoothing)')
  parser.add_argument('--show', action='store_true', help='display the plot window')
  args = parser.parse_args()

  csv_path = Path(args.csv)
  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)

  steps, losses = load_loss_csv(csv_path)
  smoothed = moving_average(losses, args.smooth)

  dev_steps = None
  dev_losses = None
  dev_smoothed = None
  if not args.no_dev:
    dev_csv_path = Path(args.dev_csv)
    if dev_csv_path.exists():
      dev_steps, dev_losses = load_loss_csv(dev_csv_path)
      dev_smoothed = moving_average(dev_losses, args.smooth)
    else:
      print(f'Dev CSV not found at {dev_csv_path}; plotting train loss only.')

  plt.figure(figsize=(10, 5))
  plt.plot(steps, losses, alpha=0.35, label='train raw loss')
  plt.plot(steps, smoothed, linewidth=2, label=f'train smoothed (window={args.smooth})')
  if dev_steps is not None and dev_losses is not None:
    plt.plot(dev_steps, dev_losses, alpha=0.4, label='dev raw loss')
    plt.plot(dev_steps, dev_smoothed, linewidth=2, label=f'dev smoothed (window={args.smooth})')
  plt.xlabel('Step')
  plt.ylabel('Loss')
  plt.title('Training and Dev Loss')
  plt.grid(True, alpha=0.3)
  plt.legend()
  plt.tight_layout()
  plt.savefig(out_path, dpi=150)

  print(f'Saved plot to {out_path}')
  if args.show:
    plt.show()


if __name__ == '__main__':
  main()
