# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import os
import torch
import torch.nn as nn
from glob import glob

from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from diffwave.dataset import from_path, from_gtzan
from diffwave.model import DiffWave
from diffwave.params import AttrDict


def _nested_map(struct, map_fn):
  if isinstance(struct, tuple):
    return tuple(_nested_map(x, map_fn) for x in struct)
  if isinstance(struct, list):
    return [_nested_map(x, map_fn) for x in struct]
  if isinstance(struct, dict):
    return { k: _nested_map(v, map_fn) for k, v in struct.items() }
  return map_fn(struct)


class DiffWaveLearner:
  def __init__(self,
               model_dir,
               model,
               dataset,
               optimizer,
               params,
               dev_dataset=None,
               eval_interval_steps=None,
               dev_max_eval_batches=None,
               *args,
               **kwargs):
    os.makedirs(model_dir, exist_ok=True)
    self.model_dir = model_dir
    self.model = model
    self.dataset = dataset
    self.dev_dataset = dev_dataset
    self.optimizer = optimizer
    self.params = params
    self.eval_interval_steps = eval_interval_steps
    self.dev_max_eval_batches = dev_max_eval_batches
    self.autocast = torch.cuda.amp.autocast(enabled=kwargs.get('fp16', False))
    self.scaler = torch.cuda.amp.GradScaler(enabled=kwargs.get('fp16', False))
    self.step = 0
    self.is_master = True

    beta = np.array(self.params.noise_schedule)
    noise_level = np.cumprod(1 - beta)
    self.noise_level = torch.tensor(noise_level.astype(np.float32))
    self.loss_fn = nn.L1Loss()
    self.summary_writer = None

  def state_dict(self):
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      model_state = self.model.module.state_dict()
    else:
      model_state = self.model.state_dict()
    return {
        'step': self.step,
        'model': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items() },
        'optimizer': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items() },
        'params': dict(self.params),
        'scaler': self.scaler.state_dict(),
    }

  def load_state_dict(self, state_dict):
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      self.model.module.load_state_dict(state_dict['model'])
    else:
      self.model.load_state_dict(state_dict['model'])
    self.optimizer.load_state_dict(state_dict['optimizer'])
    self.scaler.load_state_dict(state_dict['scaler'])
    self.step = state_dict['step']

  def save_to_checkpoint(self, filename='weights'):
    save_basename = f'{filename}-{self.step}.pt'
    save_name = f'{self.model_dir}/{save_basename}'
    link_name = f'{self.model_dir}/{filename}.pt'
    torch.save(self.state_dict(), save_name)
    if os.name == 'nt':
      torch.save(self.state_dict(), link_name)
    else:
      if os.path.islink(link_name):
        os.unlink(link_name)
      os.symlink(save_basename, link_name)
    self._delete_old_checkpoints(filename)

  def _delete_old_checkpoints(self, filename='weights'):
    keep_last = int(getattr(self.params, 'checkpoint_keep_last', 5) or 5)
    keep_last = max(1, keep_last)
    pattern = os.path.join(self.model_dir, f'{filename}-*.pt')
    checkpoints = []
    for path in glob(pattern):
      stem = os.path.basename(path)
      prefix = f'{filename}-'
      suffix = '.pt'
      if not (stem.startswith(prefix) and stem.endswith(suffix)):
        continue
      step_str = stem[len(prefix):-len(suffix)]
      if step_str.isdigit():
        checkpoints.append((int(step_str), path))
    checkpoints.sort(key=lambda x: x[0])
    stale = checkpoints[:-keep_last]
    for _, path in stale:
      try:
        os.remove(path)
      except FileNotFoundError:
        pass

  def _append_loss_log(self, step, loss):
    if isinstance(self.grad_norm, torch.Tensor):
      grad_norm = float(self.grad_norm.detach().cpu().item())
    else:
      grad_norm = float(self.grad_norm)
    log_path = os.path.join(self.model_dir, 'train_loss.csv')
    write_header = not os.path.exists(log_path)
    with open(log_path, 'a') as f:
      if write_header:
        f.write('step,loss,grad_norm\n')
      f.write(f'{step},{float(loss.detach().cpu().item())},{grad_norm}\n')

  def _append_dev_loss_log(self, step, loss, batches):
    log_path = os.path.join(self.model_dir, 'dev_loss.csv')
    write_header = not os.path.exists(log_path)
    with open(log_path, 'a') as f:
      if write_header:
        f.write('step,loss,batches\n')
      f.write(f'{step},{float(loss)},{int(batches)}\n')

  def restore_from_checkpoint(self, filename='weights'):
    try:
      self.restore_from_checkpoint_path(f'{self.model_dir}/{filename}.pt')
      return True
    except FileNotFoundError:
      return False

  def restore_from_checkpoint_path(self, checkpoint_path):
    if not os.path.isabs(checkpoint_path):
      model_relative_path = os.path.join(self.model_dir, checkpoint_path)
      if os.path.exists(model_relative_path):
        checkpoint_path = model_relative_path
    checkpoint = torch.load(checkpoint_path)
    self.load_state_dict(checkpoint)

  def train(self, max_steps=None):
    device = next(self.model.parameters()).device
    eval_interval = self.eval_interval_steps or len(self.dataset)
    while True:
      for features in tqdm(self.dataset, desc=f'Epoch {self.step // len(self.dataset)}') if self.is_master else self.dataset:
        if max_steps is not None and self.step >= max_steps:
          return
        features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
        loss = self.train_step(features)
        if torch.isnan(loss).any():
          raise RuntimeError(f'Detected NaN loss at step {self.step}.')
        if self.is_master:
          self._append_loss_log(self.step, loss)
          if self.step % 50 == 0:
            self._write_summary(self.step, features, loss)
            print(f'Step {self.step}, Loss: {loss.item():.4f}, Grad Norm: {self.grad_norm:.4f}')
          if self.dev_dataset is not None and self.step % eval_interval == 0:
            dev_loss = self.eval_step()
            self._append_dev_loss_log(self.step, dev_loss, self._dev_batches)
            writer = self.summary_writer or SummaryWriter(self.model_dir, purge_step=self.step)
            writer.add_scalar('dev/loss', dev_loss, self.step)
            writer.flush()
            self.summary_writer = writer
            print(f'Step {self.step}, Dev Loss: {dev_loss:.4f} ({self._dev_batches} batches)')
          if self.step % len(self.dataset) == 0:
            self.save_to_checkpoint()
        self.step += 1

  def _compute_loss(self, features):
    audio = features['audio']
    spectrogram = features['spectrogram']
    global_condition = features.get('global_condition')

    N, T = audio.shape
    del T
    device = audio.device
    self.noise_level = self.noise_level.to(device)

    t = torch.randint(0, len(self.params.noise_schedule), [N], device=audio.device)
    noise_scale = self.noise_level[t].unsqueeze(1)
    noise_scale_sqrt = noise_scale**0.5
    noise = torch.randn_like(audio)
    noisy_audio = noise_scale_sqrt * audio + (1.0 - noise_scale)**0.5 * noise

    predicted = self.model(noisy_audio, t, spectrogram, global_condition)
    loss = self.loss_fn(noise, predicted.squeeze(1))
    return loss

  def train_step(self, features):
    for param in self.model.parameters():
      param.grad = None

    with self.autocast:
      loss = self._compute_loss(features)

    self.scaler.scale(loss).backward()
    self.scaler.unscale_(self.optimizer)
    self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.params.max_grad_norm or 1e9)
    self.scaler.step(self.optimizer)
    self.scaler.update()
    return loss

  @torch.no_grad()
  def eval_step(self):
    if self.dev_dataset is None:
      return None

    device = next(self.model.parameters()).device
    max_batches = self.dev_max_eval_batches
    losses = []
    batches = 0

    was_training = self.model.training
    self.model.eval()
    iterator = tqdm(self.dev_dataset, desc=f'Dev @ step {self.step}') if self.is_master else self.dev_dataset
    for features in iterator:
      if max_batches is not None and batches >= max_batches:
        break
      features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
      with self.autocast:
        loss = self._compute_loss(features)
      if torch.isnan(loss).any():
        raise RuntimeError(f'Detected NaN dev loss at step {self.step}.')
      losses.append(float(loss.detach().cpu().item()))
      batches += 1

    if was_training:
      self.model.train()
    self._dev_batches = batches
    if not losses:
      raise RuntimeError('Dev dataloader produced no batches; cannot compute dev loss.')
    return float(np.mean(losses))

  def _write_summary(self, step, features, loss):
    writer = self.summary_writer or SummaryWriter(self.model_dir, purge_step=step)
    writer.add_audio('feature/audio', features['audio'][0], step, sample_rate=self.params.sample_rate)
    if not self.params.unconditional:
      writer.add_image('feature/spectrogram', torch.flip(features['spectrogram'][:1], [1]), step)
    writer.add_scalar('train/loss', loss, step)
    writer.add_scalar('train/grad_norm', self.grad_norm, step)
    writer.flush()
    self.summary_writer = writer


def _train_impl(replica_id, model, dataset, dev_dataset, args, params):
  torch.backends.cudnn.benchmark = True
  opt = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

  learner = DiffWaveLearner(
      args.model_dir,
      model,
      dataset,
      opt,
      params,
      dev_dataset=dev_dataset,
      eval_interval_steps=args.eval_interval_steps,
      dev_max_eval_batches=args.dev_max_eval_batches,
      fp16=args.fp16)
  learner.is_master = (replica_id == 0)
  if args.resume_checkpoint is not None:
    learner.restore_from_checkpoint_path(args.resume_checkpoint)
  elif args.resume:
    if not learner.restore_from_checkpoint():
      raise FileNotFoundError(f'--resume requested but no checkpoint found at {args.model_dir}/weights.pt')
  learner.train(max_steps=args.max_steps)


def train(args, params):
  if args.data_dirs[0] == 'gtzan':
    dataset = from_gtzan(params)
  else:
    dataset = from_path(args.data_dirs, params)
  dev_dataset = from_path(args.dev_data_dirs, params) if args.dev_data_dirs else None
  model = DiffWave(params)#.cuda()
  _train_impl(0, model, dataset, dev_dataset, args, params)


def train_distributed(replica_id, replica_count, port, args, params):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = str(port)
  torch.distributed.init_process_group('nccl', rank=replica_id, world_size=replica_count)
  if args.data_dirs[0] == 'gtzan':
    dataset = from_gtzan(params, is_distributed=True)
  else:
    dataset = from_path(args.data_dirs, params, is_distributed=True)
  dev_dataset = None
  if replica_id == 0 and args.dev_data_dirs:
    dev_dataset = from_path(args.dev_data_dirs, params)
  device = torch.device('cuda', replica_id)
  torch.cuda.set_device(device)
  model = DiffWave(params).to(device)
  model = DistributedDataParallel(model, device_ids=[replica_id])
  _train_impl(replica_id, model, dataset, dev_dataset, args, params)
