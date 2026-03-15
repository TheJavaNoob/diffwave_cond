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
import random
import torch
import torch.nn.functional as F
import torchaudio

from glob import glob
from torch.utils.data.distributed import DistributedSampler


def _build_file_index(paths):
  files = []
  for path in paths:
    wavs = glob(f'{path}/**/*.wav', recursive=True)
    files += [(filename, path) for filename in wavs]
  return files


def _resolve_global_condition_path(audio_filename, data_root, params):
  suffix = params.global_conditioning_suffix
  if not suffix.endswith('.npy'):
    raise ValueError(f'global_conditioning_suffix must end with .npy, got {suffix}')

  # Default: sidecar label next to the audio file.
  if not params.global_conditioning_dir:
    sidecar = f'{audio_filename}{suffix}'
    if os.path.exists(sidecar):
      return sidecar
    stem_sidecar = f'{os.path.splitext(audio_filename)[0]}{suffix}'
    if os.path.exists(stem_sidecar):
      return stem_sidecar
    return None

  rel = os.path.relpath(audio_filename, data_root)
  rel_no_ext = os.path.splitext(rel)[0]
  root = params.global_conditioning_dir

  candidates = [
      os.path.join(root, f'{rel}{suffix}'),
      os.path.join(root, f'{rel_no_ext}{suffix}'),
      os.path.join(root, f'{os.path.basename(audio_filename)}{suffix}'),
      os.path.join(root, f'{os.path.splitext(os.path.basename(audio_filename))[0]}{suffix}'),
  ]
  for candidate in candidates:
    if os.path.exists(candidate):
      return candidate
  return None


def _load_global_condition(audio_filename, data_root, params):
  if not params.global_conditioning:
    return None

  label_path = _resolve_global_condition_path(audio_filename, data_root, params)
  if label_path is None:
    raise FileNotFoundError(f'Global label file not found for {audio_filename}')

  label = np.load(label_path)
  label = np.asarray(label, dtype=np.float32).reshape(-1)
  expected_dim = params.global_condition_dim
  if expected_dim is not None and label.shape[0] != expected_dim:
    raise ValueError(
        f'Expected global label dim {expected_dim}, got {label.shape[0]} from {label_path}')
  return label


def _infer_global_condition_dim(file_index, params):
  if not params.global_conditioning:
    return
  if params.global_condition_dim is not None:
    if params.global_condition_dim <= 0:
      raise ValueError(f'global_condition_dim must be > 0, got {params.global_condition_dim}')
    return

  for audio_filename, data_root in file_index:
    label_path = _resolve_global_condition_path(audio_filename, data_root, params)
    if label_path is None:
      continue
    label = np.load(label_path)
    label = np.asarray(label).reshape(-1)
    if label.shape[0] <= 0:
      raise ValueError(f'Global label file is empty: {label_path}')
    params.global_condition_dim = int(label.shape[0])
    return

  raise FileNotFoundError('Unable to infer global_condition_dim: no label .npy files found')


class ConditionalDataset(torch.utils.data.Dataset):
  def __init__(self, paths, params):
    super().__init__()
    self.file_index = _build_file_index(paths)
    self.params = params

  def __len__(self):
    return len(self.file_index)

  def __getitem__(self, idx):
    audio_filename, data_root = self.file_index[idx]
    spec_filename = f'{audio_filename}.spec.npy'
    signal, _ = torchaudio.load(audio_filename)
    spectrogram = np.load(spec_filename)
    return {
        'audio': signal[0],
        'spectrogram': spectrogram.T,
        'global_condition': _load_global_condition(audio_filename, data_root, self.params),
    }


class UnconditionalDataset(torch.utils.data.Dataset):
  def __init__(self, paths, params):
    super().__init__()
    self.file_index = _build_file_index(paths)
    self.params = params

  def __len__(self):
    return len(self.file_index)

  def __getitem__(self, idx):
    audio_filename, data_root = self.file_index[idx]
    signal, _ = torchaudio.load(audio_filename)
    return {
        'audio': signal[0],
        'spectrogram': None,
        'global_condition': _load_global_condition(audio_filename, data_root, self.params),
    }


class Collator:
  def __init__(self, params):
    self.params = params

  def collate(self, minibatch):
    samples_per_frame = self.params.hop_samples
    for record in minibatch:
      if self.params.unconditional:
          # Filter out records that aren't long enough.
          if len(record['audio']) < self.params.audio_len:
            del record['spectrogram']
            del record['audio']
            continue

          start = random.randint(0, record['audio'].shape[-1] - self.params.audio_len)
          end = start + self.params.audio_len
          record['audio'] = record['audio'][start:end]
          record['audio'] = np.pad(record['audio'], (0, (end - start) - len(record['audio'])), mode='constant')
      else:
          # Filter out records that aren't long enough.
          if len(record['spectrogram']) < self.params.crop_mel_frames:
            del record['spectrogram']
            del record['audio']
            continue

          start = random.randint(0, record['spectrogram'].shape[0] - self.params.crop_mel_frames)
          end = start + self.params.crop_mel_frames
          record['spectrogram'] = record['spectrogram'][start:end].T

          start *= samples_per_frame
          end *= samples_per_frame
          record['audio'] = record['audio'][start:end]
          record['audio'] = np.pad(record['audio'], (0, (end-start) - len(record['audio'])), mode='constant')

    audio = np.stack([record['audio'] for record in minibatch if 'audio' in record])
    if self.params.unconditional:
      labels = [record['global_condition'] for record in minibatch
          if 'audio' in record and record.get('global_condition') is not None]
      return {
        'audio': torch.from_numpy(audio),
        'spectrogram': None,
        'global_condition': torch.from_numpy(np.stack(labels)) if labels else None,
      }
    spectrogram = np.stack([record['spectrogram'] for record in minibatch if 'spectrogram' in record])
    labels = [record['global_condition'] for record in minibatch
          if 'audio' in record and record.get('global_condition') is not None]
    return {
        'audio': torch.from_numpy(audio),
        'spectrogram': torch.from_numpy(spectrogram),
      'global_condition': torch.from_numpy(np.stack(labels)) if labels else None,
    }

  # for gtzan
  def collate_gtzan(self, minibatch):
    ldata = []
    mean_audio_len = self.params.audio_len # change to fit in gpu memory
    # audio total generated time = audio_len * sample_rate
    # GTZAN statistics
    # max len audio 675808; min len audio sample 660000; mean len audio sample 662117
    # max audio sample 1; min audio sample -1; mean audio sample -0.0010 (normalized)
    # sample rate of all is 22050
    for data in minibatch:
      if data[0].shape[-1] < mean_audio_len:  # pad
        data_audio = F.pad(data[0], (0, mean_audio_len - data[0].shape[-1]), mode='constant', value=0)
      elif data[0].shape[-1] > mean_audio_len:  # crop
        start = random.randint(0, data[0].shape[-1] - mean_audio_len)
        end = start + mean_audio_len
        data_audio = data[0][:, start:end]
      else:
        data_audio = data[0]
      ldata.append(data_audio)
    audio = torch.cat(ldata, dim=0)
    return {
          'audio': audio,
          'spectrogram': None,
    }


def from_path(data_dirs, params, is_distributed=False):
  if params.global_conditioning:
    file_index = _build_file_index(data_dirs)
    _infer_global_condition_dim(file_index, params)
  if params.unconditional:
    dataset = UnconditionalDataset(data_dirs, params)
  else:#with condition
    dataset = ConditionalDataset(data_dirs, params)
  return torch.utils.data.DataLoader(
      dataset,
      batch_size=params.batch_size,
      collate_fn=Collator(params).collate,
      shuffle=not is_distributed,
      num_workers=os.cpu_count(),
      sampler=DistributedSampler(dataset) if is_distributed else None,
      pin_memory=True,
      drop_last=True)


def from_gtzan(params, is_distributed=False):
  dataset = torchaudio.datasets.GTZAN('./data', download=True)
  return torch.utils.data.DataLoader(
      dataset,
      batch_size=params.batch_size,
      collate_fn=Collator(params).collate_gtzan,
      shuffle=not is_distributed,
      num_workers=os.cpu_count(),
      sampler=DistributedSampler(dataset) if is_distributed else None,
      pin_memory=True,
      drop_last=True)
