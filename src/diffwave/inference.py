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
import torchaudio
import torchaudio.functional as AF

from argparse import ArgumentParser

from diffwave.params import AttrDict, params as base_params
from diffwave.model import DiffWave


models = {}


def _to_mono_waveform(audio):
  if audio is None:
    return None
  if audio.dim() == 1:
    return audio
  if audio.dim() == 2:
    return audio.mean(dim=0)
  if audio.dim() == 3:
    return audio[0]
  return audio.reshape(-1)


def _save_denoise_step_plot(step_idx, diffusion_idx, pred_audio, gt_audio, output_dir):
  import matplotlib.pyplot as plt

  pred = _to_mono_waveform(pred_audio.detach().cpu().float())
  gt = _to_mono_waveform(gt_audio.detach().cpu().float())
  if pred is None or gt is None:
    return

  sample_count = int(min(pred.shape[-1], gt.shape[-1]))
  if sample_count <= 0:
    return

  x_axis = np.arange(sample_count)
  pred_np = pred[:sample_count].numpy()
  gt_np = gt[:sample_count].numpy()

  os.makedirs(output_dir, exist_ok=True)
  output_path = os.path.join(output_dir, f'denoise_step_{step_idx:03d}.png')

  plt.figure(figsize=(12, 4))
  plt.plot(x_axis, gt_np, label='Ground Truth', linewidth=1.0, alpha=0.9)
  plt.plot(x_axis, pred_np, label='Predicted', linewidth=1.0, alpha=0.8)
  plt.xlabel('Sample Index')
  plt.ylabel('Amplitude')
  plt.title(f'Denoising Step {step_idx} (diffusion index n={diffusion_idx})')
  plt.legend(loc='upper right')
  plt.tight_layout()
  plt.savefig(output_path, dpi=120)
  plt.close()

def predict(spectrogram=None, global_condition=None, model_dir=None,
            params=None, device=None, fast_sampling=False,
            ground_truth_audio=None, ground_truth_sample_rate=None,
            denoise_plots_dir=None):
  if device is None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Lazy load model.
  if not model_dir in models:
    if os.path.exists(f'{model_dir}/weights.pt'):
      checkpoint = torch.load(f'{model_dir}/weights.pt', map_location=device)
    else:
      checkpoint = torch.load(model_dir, map_location=device)
    model_params = AttrDict(dict(base_params))
    if 'params' in checkpoint and checkpoint['params'] is not None:
      model_params.override(checkpoint['params'])
    model = DiffWave(model_params).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    models[model_dir] = model

  model = models[model_dir]
  model.params.override(params)

  gt_for_plot = None
  if denoise_plots_dir is not None:
    if ground_truth_audio is None:
      raise ValueError('ground_truth_audio is required when denoise_plots_dir is set')
    gt_for_plot = ground_truth_audio.detach().cpu().float()
    if gt_for_plot.dim() == 1:
      gt_for_plot = gt_for_plot.unsqueeze(0)
    if ground_truth_sample_rate is not None and ground_truth_sample_rate != model.params.sample_rate:
      gt_for_plot = AF.resample(gt_for_plot, ground_truth_sample_rate, model.params.sample_rate)

  with torch.no_grad():
    # Change in notation from the DiffWave paper for fast sampling.
    # DiffWave paper -> Implementation below
    # --------------------------------------
    # alpha -> talpha
    # beta -> training_noise_schedule
    # gamma -> alpha
    # eta -> beta
    training_noise_schedule = np.array(model.params.noise_schedule)
    inference_noise_schedule = np.array(model.params.inference_noise_schedule) if fast_sampling else training_noise_schedule

    talpha = 1 - training_noise_schedule
    talpha_cum = np.cumprod(talpha)

    beta = inference_noise_schedule
    alpha = 1 - beta
    alpha_cum = np.cumprod(alpha)

    T = []
    for s in range(len(inference_noise_schedule)):
      for t in range(len(training_noise_schedule) - 1):
        if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
          twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
          T.append(t + twiddle)
          break
    T = np.array(T, dtype=np.float32)


    if not model.params.unconditional:
      if spectrogram is None:
        raise ValueError('spectrogram is required for locally conditioned models')
      if len(spectrogram.shape) == 2:# Expand rank 2 tensors by adding a batch dimension.
        spectrogram = spectrogram.unsqueeze(0)
      spectrogram = spectrogram.to(device)
      audio = torch.randn(spectrogram.shape[0], model.params.hop_samples * spectrogram.shape[-1], device=device)
    else:
      audio = torch.randn(1, model.params.audio_len, device=device)
    noise_scale = torch.from_numpy(alpha_cum**0.5).float().unsqueeze(1).to(device)
    if global_condition is not None:
      if len(global_condition.shape) == 1:
        global_condition = global_condition.unsqueeze(0)
      global_condition = global_condition.to(device)

    for n in range(len(alpha) - 1, -1, -1):
      c1 = 1 / alpha[n]**0.5
      c2 = beta[n] / (1 - alpha_cum[n])**0.5
      audio = c1 * (audio - c2 * model(
          audio,
          torch.tensor([T[n]], device=audio.device),
          spectrogram,
          global_condition,
      ).squeeze(1))
      if n > 0:
        # Add noise, except for the last step.
        # This is equivalent to sampling from q(x_{n-1} | x_n) 
        noise = torch.randn_like(audio)
        sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
        audio += sigma * noise
      audio = torch.clamp(audio, -1.0, 1.0)

      if denoise_plots_dir is not None:
        step_idx = len(alpha) - n
        _save_denoise_step_plot(
            step_idx=step_idx,
            diffusion_idx=n,
            pred_audio=audio,
            gt_audio=gt_for_plot,
            output_dir=denoise_plots_dir,
        )
  return audio, model.params.sample_rate


def main(args):
  if args.spectrogram_path:
    spectrogram = torch.from_numpy(np.load(args.spectrogram_path))
  else:
    spectrogram = None
  if args.global_condition_path:
    global_condition = torch.from_numpy(np.load(args.global_condition_path)).float().reshape(-1)
  else:
    global_condition = None

  if args.ground_truth_wav:
    ground_truth_audio, ground_truth_sr = torchaudio.load(args.ground_truth_wav)
  else:
    ground_truth_audio = None
    ground_truth_sr = None

  audio, sr = predict(
      spectrogram,
      global_condition=global_condition,
      model_dir=args.model_dir,
      fast_sampling=args.fast,
      params=None,
      ground_truth_audio=ground_truth_audio,
      ground_truth_sample_rate=ground_truth_sr,
      denoise_plots_dir=args.denoise_plots_dir)
  torchaudio.save(args.output, audio.cpu(), sample_rate=sr)


if __name__ == '__main__':
  parser = ArgumentParser(description='runs inference on a spectrogram file generated by diffwave.preprocess')
  parser.add_argument('model_dir',
      help='directory containing a trained model (or full path to weights.pt file)')
  parser.add_argument('--spectrogram_path', '-s',
      help='path to a spectrogram file generated by diffwave.preprocess')
  parser.add_argument('--output', '-o', default='output.wav',
      help='output file name')
  parser.add_argument('--global_condition_path', '-g',
      help='path to a .npy global conditioning vector')
  parser.add_argument('--ground_truth_wav', default=None,
      help='optional path to a ground-truth wav used for per-step amplitude plots')
  parser.add_argument('--denoise_plots_dir', default=None,
      help='optional directory to save ground-truth vs prediction plots for each denoising step')
  parser.add_argument('--fast', '-f', action='store_true',
      help='fast sampling procedure')
  main(parser.parse_args())
