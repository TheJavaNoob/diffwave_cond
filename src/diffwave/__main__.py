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

from argparse import ArgumentParser
from torch.cuda import device_count
from torch.multiprocessing import spawn

from diffwave.learner import train, train_distributed
from diffwave.params import params


def _get_free_port():
  import socketserver
  with socketserver.TCPServer(('localhost', 0), None) as s:
    return s.server_address[1]


def main(args):
  unconditional_enabled = args.unconditional or args.global_conditioning_only
  global_conditioning_enabled = args.global_conditioning or args.global_conditioning_only or \
      args.global_conditioning_dir is not None or \
      args.global_condition_dim is not None
  if args.global_condition_dim is not None and args.global_condition_dim <= 0:
    raise ValueError(f'global_condition_dim must be > 0, got {args.global_condition_dim}')

  params.override({
      'unconditional': unconditional_enabled,
      'global_conditioning': global_conditioning_enabled,
      'global_conditioning_dir': args.global_conditioning_dir,
      'global_conditioning_suffix': args.global_conditioning_suffix,
      'global_condition_dim': args.global_condition_dim,
  })
  replica_count = device_count()
  if replica_count > 1:
    if params.batch_size % replica_count != 0:
      raise ValueError(f'Batch size {params.batch_size} is not evenly divisble by # GPUs {replica_count}.')
    params.batch_size = params.batch_size // replica_count
    port = _get_free_port()
    spawn(train_distributed, args=(replica_count, port, args, params), nprocs=replica_count, join=True)
  else:
    train(args, params)


if __name__ == '__main__':
  parser = ArgumentParser(description='train (or resume training) a DiffWave model')
  parser.add_argument('model_dir',
      help='directory in which to store model checkpoints and training logs')
  parser.add_argument('data_dirs', nargs='+',
      help='space separated list of directories from which to read .wav files for training')
  parser.add_argument('--max_steps', default=None, type=int,
      help='maximum number of training steps')
  parser.add_argument('--resume', action='store_true', default=False,
      help='resume training from model_dir/weights.pt')
  parser.add_argument('--resume_checkpoint', default=None,
      help='path to a specific checkpoint .pt file to resume from')
  parser.add_argument('--fp16', action='store_true', default=False,
      help='use 16-bit floating point operations for training')
  parser.add_argument('--unconditional', action='store_true', default=False,
      help='disable local spectrogram conditioning')
  parser.add_argument('--global_conditioning_only', action='store_true', default=False,
      help='use only global label conditioning (equivalent to --unconditional --global_conditioning)')
  parser.add_argument('--global_conditioning', action='store_true', default=False,
      help='enable global label conditioning from .npy files')
  parser.add_argument('--global_conditioning_dir', default=None,
      help='optional root directory for label .npy files; defaults to audio directory')
  parser.add_argument('--global_conditioning_suffix', default='.label.npy',
      help='suffix for label files (default: .label.npy)')
  parser.add_argument('--global_condition_dim', default=None, type=int,
      help='label vector dimension; inferred from first label file when omitted')
  main(parser.parse_args())
