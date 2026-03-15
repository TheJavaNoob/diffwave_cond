import json
import sys

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torchaudio

from diffwave.inference import predict


def _parse_point(values, name):
  if values is None:
    return None
  if len(values) != 3:
    raise ValueError(f'{name} must have exactly 3 values: x y z')
  return np.asarray(values, dtype=np.float32)


def _load_spectrogram(path):
  if path is None:
    return None
  return torch.from_numpy(np.load(path))


def _load_feature(path):
  feat = np.load(path)
  return np.asarray(feat, dtype=np.float32).reshape(-1)


def _build_pair_feature_from_points(metadata_path, mesh_path, source_point, receiver_point,
                                    source_name, receiver_name, occlusion_list, n1):
  project_root = Path(__file__).resolve().parents[2]
  preprocessing_dir = project_root / 'preprocessing'
  if str(preprocessing_dir) not in sys.path:
    sys.path.insert(0, str(preprocessing_dir))

  from data_utils import probe_environment_dict

  if not metadata_path.exists():
    raise FileNotFoundError(f'Metadata file not found: {metadata_path}')
  if not mesh_path.exists():
    raise FileNotFoundError(f'Mesh file not found: {mesh_path}')

  with open(metadata_path, 'r') as f:
    all_metadata = json.load(f)

  source_points = {}
  receiver_points = {}
  for row in all_metadata:
    s_idx = int(row['source_idx'])
    r_idx = int(row['receiver_idx'])
    s_name = f'source_{s_idx:02d}'
    r_name = f'source_{s_idx:02d}_rec_{r_idx:02d}'

    source_points[s_name] = np.asarray(row['source_position'], dtype=np.float64)
    receiver_points[r_name] = np.asarray(row['receiver_position'], dtype=np.float64)

  source_points[source_name] = np.asarray(source_point, dtype=np.float64)
  receiver_points[receiver_name] = np.asarray(receiver_point, dtype=np.float64)

  source_features = probe_environment_dict(
      named_points=source_points,
      mesh_path=str(mesh_path),
      occlusion_lst=occlusion_list,
      direction_method='Fibonacci',
      N1=n1,
      N2=None,
  )

  receiver_features = probe_environment_dict(
      named_points=receiver_points,
      mesh_path=str(mesh_path),
      occlusion_lst=occlusion_list,
      direction_method='Fibonacci',
      N1=n1,
      N2=None,
  )

  src_loc = np.asarray(source_point, dtype=np.float32).reshape(-1)
  rec_loc = np.asarray(receiver_point, dtype=np.float32).reshape(-1)
  src_feat = np.asarray(source_features[source_name], dtype=np.float32).reshape(-1)
  rec_feat = np.asarray(receiver_features[receiver_name], dtype=np.float32).reshape(-1)
  return np.concatenate([src_loc, rec_loc, src_feat, rec_feat], axis=0)


def _default_occlusion_list():
  return np.array([10, 25, 50, 75, 100, 125, 150, 200, 250, 300, 350, 400, 450, 500], dtype=np.float32) * 0.01


def main(args):
  feature_mode = args.feature_npy is not None
  point_mode = args.source_point is not None or args.receiver_point is not None
  if feature_mode == point_mode:
    raise ValueError('Choose exactly one mode: --feature_npy OR both --source_point and --receiver_point')

  if feature_mode:
    feature = _load_feature(args.feature_npy)
  else:
    source_point = _parse_point(args.source_point, '--source_point')
    receiver_point = _parse_point(args.receiver_point, '--receiver_point')
    if source_point is None or receiver_point is None:
      raise ValueError('Point mode requires both --source_point and --receiver_point')
    if args.metadata_path is None or args.mesh_path is None:
      raise ValueError('Point mode requires --metadata_path and --mesh_path')

    feature = _build_pair_feature_from_points(
        metadata_path=Path(args.metadata_path),
        mesh_path=Path(args.mesh_path),
        source_point=source_point,
        receiver_point=receiver_point,
        source_name=args.source_name,
        receiver_name=args.receiver_name,
        occlusion_list=np.asarray(args.occlusion_list, dtype=np.float32),
        n1=args.n1,
    )

  if args.save_feature_npy:
    out_feature = Path(args.save_feature_npy)
    out_feature.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_feature, feature.astype(np.float32))
    print(f'Saved feature vector: {out_feature}')

  spectrogram = _load_spectrogram(args.spectrogram_path)
  global_condition = torch.from_numpy(feature).float()

  audio, sample_rate = predict(
      spectrogram=spectrogram,
      global_condition=global_condition,
      model_dir=args.model_dir,
      params=None,
      device=None,
      fast_sampling=args.fast,
  )
  torchaudio.save(args.output, audio.cpu(), sample_rate=sample_rate)
  print(f'Saved generated audio: {args.output}')
  print(f'Feature dimension: {feature.shape[0]}')


if __name__ == '__main__':
  parser = ArgumentParser(description='Run DiffWave inference from a feature .npy file or arbitrary source/receiver points')
  parser.add_argument('model_dir',
      help='directory containing a trained model (or full path to weights.pt file)')
  parser.add_argument('--output', '-o', default='output.wav',
      help='output file name')
  parser.add_argument('--spectrogram_path', '-s', default=None,
      help='optional spectrogram .npy path for locally conditioned models')
  parser.add_argument('--fast', '-f', action='store_true',
      help='fast sampling procedure')

  parser.add_argument('--feature_npy', default=None,
      help='path to a precomputed 1-D feature .npy file')

  parser.add_argument('--source_point', nargs=3, type=float, default=None,
      help='source point as x y z')
  parser.add_argument('--receiver_point', nargs=3, type=float, default=None,
      help='receiver point as x y z')
  parser.add_argument('--metadata_path', default=None,
      help='path to all_metadata.json used to build reference point sets')
  parser.add_argument('--mesh_path', default=None,
      help='path to room mesh file (for example empty_room1.stl)')
  parser.add_argument('--source_name', default='custom_source',
      help='key used for the custom source point in point mode')
  parser.add_argument('--receiver_name', default='custom_receiver',
      help='key used for the custom receiver point in point mode')
  parser.add_argument('--n1', type=int, default=1024,
      help='number of Fibonacci rays for feature extraction')
  parser.add_argument('--occlusion_list', nargs='+', type=float,
      default=_default_occlusion_list().tolist(),
      help='occlusion thresholds in meters')
  parser.add_argument('--save_feature_npy', default=None,
      help='optional output path to save the built feature vector')
  main(parser.parse_args())