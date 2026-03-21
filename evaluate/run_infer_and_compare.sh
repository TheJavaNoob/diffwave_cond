#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <model_or_weights.pt> <labels_dir> <gt_wavs_dir> <generated_wavs_dir> [options]"
  echo
  echo "Options:"
  echo "  --no-fast                  Disable fast sampling (default: fast enabled)"
  echo "  --match-mode <mode>        Pairing mode for compare_wavs: relative|basename (default: relative)"
  echo "  --resample-to <hz>         Resample both pred/gt before scoring"
  echo "  --csv-out <path>           Save per-file metrics to CSV"
  exit 1
fi

MODEL_PATH="$1"
LABELS_DIR="$2"
GT_WAVS_DIR="$3"
GEN_WAVS_DIR="$4"
shift 4

FAST_FLAG="--fast"
MATCH_MODE="relative"
RESAMPLE_TO=""
CSV_OUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-fast)
      FAST_FLAG=""
      shift
      ;;
    --match-mode)
      MATCH_MODE="$2"
      shift 2
      ;;
    --resample-to)
      RESAMPLE_TO="$2"
      shift 2
      ;;
    --csv-out)
      CSV_OUT="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ ! -e "$MODEL_PATH" ]]; then
  echo "Model path not found: $MODEL_PATH"
  exit 1
fi
if [[ ! -d "$LABELS_DIR" ]]; then
  echo "Labels directory not found: $LABELS_DIR"
  exit 1
fi
if [[ ! -d "$GT_WAVS_DIR" ]]; then
  echo "GT WAV directory not found: $GT_WAVS_DIR"
  exit 1
fi
if [[ "$MATCH_MODE" != "relative" && "$MATCH_MODE" != "basename" ]]; then
  echo "Invalid --match-mode: $MATCH_MODE (expected: relative or basename)"
  exit 1
fi

mkdir -p "$GEN_WAVS_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "[1/2] Running batch inference from labels..."
count=0
skipped_existing=0
while IFS= read -r -d '' label_file; do
  rel_path="${label_file#$LABELS_DIR/}"
  rel_no_ext="${rel_path%.npy}"
  out_wav="$GEN_WAVS_DIR/${rel_no_ext}.wav"
  out_dir="$(dirname "$out_wav")"
  mkdir -p "$out_dir"

  if [[ -s "$out_wav" ]]; then
    echo "  - skip existing: $out_wav"
    skipped_existing=$((skipped_existing + 1))
    continue
  fi

  echo "  - $(basename "$label_file") -> $out_wav"
  if [[ -n "$FAST_FLAG" ]]; then
    python -m diffwave.infer_points "$MODEL_PATH" --feature_npy "$label_file" --output "$out_wav" "$FAST_FLAG"
  else
    python -m diffwave.infer_points "$MODEL_PATH" --feature_npy "$label_file" --output "$out_wav"
  fi
  count=$((count + 1))
done < <(find "$LABELS_DIR" -type f -name "*.npy" -print0 | sort -z)

if [[ "$count" -eq 0 && "$skipped_existing" -eq 0 ]]; then
  echo "No .npy label files found under: $LABELS_DIR"
  exit 1
fi

echo "[2/2] Comparing generated WAVs vs GT WAVs..."
COMPARE_ARGS=(
  -m diffwave.compare_wavs
  "$GEN_WAVS_DIR"
  "$GT_WAVS_DIR"
  --recursive
  --match-mode "$MATCH_MODE"
)

if [[ -n "$RESAMPLE_TO" ]]; then
  COMPARE_ARGS+=(--resample-to "$RESAMPLE_TO")
fi
if [[ -n "$CSV_OUT" ]]; then
  COMPARE_ARGS+=(--csv-out "$CSV_OUT")
fi

python "${COMPARE_ARGS[@]}"

echo "Done. Generated files: $count | Skipped existing files: $skipped_existing"
