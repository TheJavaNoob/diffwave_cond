from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as AF

from diffwave.utils import (
    calculate_c50_error,
    calculate_edt_error,
    calculate_edt_relative_error,
    calculate_t60_absolute_error,
    calculate_t60_percentage,
    compute_psnr,
    compute_snr,
)


@dataclass
class FileMetrics:
    pair: str
    sr: int
    channels: int
    samples: int
    t60_pct: float
    t60_abs: float
    c50: float
    edt: float
    edt_pct: float
    snr: float
    psnr: float


def _discover_wavs(root: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.wav" if recursive else "*.wav"
    return sorted(p for p in root.glob(pattern) if p.is_file())


def _build_gt_index(files: list[Path]) -> dict[str, Path]:
    by_name: dict[str, Path] = {}
    duplicates: set[str] = set()
    for p in files:
        name = p.name
        if name in by_name:
            duplicates.add(name)
        else:
            by_name[name] = p
    for name in duplicates:
        by_name.pop(name, None)
    return by_name


def _match_channels(pred: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pred_ch, gt_ch = pred.shape[0], gt.shape[0]
    if pred_ch == gt_ch:
        return pred, gt

    if pred_ch == 1 and gt_ch > 1:
        return pred.repeat(gt_ch, 1), gt

    if gt_ch == 1 and pred_ch > 1:
        return pred, gt.repeat(pred_ch, 1)

    min_ch = min(pred_ch, gt_ch)
    return pred[:min_ch], gt[:min_ch]


def _match_length(pred: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = min(pred.shape[-1], gt.shape[-1])
    return pred[..., :n], gt[..., :n]


def _resample_if_needed(audio: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
    if src_sr == dst_sr:
        return audio
    return AF.resample(audio, src_sr, dst_sr)


def _compute_metrics(pred: torch.Tensor, gt: torch.Tensor, sr: int) -> dict[str, float]:
    # Metrics utilities are batch-first: [B, C, T]
    pred_b = pred.unsqueeze(0)
    gt_b = gt.unsqueeze(0)

    return {
        "t60_pct": calculate_t60_percentage(pred_b, gt_b, sr=sr).mean().item() * 100.0,
        "t60_abs": calculate_t60_absolute_error(pred_b, gt_b, sr=sr).mean().item(),
        "c50": calculate_c50_error(pred_b, gt_b, sr=sr).mean().item(),
        "edt": calculate_edt_error(pred_b, gt_b, sr=sr).mean().item(),
        "edt_pct": calculate_edt_relative_error(pred_b, gt_b, sr=sr).mean().item() * 100.0,
        "snr": compute_snr(pred_b, gt_b).mean().item(),
        "psnr": compute_psnr(pred_b, gt_b).mean().item(),
    }


def _print_report(results: list[FileMetrics], skipped: list[str]) -> None:
    if not results:
        print("No valid file pairs were processed.")
        if skipped:
            print("Skipped files:")
            for reason in skipped:
                print(f"  - {reason}")
        return

    headers = (
        "Pair",
        "T60 %",
        "T60_ABS",
        "C50",
        "EDT",
        "EDT %",
        "SNR",
        "PSNR",
    )
    row_fmt = "{:<45s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s}"

    print("\n====================  Batch WAV Comparison  ====================")
    print(row_fmt.format(*headers))
    print("-" * 105)
    for r in results:
        print(
            row_fmt.format(
                r.pair[:45],
                f"{r.t60_pct:6.2f}",
                f"{r.t60_abs:6.4f}",
                f"{r.c50:6.3f}",
                f"{r.edt:6.4f}",
                f"{r.edt_pct:6.2f}",
                f"{r.snr:6.3f}",
                f"{r.psnr:6.3f}",
            )
        )

    mean = {
        "t60_pct": sum(r.t60_pct for r in results) / len(results),
        "t60_abs": sum(r.t60_abs for r in results) / len(results),
        "c50": sum(r.c50 for r in results) / len(results),
        "edt": sum(r.edt for r in results) / len(results),
        "edt_pct": sum(r.edt_pct for r in results) / len(results),
        "snr": sum(r.snr for r in results) / len(results),
        "psnr": sum(r.psnr for r in results) / len(results),
    }

    print("-" * 105)
    print(
        row_fmt.format(
            "MEAN",
            f"{mean['t60_pct']:6.2f}",
            f"{mean['t60_abs']:6.4f}",
            f"{mean['c50']:6.3f}",
            f"{mean['edt']:6.4f}",
            f"{mean['edt_pct']:6.2f}",
            f"{mean['snr']:6.3f}",
            f"{mean['psnr']:6.3f}",
        )
    )
    print("================================================================")
    print(f"Processed pairs: {len(results)}")
    print(f"Skipped pairs:   {len(skipped)}")

    if skipped:
        print("\nSkipped details:")
        for reason in skipped:
            print(f"  - {reason}")


def _write_csv(path: Path, results: list[FileMetrics]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "pair",
                "sample_rate",
                "channels",
                "samples",
                "t60_pct",
                "t60_abs",
                "c50",
                "edt",
                "edt_pct",
                "snr",
                "psnr",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.pair,
                    r.sr,
                    r.channels,
                    r.samples,
                    r.t60_pct,
                    r.t60_abs,
                    r.c50,
                    r.edt,
                    r.edt_pct,
                    r.snr,
                    r.psnr,
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a batch of predicted WAV files with ground truth WAV files."
    )
    parser.add_argument("pred_dir", type=Path, help="Directory containing predicted WAV files")
    parser.add_argument("gt_dir", type=Path, help="Directory containing ground-truth WAV files")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search WAV files recursively in both directories",
    )
    parser.add_argument(
        "--match-mode",
        choices=("relative", "basename"),
        default="relative",
        help="How to pair files: by relative path or by basename",
    )
    parser.add_argument(
        "--resample-to",
        type=int,
        default=None,
        help="Optional target sample rate. If omitted, sample rates must match.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional output CSV path for per-file metrics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pred_dir = args.pred_dir.expanduser().resolve()
    gt_dir = args.gt_dir.expanduser().resolve()

    if not pred_dir.exists() or not pred_dir.is_dir():
        raise FileNotFoundError(f"Predicted directory does not exist: {pred_dir}")
    if not gt_dir.exists() or not gt_dir.is_dir():
        raise FileNotFoundError(f"Ground-truth directory does not exist: {gt_dir}")

    pred_files = _discover_wavs(pred_dir, args.recursive)
    gt_files = _discover_wavs(gt_dir, args.recursive)

    if not pred_files:
        raise RuntimeError(f"No WAV files found under predicted directory: {pred_dir}")
    if not gt_files:
        raise RuntimeError(f"No WAV files found under ground-truth directory: {gt_dir}")

    gt_by_name = _build_gt_index(gt_files) if args.match_mode == "basename" else {}

    results: list[FileMetrics] = []
    skipped: list[str] = []

    for pred_path in pred_files:
        if args.match_mode == "relative":
            rel = pred_path.relative_to(pred_dir)
            gt_path = gt_dir / rel
        else:
            gt_path = gt_by_name.get(pred_path.name)
            rel = pred_path.name

        if gt_path is None or not gt_path.exists():
            skipped.append(f"missing GT for {pred_path}")
            continue

        try:
            pred_audio, pred_sr = torchaudio.load(str(pred_path))
            gt_audio, gt_sr = torchaudio.load(str(gt_path))

            if args.resample_to is not None:
                sr = args.resample_to
                pred_audio = _resample_if_needed(pred_audio, pred_sr, sr)
                gt_audio = _resample_if_needed(gt_audio, gt_sr, sr)
            elif pred_sr != gt_sr:
                skipped.append(
                    f"sample-rate mismatch for {pred_path} vs {gt_path} ({pred_sr} != {gt_sr})"
                )
                continue
            else:
                sr = pred_sr

            pred_audio, gt_audio = _match_channels(pred_audio, gt_audio)
            pred_audio, gt_audio = _match_length(pred_audio, gt_audio)

            if pred_audio.shape[-1] == 0:
                skipped.append(f"empty audio after alignment for {pred_path} vs {gt_path}")
                continue

            metrics = _compute_metrics(pred_audio, gt_audio, sr)
            results.append(
                FileMetrics(
                    pair=str(rel),
                    sr=sr,
                    channels=int(pred_audio.shape[0]),
                    samples=int(pred_audio.shape[-1]),
                    t60_pct=metrics["t60_pct"],
                    t60_abs=metrics["t60_abs"],
                    c50=metrics["c50"],
                    edt=metrics["edt"],
                    edt_pct=metrics["edt_pct"],
                    snr=metrics["snr"],
                    psnr=metrics["psnr"],
                )
            )
        except Exception as e:
            skipped.append(f"error for {pred_path} vs {gt_path}: {e}")

    _print_report(results, skipped)

    if args.csv_out is not None:
        csv_path = args.csv_out.expanduser().resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(csv_path, results)
        print(f"\nSaved CSV report: {csv_path}")


if __name__ == "__main__":
    main()