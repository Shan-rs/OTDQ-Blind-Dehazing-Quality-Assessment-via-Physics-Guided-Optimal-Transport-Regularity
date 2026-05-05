#!/usr/bin/env python3
"""Compute OTDQ and component scores for hazy/dehazed pairs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from otdq_eval.config import merged_otdq_config
from otdq_eval.io import match_image_pairs, read_manifest
from otdq_eval.scoring import score_records


EXPORT_COLUMNS = [
    "id",
    "hazy_path",
    "dehazed_path",
    "otdq",
    "s_vis",
    "s_str",
    "s_arti",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute OTDQ for one image pair, directories, or a CSV manifest.")
    mode = parser.add_argument_group("input")
    mode.add_argument("--hazy", type=Path, help="Path to one hazy RGB image.")
    mode.add_argument("--dehazed", type=Path, help="Path to one dehazed RGB image.")
    mode.add_argument("--hazy-dir", type=Path, help="Directory of hazy images.")
    mode.add_argument("--dehazed-dir", type=Path, help="Directory of dehazed images.")
    mode.add_argument("--manifest", type=Path, help="CSV with hazy/dehazed image paths.")
    mode.add_argument("--base-dir", type=Path, default=None, help="Base directory for relative manifest paths.")
    mode.add_argument("--id-col", default="id")
    mode.add_argument("--hazy-col", default="hazy_path")
    mode.add_argument("--dehazed-col", default="dehazed_path")

    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "otdq_default.json")
    parser.add_argument("--output", "-o", type=Path, default=None)
    parser.add_argument("--resize-method", default=None, choices=["NEAREST", "BILINEAR", "BICUBIC", "LANCZOS"])
    parser.add_argument("--min-size", type=int, default=None)
    parser.add_argument("--n-q", type=int, default=None)
    parser.add_argument("--var-thresh", type=float, default=None)
    parser.add_argument("--flat-alpha", type=float, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--texture-pct", type=float, default=None)
    parser.add_argument("--halo-pct", type=float, default=None)
    parser.add_argument("--patch-window", type=int, default=None)
    return parser


def collect_records(args: argparse.Namespace):
    if args.hazy and args.dehazed:
        return [{"id": "single", "hazy_path": str(args.hazy), "dehazed_path": str(args.dehazed)}]
    if args.hazy_dir and args.dehazed_dir:
        pairs = match_image_pairs(args.hazy_dir, args.dehazed_dir)
        if not pairs:
            raise RuntimeError(f"No matching pairs found: {args.hazy_dir} vs {args.dehazed_dir}")
        return pairs
    if args.manifest:
        return read_manifest(
            args.manifest,
            base_dir=args.base_dir,
            id_col=args.id_col,
            hazy_col=args.hazy_col,
            dehazed_col=args.dehazed_col,
        )
    raise SystemExit("Choose --hazy/--dehazed, --hazy-dir/--dehazed-dir, or --manifest.")


def main() -> None:
    args = build_parser().parse_args()
    config = merged_otdq_config(
        args.config,
        min_size=args.min_size,
        n_q=args.n_q,
        var_thresh=args.var_thresh,
        flat_alpha=args.flat_alpha,
        window_size=args.window_size,
        texture_pct=args.texture_pct,
        halo_pct=args.halo_pct,
        patch_window=args.patch_window,
        resize_method=args.resize_method,
    )
    records = collect_records(args)
    rows = score_records(records, resize_method=config["resize_method"], config=config)
    df = pd.DataFrame(rows)
    export_df = df[[col for col in EXPORT_COLUMNS if col in df.columns]]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        export_df.to_csv(args.output, index=False)
        print(f"[SAVE] {args.output}")
    else:
        print(export_df.to_csv(index=False))

    failures = df[df["status"] != "ok"] if "status" in df.columns else pd.DataFrame()
    if not failures.empty:
        raise SystemExit(f"{len(failures)} pair(s) failed. See the status/error columns.")


if __name__ == "__main__":
    main()
