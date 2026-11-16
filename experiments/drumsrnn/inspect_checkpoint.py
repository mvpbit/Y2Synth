#!/usr/bin/env python3
"""Inspect a local Magenta.js MusicRNN checkpoint without TensorFlow or NumPy.

Reads configuration and TF.js weights manifest, checks tensor shape metadata and
stored byte counts against shard sizes. This is NOT an inference or weight test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from math import prod
from pathlib import Path

DTYPE_SIZES = {
    "float32": 4,
    "float16": 2,
    "int32": 4,
    "uint8": 1,
    "uint16": 2,
    "uint32": 4,
    "bool": 1,
    "complex64": 8,
}


def load_json(path: Path):
    if not path.is_file():
        raise ValueError(f"Missing file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def describe(root: Path) -> None:
    config = load_json(root / "config.json")
    manifest = load_json(root / "weights_manifest.json")
    if not isinstance(config, dict) or config.get("type") != "MusicRNN":
        raise ValueError("Expected a MusicRNN config.json")
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("Expected a nonempty TF.js weights manifest array")

    print(f"Checkpoint: {root}")
    print("Config:", json.dumps(config, ensure_ascii=False, sort_keys=True))
    print()
    total_items = 0
    total_elements = 0
    total_bytes = 0
    seen_names = set()
    seen_paths = set()

    for group_no, group in enumerate(manifest, start=1):
        if not isinstance(group, dict):
            raise ValueError(f"Group {group_no} is not an object")
        specs = group.get("weights")
        paths = group.get("paths")
        if not isinstance(specs, list) or not specs:
            raise ValueError(f"Group {group_no}: no weight specs")
        if not isinstance(paths, list) or not paths:
            raise ValueError(f"Group {group_no}: no shard paths")

        print(f"Group {group_no}:")
        expected_bytes = 0
        for spec in specs:
            if not isinstance(spec, dict):
                raise ValueError("Malformed tensor specification")
            name = spec.get("name")
            shape = spec.get("shape")
            dtype = spec.get("dtype")
            if not isinstance(name, str) or not name or name in seen_names:
                raise ValueError(f"Invalid or duplicate tensor name: {name!r}")
            if not isinstance(shape, list) or any(
                not isinstance(d, int) or isinstance(d, bool) or d < 0 for d in shape
            ):
                raise ValueError(f"Invalid shape for tensor {name}")
            if not isinstance(dtype, str):
                raise ValueError(f"Invalid dtype for tensor {name}")
            quant = spec.get("quantization")
            stored_dtype = dtype
            if quant is not None:
                if not isinstance(quant, dict):
                    raise ValueError(f"Invalid quantization metadata for {name}")
                stored_dtype = quant.get("dtype")
            if stored_dtype not in DTYPE_SIZES:
                raise ValueError(
                    f"Unsupported stored dtype {stored_dtype!r} for {name}; "
                    "validation of this format needs an extension."
                )
            count = prod(shape)
            size_bytes = count * DTYPE_SIZES[stored_dtype]
            seen_names.add(name)
            total_items += 1
            total_elements += count
            expected_bytes += size_bytes
            q = f" [stored as {stored_dtype}]" if quant is not None else ""
            print(f"  {name}: shape={shape}, dtype={dtype}{q}, "
                  f"elements={count:,}, bytes={size_bytes:,}")

        actual_bytes = 0
        for filename in paths:
            if (
                not isinstance(filename, str)
                or not filename
                or Path(filename).name != filename
                or filename in seen_paths
            ):
                raise ValueError(f"Unsafe/duplicate shard filename: {filename!r}")
            seen_paths.add(filename)
            shard = root / filename
            if not shard.is_file():
                raise ValueError(f"Missing shard: {shard}")
            data_size = shard.stat().st_size
            actual_bytes += data_size
            digest = hashlib.file_digest(shard.open("rb"), "sha256").hexdigest()
            print(f"  shard {filename}: {data_size:,} bytes, SHA-256 {digest}")
        print(f"  group bytes: expected={expected_bytes:,}, actual={actual_bytes:,}")
        if expected_bytes != actual_bytes:
            raise ValueError(
                f"Group {group_no}: manifest byte count != downloaded shard bytes"
            )
        total_bytes += actual_bytes
        print()

    print(f"Total tensors: {total_items}")
    print(f"Total stored elements: {total_elements:,}")
    print(f"Total stored bytes: {total_bytes:,} ({total_bytes / 1024**2:.2f} MiB)")
    print("OK: manifest storage sizes match the local shard files.")
    print("NOTE: this checks metadata/lengths, not numerical correctness or provenance.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint_dir",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent / "models/drum_kit_rnn_tfjs",
    )
    args = parser.parse_args()
    try:
        describe(args.checkpoint_dir)
    except (ValueError, OSError) as exc:
        parser.exit(status=1, message=f"ERROR: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
