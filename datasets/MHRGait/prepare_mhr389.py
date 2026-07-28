#!/usr/bin/env python3
"""Convert raw SAM-3D-Body MHR dictionaries to OpenGait MHR389 arrays."""

import argparse
import multiprocessing as mp
import pickle
import sys
from functools import partial
from pathlib import Path

import numpy as np

try:
    import numpy.core.numeric as _np_numeric

    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.numeric", _np_numeric)
except ImportError:
    pass


FIELD_SPECS = (
    ("global_rot", 3),
    ("body_pose_params", 133),
    ("hand_pose_params", 108),
    ("scale_params", 28),
    ("shape_params", 45),
    ("expr_params", 72),
)


def _require_field(mhr, name, width):
    if name not in mhr:
        raise KeyError(f"MHR dictionary has no '{name}' field")
    array = np.asarray(mhr[name], dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(
            f"Expected '{name}' with shape [T, {width}], got {array.shape}"
        )
    return array


def concatenate_mhr_fields(mhr):
    """Concatenate MHR fields in the fixed public MHR389 order."""
    if not isinstance(mhr, dict):
        raise TypeError(f"Expected an MHR dictionary, got {type(mhr).__name__}")
    arrays = [_require_field(mhr, *spec) for spec in FIELD_SPECS]
    frame_count = arrays[0].shape[0]
    for (name, _), array in zip(FIELD_SPECS[1:], arrays[1:]):
        if array.shape[0] != frame_count:
            raise ValueError(
                "MHR frame-count mismatch: "
                f"global_rot has {frame_count}, {name} has {array.shape[0]}"
            )
    return np.concatenate(arrays, axis=1).astype(np.float32, copy=False)


def find_raw_mhr_files(input_root):
    input_root = Path(input_root)
    candidates = sorted(
        path
        for path in input_root.glob("*/*/*/*.pkl")
        if "mhr" in path.name.lower() and "mhr389" not in path.name.lower()
    )
    if not candidates:
        raise FileNotFoundError(
            f"No raw MHR pkl files found under {input_root}/<id>/<type>/<view>"
        )
    sequence_sources = {}
    for path in candidates:
        sequence_sources.setdefault(path.parent, []).append(path)
    duplicates = {
        sequence: paths
        for sequence, paths in sequence_sources.items()
        if len(paths) > 1
    }
    if duplicates:
        sequence, paths = next(iter(duplicates.items()))
        names = ", ".join(path.name for path in paths)
        raise ValueError(
            f"Multiple raw MHR files found for {sequence}: {names}"
        )
    return candidates


def convert_one(source, input_root, output_root, skip_existing=False):
    source = Path(source)
    input_root = Path(input_root)
    output_root = Path(output_root)
    relative = source.parent.relative_to(input_root)
    destination = output_root / relative / "00-mhr389.pkl"
    if skip_existing and destination.is_file():
        return "skipped"

    with source.open("rb") as file:
        raw = pickle.load(file)
    mhr389 = concatenate_mhr_fields(raw)
    if mhr389.shape[0] == 0:
        raise ValueError(f"Empty MHR sequence: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as file:
        pickle.dump(mhr389, file, protocol=pickle.HIGHEST_PROTOCOL)
    return "converted"


def convert_dataset(
    input_root,
    output_root,
    workers=8,
    skip_existing=False,
):
    sources = find_raw_mhr_files(input_root)
    worker = partial(
        convert_one,
        input_root=input_root,
        output_root=output_root,
        skip_existing=skip_existing,
    )
    with mp.Pool(processes=workers) as pool:
        statuses = list(pool.imap_unordered(worker, sources))
    return {
        "sequences": len(statuses),
        "converted": statuses.count("converted"),
        "skipped": statuses.count("skipped"),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    stats = convert_dataset(
        args.input_root,
        args.output_root,
        workers=args.workers,
        skip_existing=args.skip_existing,
    )
    print(
        "MHR389 conversion complete: "
        f"{stats['converted']} converted, {stats['skipped']} skipped."
    )


if __name__ == "__main__":
    main()
