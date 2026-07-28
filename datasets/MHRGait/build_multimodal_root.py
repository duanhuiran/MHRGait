#!/usr/bin/env python3
"""Build an OpenGait silhouette/MHR389 root with relative symbolic links."""

import argparse
import os
import pickle
from pathlib import Path


def discover_sequences(root):
    """Map each <id>/<type>/<view> leaf to its only pkl file."""
    root = Path(root)
    sequences = {}
    for file_path in sorted(root.glob("*/*/*/*.pkl")):
        key = file_path.parent.relative_to(root)
        if key in sequences:
            raise ValueError(
                f"Expected one pkl per source sequence, found multiple in "
                f"{file_path.parent}"
            )
        sequences[key] = file_path
    if not sequences:
        raise FileNotFoundError(f"No OpenGait sequences found under {root}")
    return sequences


def _sequence_length(path):
    with path.open("rb") as file:
        return len(pickle.load(file))


def _relative_symlink(source, destination):
    relative_target = os.path.relpath(source, start=destination.parent)
    destination.symlink_to(relative_target)


def build_multimodal_root(silhouette_root, mhr_root, output_root):
    """Validate and pair two aligned OpenGait modality roots."""
    silhouette_root = Path(silhouette_root).resolve()
    mhr_root = Path(mhr_root).resolve()
    output_root = Path(output_root).resolve()
    silhouettes = discover_sequences(silhouette_root)
    mhrs = discover_sequences(mhr_root)

    sil_keys = set(silhouettes)
    mhr_keys = set(mhrs)
    if sil_keys != mhr_keys:
        missing_mhr = sorted(sil_keys - mhr_keys)
        missing_silhouette = sorted(mhr_keys - sil_keys)
        raise ValueError(
            "Modality sequence sets differ. "
            f"Missing MHR: {missing_mhr[:5]}; "
            f"missing silhouette: {missing_silhouette[:5]}"
        )
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output root must be absent or empty: {output_root}"
        )

    lengths = {}
    for key in sorted(sil_keys):
        sil_length = _sequence_length(silhouettes[key])
        mhr_length = _sequence_length(mhrs[key])
        if sil_length == 0:
            raise ValueError(f"Empty sequence: {key}")
        if sil_length != mhr_length:
            raise ValueError(
                f"Frame-count mismatch for {key}: "
                f"silhouette={sil_length}, MHR={mhr_length}"
            )
        lengths[key] = sil_length

    output_root.mkdir(parents=True, exist_ok=True)
    for key in sorted(lengths):
        destination = output_root / key
        destination.mkdir(parents=True)
        _relative_symlink(
            silhouettes[key], destination / "00-silhouette.pkl"
        )
        _relative_symlink(mhrs[key], destination / "01-mhr389.pkl")
    return len(lengths)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--silhouette-root", required=True, type=Path)
    parser.add_argument("--mhr-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    count = build_multimodal_root(
        args.silhouette_root,
        args.mhr_root,
        args.output_root,
    )
    print(f"Created {count} paired silhouette/MHR sequences.")


if __name__ == "__main__":
    main()
