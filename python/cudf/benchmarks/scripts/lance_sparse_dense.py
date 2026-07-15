# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark dense and sparse Lance reads on the MINT-1T handoff datasets.

This is intentionally a standalone benchmark harness rather than a pytest
benchmark.  It needs private storage options for the MINT-1T Lance datasets and
can also materialize a local subset for repeatable local reads.

The script never prints credentials.  It reads DataMover storage locations by
default and forwards the selected endpoint/credential fields to Lance.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import lance

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised by runtime setup
    raise SystemExit("PyYAML is required to read DataMover storage locations") from exc


IMAGE_URI = (
    "s3://mm-nemo-curator/lance_dbs/mint_1t_html_images/"
    "47f4e65f452f20ffca8b205a/stable_row_ids/dataset"
)
IMAGE_VERSION = 4

TEXT_URI = (
    "s3://mm-nemo-curator/lance_dbs/mint_1t_html_interleaved/"
    "46d97d7f5593fa78d55b81bb/drop_missing_images/dataset"
)
TEXT_VERSION = 3

IMAGE_COLUMNS = ["url", "image", "image_size_bytes", "width", "height"]
TEXT_COLUMNS = ["sample_id", "position", "modality", "text_content", "source_ref"]


@dataclass
class Measurement:
    name: str
    rows: int
    bytes: int
    seconds: float
    rows_per_second: float
    mib_per_second: float
    iterations: int


def table_nbytes(table) -> int:
    return int(getattr(table, "nbytes", 0))


def storage_options_from_datamover(path: Path, location: str) -> dict[str, str]:
    config = yaml.safe_load(path.read_text())
    try:
        entry = config[location]
        secrets = entry["secrets"]["local"]
    except KeyError as exc:
        raise SystemExit(f"Missing DataMover storage location: {location}") from exc

    options = {
        "aws_endpoint": entry["endpoint"],
        "region": entry["region"],
        "aws_access_key_id": secrets["access_key_id"],
        "aws_secret_access_key": secrets["secret_access_key"],
    }
    if secrets.get("session_token"):
        options["aws_session_token"] = secrets["session_token"]
    return options


def timed_table(
    name: str,
    iterations: int,
    fn: Callable[[], object],
) -> tuple[Measurement, object]:
    timings: list[float] = []
    last_table = None
    for _ in range(iterations):
        start = time.perf_counter()
        last_table = fn()
        timings.append(time.perf_counter() - start)

    assert last_table is not None
    seconds = statistics.median(timings)
    rows = int(last_table.num_rows)
    bytes_read = table_nbytes(last_table)
    mib = bytes_read / (1024 * 1024)
    return (
        Measurement(
            name=name,
            rows=rows,
            bytes=bytes_read,
            seconds=seconds,
            rows_per_second=rows / seconds if seconds else 0.0,
            mib_per_second=mib / seconds if seconds else 0.0,
            iterations=iterations,
        ),
        last_table,
    )


def scan_table(dataset, columns: list[str], rows: int):
    return dataset.scanner(columns=columns, limit=rows).to_table()


def take_rows(dataset, row_ids: list[int], columns: list[str]):
    return dataset.take(row_ids, columns=columns)


def write_subset(table, path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    lance.write_dataset(
        table,
        path,
        data_storage_version="2.2",
        enable_stable_row_ids=True,
    )


def random_row_ids(row_count: int, rows: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    if rows > row_count:
        raise SystemExit(f"Requested {rows} sparse rows from a {row_count}-row dataset")
    return rng.sample(range(row_count), rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--storage-config",
        type=Path,
        default=Path.home() / ".config/datamover/storage_locations",
    )
    parser.add_argument("--storage-location", default="pdx-multimodal")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "bench-data/lance_sparse_dense",
    )
    parser.add_argument("--image-rows", type=int, default=2048)
    parser.add_argument("--text-rows", type=int, default=200_000)
    parser.add_argument("--sparse-rows", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--skip-remote", action="store_true")
    parser.add_argument("--skip-local-write", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    storage_options = storage_options_from_datamover(
        args.storage_config, args.storage_location
    )
    image_ds = lance.dataset(
        IMAGE_URI, version=IMAGE_VERSION, storage_options=storage_options
    )
    text_ds = lance.dataset(TEXT_URI, version=TEXT_VERSION, storage_options=storage_options)

    image_subset = args.output_dir / "mint_images_subset.lance"
    text_subset = args.output_dir / "mint_text_subset.lance"
    measurements: list[Measurement] = []

    remote_image_table = None
    remote_text_table = None
    if not args.skip_remote:
        measurement, remote_image_table = timed_table(
            "remote_dense_images_scan",
            args.iterations,
            lambda: scan_table(image_ds, IMAGE_COLUMNS, args.image_rows),
        )
        measurements.append(measurement)

        measurement, remote_text_table = timed_table(
            "remote_dense_text_scan",
            args.iterations,
            lambda: scan_table(text_ds, TEXT_COLUMNS, args.text_rows),
        )
        measurements.append(measurement)

        remote_sparse_rows = random_row_ids(
            image_ds.count_rows(), args.sparse_rows, args.seed
        )
        measurement, _ = timed_table(
            "remote_sparse_images_take",
            args.iterations,
            lambda: take_rows(image_ds, remote_sparse_rows, IMAGE_COLUMNS),
        )
        measurements.append(measurement)

    if not args.skip_local_write:
        if remote_image_table is None:
            remote_image_table = scan_table(image_ds, IMAGE_COLUMNS, args.image_rows)
        if remote_text_table is None:
            remote_text_table = scan_table(text_ds, TEXT_COLUMNS, args.text_rows)
        write_subset(remote_image_table, image_subset)
        write_subset(remote_text_table, text_subset)

    local_image_ds = lance.dataset(image_subset)
    local_text_ds = lance.dataset(text_subset)

    measurement, _ = timed_table(
        "local_dense_images_scan",
        args.iterations,
        lambda: scan_table(local_image_ds, IMAGE_COLUMNS, args.image_rows),
    )
    measurements.append(measurement)

    measurement, _ = timed_table(
        "local_dense_text_scan",
        args.iterations,
        lambda: scan_table(local_text_ds, TEXT_COLUMNS, args.text_rows),
    )
    measurements.append(measurement)

    local_sparse_rows = random_row_ids(
        local_image_ds.count_rows(), args.sparse_rows, args.seed
    )
    measurement, _ = timed_table(
        "local_sparse_images_take",
        args.iterations,
        lambda: take_rows(local_image_ds, local_sparse_rows, IMAGE_COLUMNS),
    )
    measurements.append(measurement)

    payload = {
        "datasets": {
            "images": {"uri": IMAGE_URI, "version": IMAGE_VERSION},
            "text": {"uri": TEXT_URI, "version": TEXT_VERSION},
        },
        "local_subsets": {
            "images": str(image_subset),
            "text": str(text_subset),
        },
        "args": {
            "image_rows": args.image_rows,
            "text_rows": args.text_rows,
            "sparse_rows": args.sparse_rows,
            "iterations": args.iterations,
            "seed": args.seed,
        },
        "measurements": [asdict(item) for item in measurements],
    }

    output = json.dumps(payload, indent=2, sort_keys=True)
    print(output)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(output + "\n")


if __name__ == "__main__":
    main()
