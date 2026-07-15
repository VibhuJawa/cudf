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
from importlib.metadata import PackageNotFoundError, version
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
IMAGE_FIXED_COLUMNS = ["image_size_bytes", "width", "height"]
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


@dataclass
class SkippedMeasurement:
    name: str
    reason: str


def table_nbytes(table) -> int:
    if hasattr(table, "nbytes"):
        return int(table.nbytes)
    if hasattr(table, "memory_usage"):
        usage = table.memory_usage(deep=True)
        return int(usage.sum() if hasattr(usage, "sum") else usage)
    return 0


def table_num_rows(table) -> int:
    if hasattr(table, "num_rows"):
        return int(table.num_rows)
    return int(len(table))


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def rapids_package_version(name: str) -> str | None:
    return (
        package_version(name)
        or package_version(f"{name}-cu13")
        or package_version(f"{name}-cu12")
    )


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
    rows = table_num_rows(last_table)
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


def remove_existing_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def write_subset(table, path: Path) -> None:
    remove_existing_path(path)
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


def try_import_cudf(skipped: list[SkippedMeasurement]):
    try:
        import cudf  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - depends on RAPIDS runtime
        skipped.append(
            SkippedMeasurement(
                name="cudf_fixed_width_lane",
                reason=f"cuDF import failed: {type(exc).__name__}: {exc}",
            )
        )
        return None
    return cudf


def run_cudf_fixed_width_lane(
    arrow_table,
    output_file: Path,
    sparse_rows: int,
    iterations: int,
    seed: int,
    measurements: list[Measurement],
    skipped: list[SkippedMeasurement],
) -> None:
    cudf = try_import_cudf(skipped)
    if cudf is None:
        return

    missing_api = []
    if not hasattr(cudf.DataFrame, "to_lance"):
        missing_api.append("DataFrame.to_lance")
    if not hasattr(cudf, "read_lance"):
        missing_api.append("cudf.read_lance")
    if missing_api:
        skipped.append(
            SkippedMeasurement(
                name="cudf_fixed_width_lane",
                reason=(
                    "installed cuDF is missing the experimental Lance API: "
                    + ", ".join(missing_api)
                ),
            )
        )
        return

    try:
        dataframe = cudf.DataFrame.from_arrow(arrow_table)
        remove_existing_path(output_file)

        start = time.perf_counter()
        dataframe.to_lance(output_file, compression="ZSTD")
        seconds = time.perf_counter() - start
        bytes_written = output_file.stat().st_size
        rows = len(dataframe)
        measurements.append(
            Measurement(
                name="cudf_local_image_fixed_write",
                rows=rows,
                bytes=bytes_written,
                seconds=seconds,
                rows_per_second=rows / seconds if seconds else 0.0,
                mib_per_second=(bytes_written / (1024 * 1024)) / seconds
                if seconds
                else 0.0,
                iterations=1,
            )
        )

        measurement, _ = timed_table(
            "cudf_local_dense_image_fixed_read",
            iterations,
            lambda: cudf.read_lance(output_file),
        )
        measurements.append(measurement)

        row_ids = random_row_ids(rows, sparse_rows, seed)
        measurement, _ = timed_table(
            "cudf_local_sparse_image_fixed_read",
            iterations,
            lambda: cudf.read_lance(output_file, rows=row_ids),
        )
        measurements.append(measurement)
    except Exception as exc:  # pragma: no cover - depends on RAPIDS runtime
        skipped.append(
            SkippedMeasurement(
                name="cudf_fixed_width_lane",
                reason=f"cuDF Lance benchmark failed: {type(exc).__name__}: {exc}",
            )
        )


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
    parser.add_argument("--skip-blob-columns", action="store_true")
    parser.add_argument("--skip-text", action="store_true")
    parser.add_argument("--skip-cudf", action="store_true")
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
    image_fixed_subset = args.output_dir / "mint_images_fixed_subset.lance"
    cudf_image_fixed_file = args.output_dir / "mint_images_fixed_cudf.lance"
    text_subset = args.output_dir / "mint_text_subset.lance"
    measurements: list[Measurement] = []
    skipped: list[SkippedMeasurement] = []

    remote_image_table = None
    remote_image_fixed_table = None
    remote_text_table = None
    if not args.skip_remote:
        if not args.skip_blob_columns:
            measurement, remote_image_table = timed_table(
                "remote_dense_images_scan",
                args.iterations,
                lambda: scan_table(image_ds, IMAGE_COLUMNS, args.image_rows),
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

        measurement, remote_image_fixed_table = timed_table(
            "remote_dense_image_fixed_scan",
            args.iterations,
            lambda: scan_table(image_ds, IMAGE_FIXED_COLUMNS, args.image_rows),
        )
        measurements.append(measurement)

        remote_fixed_sparse_rows = random_row_ids(
            image_ds.count_rows(), args.sparse_rows, args.seed
        )
        measurement, _ = timed_table(
            "remote_sparse_image_fixed_take",
            args.iterations,
            lambda: take_rows(image_ds, remote_fixed_sparse_rows, IMAGE_FIXED_COLUMNS),
        )
        measurements.append(measurement)

        if not args.skip_text:
            measurement, remote_text_table = timed_table(
                "remote_dense_text_scan",
                args.iterations,
                lambda: scan_table(text_ds, TEXT_COLUMNS, args.text_rows),
            )
            measurements.append(measurement)

    if not args.skip_local_write:
        if remote_image_table is None and not args.skip_blob_columns:
            remote_image_table = scan_table(image_ds, IMAGE_COLUMNS, args.image_rows)
        if remote_image_fixed_table is None:
            remote_image_fixed_table = scan_table(
                image_ds, IMAGE_FIXED_COLUMNS, args.image_rows
            )
        if remote_text_table is None and not args.skip_text:
            remote_text_table = scan_table(text_ds, TEXT_COLUMNS, args.text_rows)
        if remote_image_table is not None:
            write_subset(remote_image_table, image_subset)
        write_subset(remote_image_fixed_table, image_fixed_subset)
        if remote_text_table is not None:
            write_subset(remote_text_table, text_subset)

    if not args.skip_blob_columns:
        local_image_ds = lance.dataset(image_subset)
        measurement, _ = timed_table(
            "local_dense_images_scan",
            args.iterations,
            lambda: scan_table(local_image_ds, IMAGE_COLUMNS, args.image_rows),
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

    local_image_fixed_ds = lance.dataset(image_fixed_subset)
    measurement, _ = timed_table(
        "local_dense_image_fixed_scan",
        args.iterations,
        lambda: scan_table(local_image_fixed_ds, IMAGE_FIXED_COLUMNS, args.image_rows),
    )
    measurements.append(measurement)

    local_fixed_sparse_rows = random_row_ids(
        local_image_fixed_ds.count_rows(), args.sparse_rows, args.seed
    )
    measurement, _ = timed_table(
        "local_sparse_image_fixed_take",
        args.iterations,
        lambda: take_rows(
            local_image_fixed_ds, local_fixed_sparse_rows, IMAGE_FIXED_COLUMNS
        ),
    )
    measurements.append(measurement)

    if not args.skip_text:
        local_text_ds = lance.dataset(text_subset)
        measurement, _ = timed_table(
            "local_dense_text_scan",
            args.iterations,
            lambda: scan_table(local_text_ds, TEXT_COLUMNS, args.text_rows),
        )
        measurements.append(measurement)

    if args.skip_cudf:
        skipped.append(SkippedMeasurement(name="cudf_fixed_width_lane", reason="disabled"))
    else:
        if remote_image_fixed_table is None:
            remote_image_fixed_table = scan_table(
                image_ds, IMAGE_FIXED_COLUMNS, args.image_rows
            )
        run_cudf_fixed_width_lane(
            remote_image_fixed_table,
            cudf_image_fixed_file,
            args.sparse_rows,
            args.iterations,
            args.seed,
            measurements,
            skipped,
        )

    payload = {
        "datasets": {
            "images": {"uri": IMAGE_URI, "version": IMAGE_VERSION},
            "text": {"uri": TEXT_URI, "version": TEXT_VERSION},
        },
        "local_subsets": {
            "images": str(image_subset),
            "image_fixed": str(image_fixed_subset),
            "cudf_image_fixed": str(cudf_image_fixed_file),
            "text": str(text_subset),
        },
        "args": {
            "image_rows": args.image_rows,
            "text_rows": args.text_rows,
            "sparse_rows": args.sparse_rows,
            "iterations": args.iterations,
            "seed": args.seed,
            "skip_blob_columns": args.skip_blob_columns,
            "skip_text": args.skip_text,
            "skip_cudf": args.skip_cudf,
            "skip_remote": args.skip_remote,
            "skip_local_write": args.skip_local_write,
        },
        "versions": {
            "lance": getattr(lance, "__version__", None)
            or package_version("pylance")
            or package_version("lance"),
            "pyarrow": package_version("pyarrow"),
            "cudf": rapids_package_version("cudf"),
            "pylibcudf": rapids_package_version("pylibcudf"),
            "rmm": rapids_package_version("rmm"),
        },
        "measurements": [asdict(item) for item in measurements],
        "skipped_measurements": [asdict(item) for item in skipped],
    }

    output = json.dumps(payload, indent=2, sort_keys=True)
    print(output)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(output + "\n")


if __name__ == "__main__":
    main()
