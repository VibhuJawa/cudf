# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import itertools
from typing import Literal

import pylibcudf as plc

from cudf.core.column import access_columns
from cudf.core.dataframe import DataFrame
from cudf.core.index import RangeIndex
from cudf.utils import ioutils


def _get_lance_compression(
    compression: Literal[False, None, "ZSTD", "NONE", "zstd", "none"],
) -> plc.io.types.CompressionType:
    if compression is None or compression is False:
        return plc.io.types.CompressionType.NONE

    normed_compression = compression.upper()
    if normed_compression == "NONE":
        return plc.io.types.CompressionType.NONE
    if normed_compression == "ZSTD":
        return plc.io.types.CompressionType.ZSTD

    raise ValueError(f"Unsupported `compression` type {compression}")


def _columns_and_names(table: DataFrame, index: bool | None):
    if index not in (True, False, None):
        raise TypeError("index must be a bool or None")

    include_index = index is True or (
        index is None and not isinstance(table.index, RangeIndex)
    )
    data_columns = [col for _, col in table._column_labels_and_values]
    data_names = [str(name) for name, _ in table._column_labels_and_values]

    if not include_index or table.index is None:
        return data_columns, data_names

    index_columns = list(table.index._columns)
    index_names = [
        str(ioutils._index_level_name(idx_name, level, table._column_names))
        for level, idx_name in enumerate(table._index.names)
    ]

    return list(itertools.chain(index_columns, data_columns)), (
        index_names + data_names
    )


def _validate_lance_columns(columns, names):
    if not columns:
        raise ValueError("Lance writer requires at least one column")

    for name, col in zip(names, columns, strict=True):
        if col.dtype.kind not in "iuf":
            raise NotImplementedError(
                "Lance writer currently supports only non-null fixed-width "
                f"integer and floating-point columns; column {name!r} has "
                f"dtype {col.dtype}"
            )
        if col.has_nulls():
            raise NotImplementedError(
                "Lance writer currently supports only non-null columns; "
                f"column {name!r} contains null values"
            )


def to_lance(
    df: DataFrame,
    path,
    compression: Literal[False, None, "ZSTD", "NONE", "zstd", "none"] = "ZSTD",
    max_rows_per_page: int | None = None,
    storage_options=None,
    index: bool | None = None,
) -> None:
    """Write a DataFrame to a Lance data file using libcudf.

    This experimental writer currently supports non-null top-level integer
    and floating-point columns. Page payloads are written by libcudf with
    optional nvCOMP ZSTD compression.
    """
    path_or_buf = ioutils.get_writer_filepath_or_buffer(
        path_or_data=path, mode="wb", storage_options=storage_options
    )

    if ioutils.is_fsspec_open_file(path_or_buf):
        with path_or_buf as file_obj:
            file_obj = ioutils.get_IOBase_writer(file_obj)
            _plc_write_lance(
                df, file_obj, compression, max_rows_per_page, index
            )
    else:
        _plc_write_lance(
            df, path_or_buf, compression, max_rows_per_page, index
        )


def _plc_write_lance(
    table: DataFrame,
    path_or_buf,
    compression: Literal[False, None, "ZSTD", "NONE", "zstd", "none"],
    max_rows_per_page: int | None,
    index: bool | None,
) -> None:
    columns, names = _columns_and_names(table, index)
    _validate_lance_columns(columns, names)

    if max_rows_per_page is not None:
        if (
            isinstance(max_rows_per_page, bool)
            or not isinstance(max_rows_per_page, int)
            or max_rows_per_page <= 0
        ):
            raise ValueError("max_rows_per_page must be a positive integer")

    with access_columns(*columns, mode="read", scope="internal"):
        plc_table = plc.Table([col.plc_column for col in columns])
        tbl_meta = plc.io.TableWithMetadata(
            plc_table, [(name, []) for name in names]
        )
        builder = (
            plc.io.experimental.LanceWriterOptions.builder(
                plc.io.SinkInfo([path_or_buf]), plc_table
            )
            .metadata(tbl_meta)
            .compression(_get_lance_compression(compression))
        )
        if max_rows_per_page is not None:
            builder.max_rows_per_page(max_rows_per_page)

        plc.io.experimental.write_lance(builder.build())
