# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from io import BytesIO
import itertools
import operator
from typing import Literal

import pylibcudf as plc

from cudf.core.column import access_columns
from cudf.core.dataframe import DataFrame
from cudf.core.dtypes import ListDtype
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

    image_columns = []
    for name, col in zip(names, columns, strict=True):
        is_image_column = (
            isinstance(col.dtype, ListDtype)
            and col.dtype.element_type.kind == "u"
            and col.dtype.element_type.itemsize == 1
        )
        if col.dtype.kind not in "iuf" and not is_image_column:
            raise NotImplementedError(
                "Lance writer currently supports non-null fixed-width integer "
                "and floating-point columns or list<uint8> image columns; "
                f"column {name!r} has dtype {col.dtype}"
            )
        if col.has_nulls():
            raise NotImplementedError(
                "Lance writer currently supports only non-null columns; "
                f"column {name!r} contains null values"
            )
        image_columns.append(is_image_column)

    if any(image_columns) and not all(image_columns):
        raise NotImplementedError(
            "Lance writer cannot mix fixed-width columns and list<uint8> image columns"
        )


def _normalize_lance_read_columns(columns):
    if columns is None:
        return None
    if isinstance(columns, str):
        return [columns]
    return [str(column) for column in columns]


def _normalize_lance_rows(rows):
    if rows is None:
        return None
    if isinstance(rows, (str, bytes)):
        raise TypeError("rows must be a sequence of integer row ids")

    try:
        iterator = iter(rows)
    except TypeError as exc:
        raise TypeError("rows must be a sequence of integer row ids") from exc

    row_ids = []
    for row in iterator:
        if isinstance(row, bool):
            raise TypeError("rows must contain integer row ids")
        try:
            row_id = operator.index(row)
        except TypeError as exc:
            raise TypeError("rows must contain integer row ids") from exc
        if row_id < 0:
            raise ValueError("rows must contain non-negative row ids")
        row_ids.append(row_id)
    return row_ids


def _normalize_lance_rows_per_source(rows_per_source):
    if rows_per_source is None:
        raise TypeError("rows_per_source is required")
    if isinstance(rows_per_source, (str, bytes)):
        raise TypeError("rows_per_source must be a sequence of row-id sequences")

    try:
        iterator = iter(rows_per_source)
    except TypeError as exc:
        raise TypeError(
            "rows_per_source must be a sequence of row-id sequences"
        ) from exc

    return [_normalize_lance_rows(rows) for rows in iterator]


def _normalize_lance_row_locations(row_locations):
    if row_locations is None:
        raise TypeError("row_locations is required")
    if isinstance(row_locations, (str, bytes)):
        raise TypeError("row_locations must be a sequence of (source, row) pairs")

    try:
        iterator = iter(row_locations)
    except TypeError as exc:
        raise TypeError(
            "row_locations must be a sequence of (source, row) pairs"
        ) from exc

    locations = []
    for location in iterator:
        if isinstance(location, (str, bytes)):
            raise TypeError("row_locations must contain (source, row) pairs")
        try:
            source_index, row_index = location
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "row_locations must contain (source, row) pairs"
            ) from exc
        if isinstance(source_index, bool) or isinstance(row_index, bool):
            raise TypeError("row_locations must contain integer source and row ids")
        try:
            source_index = operator.index(source_index)
            row_index = operator.index(row_index)
        except TypeError as exc:
            raise TypeError(
                "row_locations must contain integer source and row ids"
            ) from exc
        if source_index < 0 or row_index < 0:
            raise ValueError("row_locations must contain non-negative source and row ids")
        locations.append((source_index, row_index))
    return locations


def read_lance(
    filepath_or_buffer,
    columns=None,
    rows=None,
    storage_options=None,
    bytes_per_thread=None,
) -> DataFrame:
    """Read a Lance data file using libcudf.

    This experimental reader currently supports non-null top-level integer and
    floating-point columns in self-described Lance v2.2 data files. Projected
    reads may select supported fixed-width columns from mixed schemas; selecting
    unsupported columns still raises. If ``rows`` is supplied, libcudf reads only
    MiniBlock chunks containing those zero-based row ids and returns rows in the
    supplied order.
    """
    path_or_buf = ioutils.get_reader_filepath_or_buffer(
        path_or_data=filepath_or_buffer,
        iotypes=(BytesIO,),
        storage_options=storage_options,
        bytes_per_thread=bytes_per_thread,
    )
    path_or_buf = ioutils._select_single_source(path_or_buf, "read_lance")

    column_names = _normalize_lance_read_columns(columns)
    row_ids = _normalize_lance_rows(rows)
    builder = plc.io.experimental.LanceReaderOptions.builder(
        plc.io.SourceInfo([path_or_buf])
    )
    if column_names is not None:
        builder.columns(column_names)
    if row_ids is not None:
        builder.rows(row_ids)

    return DataFrame.from_pylibcudf(
        plc.io.experimental.read_lance(builder.build())
    )


def read_lance_bulk(
    filepath_or_buffer,
    rows_per_source=None,
    columns=None,
    storage_options=None,
    bytes_per_thread=None,
    read_coalesce_gap_bytes: int | None = None,
    row_locations=None,
) -> DataFrame:
    """Read sparse rows from multiple Lance data files using libcudf.

    This experimental reader batches a sparse lookup across many Lance files.
    Supply ``rows_per_source`` to emit rows in source order, or supply
    ``row_locations`` as ``(source_index, row_index)`` pairs to preserve an
    arbitrary global lookup order. It supports the fixed-width columns handled
    by :func:`read_lance` and the canonical non-null ``large_binary`` ``image``
    column as a ``list<uint8>`` cuDF column.
    """
    path_or_buf = ioutils.get_reader_filepath_or_buffer(
        path_or_data=filepath_or_buffer,
        iotypes=(BytesIO,),
        storage_options=storage_options,
        bytes_per_thread=bytes_per_thread,
    )
    if isinstance(path_or_buf, (str, bytes, BytesIO)):
        raise TypeError("read_lance_bulk requires a sequence of sources")
    sources = list(path_or_buf)
    if (rows_per_source is None) == (row_locations is None):
        raise TypeError("supply exactly one of rows_per_source or row_locations")
    if row_locations is None:
        rows = _normalize_lance_rows_per_source(rows_per_source)
        if len(rows) != len(sources):
            raise ValueError("rows_per_source must contain one row list per source")
    else:
        locations = _normalize_lance_row_locations(row_locations)
        if any(source_index >= len(sources) for source_index, _ in locations):
            raise ValueError("row_locations contains a source index outside the sources")

    column_names = _normalize_lance_read_columns(columns)
    builder = plc.io.experimental.LanceBulkReaderOptions.builder(
        plc.io.SourceInfo(sources)
    )
    if row_locations is None:
        builder.rows(rows)
    else:
        builder.row_locations(locations)
    if column_names is not None:
        builder.columns(column_names)
    if read_coalesce_gap_bytes is not None:
        if (
            isinstance(read_coalesce_gap_bytes, bool)
            or not isinstance(read_coalesce_gap_bytes, int)
            or read_coalesce_gap_bytes < 0
        ):
            raise ValueError("read_coalesce_gap_bytes must be a non-negative integer")
        builder.read_coalesce_gap_bytes(read_coalesce_gap_bytes)

    return DataFrame.from_pylibcudf(
        plc.io.experimental.read_lance_bulk(builder.build())
    )


def to_lance(
    df: DataFrame,
    path,
    compression: Literal[False, None, "ZSTD", "NONE", "zstd", "none"] = "ZSTD",
    max_rows_per_page: int | None = None,
    max_rows_per_miniblock: int | None = None,
    storage_options=None,
    index: bool | None = None,
) -> None:
    """Write a DataFrame to a Lance data file using libcudf.

    This experimental writer currently supports either non-null top-level
    integer/floating-point columns or non-null ``list<uint8>`` image columns.
    Fixed-width page payloads are written with optional nvCOMP ZSTD
    compression. Image columns are written as Lance ``large_binary`` pages
    and currently require ``compression="NONE"``.

    ``max_rows_per_miniblock`` may be set to a power-of-two value such as
    512 or 2048 to tune sparse row lookup read amplification. The default is
    1024 rows per MiniBlock; set 4096 to use Lance's default layout.
    """
    path_or_buf = ioutils.get_writer_filepath_or_buffer(
        path_or_data=path, mode="wb", storage_options=storage_options
    )

    if ioutils.is_fsspec_open_file(path_or_buf):
        with path_or_buf as file_obj:
            file_obj = ioutils.get_IOBase_writer(file_obj)
            _plc_write_lance(
                df,
                file_obj,
                compression,
                max_rows_per_page,
                max_rows_per_miniblock,
                index,
            )
    else:
        _plc_write_lance(
            df,
            path_or_buf,
            compression,
            max_rows_per_page,
            max_rows_per_miniblock,
            index,
        )


def _plc_write_lance(
    table: DataFrame,
    path_or_buf,
    compression: Literal[False, None, "ZSTD", "NONE", "zstd", "none"],
    max_rows_per_page: int | None,
    max_rows_per_miniblock: int | None,
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
    if max_rows_per_miniblock is not None:
        if (
            isinstance(max_rows_per_miniblock, bool)
            or not isinstance(max_rows_per_miniblock, int)
            or max_rows_per_miniblock < 2
            or max_rows_per_miniblock > 32768
            or (max_rows_per_miniblock & (max_rows_per_miniblock - 1)) != 0
        ):
            raise ValueError(
                "max_rows_per_miniblock must be a power-of-two integer "
                "between 2 and 32768"
            )

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
        if max_rows_per_miniblock is not None:
            builder.max_rows_per_miniblock(max_rows_per_miniblock)

        plc.io.experimental.write_lance(builder.build())
