# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest

import cudf
from cudf.testing import assert_eq


def _assert_lance_footer(data: bytes):
    assert len(data) > 40
    assert int.from_bytes(data[-8:-6], "little") == 2
    assert int.from_bytes(data[-6:-4], "little") == 2
    assert data[-4:] == b"LANC"


def test_to_lance_writer_footer(tmp_path):
    df = cudf.DataFrame(
        {
            "key": [1, 2, 3, 4, 5],
            "value": [0.5, 1.5, 2.5, 3.5, 4.5],
        }
    )

    path = tmp_path / "test.lance"
    df.to_lance(path, max_rows_per_page=2)

    _assert_lance_footer(path.read_bytes())
    assert_eq(df, cudf.read_lance(path))


def test_to_lance_uncompressed_writer_footer(tmp_path):
    df = cudf.DataFrame({"key": [1, 2, 3]})

    path = tmp_path / "test_uncompressed.lance"
    cudf.io.to_lance(df, path, compression="NONE")

    _assert_lance_footer(path.read_bytes())


def test_read_lance_sparse_rows_and_columns(tmp_path):
    df = cudf.DataFrame(
        {
            "key": [1, 2, 3, 4, 5, 6, 7, 8],
            "value": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
            "offset": [10, 20, 30, 40, 50, 60, 70, 80],
        }
    )

    path = tmp_path / "sparse.lance"
    df.to_lance(path, compression="NONE", max_rows_per_page=3)

    got = cudf.read_lance(path, columns=["value", "key"], rows=[4, 1, 4, 6])
    expect = cudf.DataFrame(
        {"value": [4.5, 1.5, 4.5, 6.5], "key": [5, 2, 5, 7]}
    )
    assert_eq(expect, got)


def test_read_lance_sparse_zstd_many_pages(tmp_path):
    rows_per_page = 12_288
    rows = []
    for page in range(4):
        page_begin = page * rows_per_page
        rows.extend([page_begin + 7, page_begin + 4096 + 11])
    df = cudf.DataFrame(
        {
            "key": range(rows_per_page * 4),
            "value": [row * 2 for row in range(rows_per_page * 4)],
            "offset": [row % 17 for row in range(rows_per_page * 4)],
        }
    )

    path = tmp_path / "sparse_zstd_many_pages.lance"
    df.to_lance(path, compression="ZSTD", max_rows_per_page=rows_per_page)

    got = cudf.read_lance(path, columns=["value", "key"], rows=rows)
    expect = df[["value", "key"]].iloc[rows].reset_index(drop=True)
    assert_eq(expect, got)


def test_read_lance_all_rows(tmp_path):
    df = cudf.DataFrame(
        {
            "key": [1, 2, 3, 4, 5, 6, 7, 8],
            "value": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
        }
    )

    path = tmp_path / "all_rows.lance"
    df.to_lance(path, compression="NONE", max_rows_per_page=3)

    got = cudf.read_lance(path)
    assert_eq(df, got)


def test_read_lance_empty_sparse_rows(tmp_path):
    df = cudf.DataFrame({"key": [1, 2, 3]})

    path = tmp_path / "empty_sparse.lance"
    df.to_lance(path, compression="NONE")

    got = cudf.read_lance(path, rows=[])
    expect = cudf.DataFrame({"key": cudf.Series([], dtype=df["key"].dtype)})
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data",
    [
        {"text": ["a", "b", "c"]},
        {"flag": [True, False, True]},
        {"nullable": [1, None, 3]},
    ],
)
def test_to_lance_rejects_unsupported_columns(tmp_path, data):
    df = cudf.DataFrame(data)

    with pytest.raises(NotImplementedError):
        df.to_lance(tmp_path / "unsupported.lance")


def test_to_lance_rejects_unsupported_compression(tmp_path):
    df = cudf.DataFrame({"key": [1, 2, 3]})

    with pytest.raises(ValueError, match="Unsupported `compression` type"):
        df.to_lance(tmp_path / "unsupported_compression.lance", compression="SNAPPY")


def test_to_lance_rejects_invalid_page_rows(tmp_path):
    df = cudf.DataFrame({"key": [1, 2, 3]})

    with pytest.raises(ValueError, match="max_rows_per_page"):
        df.to_lance(tmp_path / "invalid_page_rows.lance", max_rows_per_page=0)


def test_read_lance_rejects_invalid_rows(tmp_path):
    df = cudf.DataFrame({"key": [1, 2, 3]})
    path = tmp_path / "invalid_rows.lance"
    df.to_lance(path, compression="NONE")

    with pytest.raises(ValueError, match="non-negative"):
        cudf.read_lance(path, rows=[0, -1])
