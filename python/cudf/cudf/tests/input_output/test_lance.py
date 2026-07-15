# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest

import cudf


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


def test_to_lance_uncompressed_writer_footer(tmp_path):
    df = cudf.DataFrame({"key": [1, 2, 3]})

    path = tmp_path / "test_uncompressed.lance"
    cudf.io.to_lance(df, path, compression="NONE")

    _assert_lance_footer(path.read_bytes())


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
