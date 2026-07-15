# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from typing import Self

from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from pylibcudf.io.types import (
    CompressionType,
    SinkInfo,
    SourceInfo,
    TableWithMetadata,
)
from pylibcudf.table import Table
from pylibcudf.utils import CudaStreamLike

class LanceReaderOptions:
    @staticmethod
    def builder(source: SourceInfo) -> LanceReaderOptionsBuilder: ...

class LanceReaderOptionsBuilder:
    def columns(self, col_names: list[str]) -> Self: ...
    def rows(self, row_indices: list[int]) -> Self: ...
    def build(self) -> LanceReaderOptions: ...

class LanceWriterOptions:
    @staticmethod
    def builder(sink: SinkInfo, table: Table) -> LanceWriterOptionsBuilder: ...

class LanceWriterOptionsBuilder:
    def metadata(self, tbl_w_meta: TableWithMetadata) -> Self: ...
    def compression(self, compression: CompressionType) -> Self: ...
    def max_rows_per_page(self, rows: int) -> Self: ...
    def build(self) -> LanceWriterOptions: ...

def write_lance(
    options: LanceWriterOptions, stream: CudaStreamLike | None = None
) -> None: ...

def read_lance(
    options: LanceReaderOptions,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> TableWithMetadata: ...
