# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from typing import Self

from pylibcudf.io.types import CompressionType, SinkInfo, TableWithMetadata
from pylibcudf.table import Table
from pylibcudf.utils import CudaStreamLike

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
