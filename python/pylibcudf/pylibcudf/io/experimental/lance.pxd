# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from pylibcudf.io.types cimport SinkInfo, SourceInfo, TableWithMetadata
from pylibcudf.libcudf.io.lance cimport (
    lance_reader_options,
    lance_reader_options_builder,
    lance_writer_options,
    lance_writer_options_builder,
)
from pylibcudf.libcudf.io.types cimport compression_type
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.table cimport Table


cdef class LanceWriterOptions:
    cdef lance_writer_options c_obj
    cdef Table table
    cdef SinkInfo sink

cdef class LanceWriterOptionsBuilder:
    cdef lance_writer_options_builder c_obj
    cdef Table table
    cdef SinkInfo sink
    cpdef LanceWriterOptionsBuilder metadata(self, TableWithMetadata tbl_w_meta)
    cpdef LanceWriterOptionsBuilder compression(self, compression_type compression)
    cpdef LanceWriterOptionsBuilder max_rows_per_page(self, size_type rows)
    cpdef LanceWriterOptions build(self)

cpdef void write_lance(LanceWriterOptions options, object stream = *)

cdef class LanceReaderOptions:
    cdef lance_reader_options c_obj
    cdef SourceInfo source

cdef class LanceReaderOptionsBuilder:
    cdef lance_reader_options_builder c_obj
    cdef SourceInfo source
    cpdef LanceReaderOptionsBuilder columns(self, list col_names)
    cpdef LanceReaderOptionsBuilder rows(self, list row_indices)
    cpdef LanceReaderOptions build(self)

cpdef TableWithMetadata read_lance(
    LanceReaderOptions options, object stream = *, DeviceMemoryResource mr=*
)
