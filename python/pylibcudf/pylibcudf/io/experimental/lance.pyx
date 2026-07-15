# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings.cyruntime cimport cudaStream_t

from rmm.pylibrmm.stream cimport Stream

from pylibcudf.io.types cimport SinkInfo, TableWithMetadata
from pylibcudf.libcudf.io.lance cimport (
    lance_writer_options,
    write_lance as cpp_write_lance,
)
from pylibcudf.libcudf.io.types cimport compression_type
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.table cimport Table
from pylibcudf.utils cimport _get_stream


__all__ = [
    "LanceWriterOptions",
    "LanceWriterOptionsBuilder",
    "write_lance",
]


cdef class LanceWriterOptions:
    """Settings for writing a table as a Lance data file.

    For details, see :cpp:class:`cudf::io::experimental::lance_writer_options`.
    """

    @staticmethod
    def builder(SinkInfo sink, Table table):
        """
        Create a builder for Lance writer options.

        Parameters
        ----------
        sink : SinkInfo
            The sink used for writer output.
        table : Table
            Table to write.

        Returns
        -------
        LanceWriterOptionsBuilder
        """
        cdef LanceWriterOptionsBuilder lance_builder = (
            LanceWriterOptionsBuilder.__new__(LanceWriterOptionsBuilder)
        )
        lance_builder.c_obj = lance_writer_options.builder(sink.c_obj, table.view())
        lance_builder.table = table
        lance_builder.sink = sink
        return lance_builder


cdef class LanceWriterOptionsBuilder:
    cpdef LanceWriterOptionsBuilder metadata(self, TableWithMetadata tbl_w_meta):
        """
        Sets associated table metadata.

        Parameters
        ----------
        tbl_w_meta : TableWithMetadata
            Metadata containing the column names to write into the Lance schema.

        Returns
        -------
        LanceWriterOptionsBuilder
        """
        self.c_obj.metadata(tbl_w_meta.metadata)
        return self

    cpdef LanceWriterOptionsBuilder compression(self, compression_type compression):
        """
        Sets the page compression type.

        Parameters
        ----------
        compression : CompressionType
            The compression type to use. The experimental Lance writer supports
            NONE and ZSTD.

        Returns
        -------
        LanceWriterOptionsBuilder
        """
        self.c_obj.compression(compression)
        return self

    cpdef LanceWriterOptionsBuilder max_rows_per_page(self, size_type rows):
        """
        Sets the maximum number of rows per Lance page.

        Parameters
        ----------
        rows : int
            Maximum rows per page.

        Returns
        -------
        LanceWriterOptionsBuilder
        """
        self.c_obj.max_rows_per_page(rows)
        return self

    cpdef LanceWriterOptions build(self):
        """Build Lance writer options."""
        cdef LanceWriterOptions lance_options = LanceWriterOptions.__new__(
            LanceWriterOptions
        )
        lance_options.c_obj = self.c_obj.build()
        lance_options.table = self.table
        lance_options.sink = self.sink
        return lance_options


cpdef void write_lance(LanceWriterOptions options, object stream = None):
    """
    Write to Lance format.

    Parameters
    ----------
    options : LanceWriterOptions
        Settings for controlling writing behavior.
    stream : Stream | None
        CUDA stream used for device memory operations and kernel launches.

    Returns
    -------
    None
    """
    cdef Stream s = _get_stream(stream)
    cdef cudaStream_t _cs = s.view().value()
    with nogil:
        cpp_write_lance(options.c_obj, _cs)
