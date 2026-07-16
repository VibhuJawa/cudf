# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings.cyruntime cimport cudaStream_t

from libc.stdint cimport uint64_t
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

from pylibcudf.io.types cimport SinkInfo, SourceInfo, TableWithMetadata
from pylibcudf.libcudf.io.lance cimport (
    lance_bulk_read_result,
    lance_bulk_reader_options,
    lance_row_location,
    lance_reader_options,
    lance_writer_options,
    read_lance_bulk as cpp_read_lance_bulk,
    read_lance as cpp_read_lance,
    write_lance as cpp_write_lance,
)
from pylibcudf.libcudf.io.types cimport compression_type, table_with_metadata
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.table cimport Table
from pylibcudf.utils cimport _get_memory_resource, _get_stream


__all__ = [
    "LanceBulkReaderOptions",
    "LanceBulkReaderOptionsBuilder",
    "LanceReaderOptions",
    "LanceReaderOptionsBuilder",
    "LanceWriterOptions",
    "LanceWriterOptionsBuilder",
    "read_lance",
    "read_lance_bulk",
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

    cpdef LanceWriterOptionsBuilder max_rows_per_miniblock(self, size_type rows):
        """
        Sets the maximum number of rows per Lance MiniBlock.

        Parameters
        ----------
        rows : int
            Maximum rows per MiniBlock.

        Returns
        -------
        LanceWriterOptionsBuilder
        """
        self.c_obj.max_rows_per_miniblock(rows)
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


cdef class LanceReaderOptions:
    """Settings for reading a Lance data file.

    For details, see :cpp:class:`cudf::io::experimental::lance_reader_options`.
    """

    @staticmethod
    def builder(SourceInfo source):
        """
        Create a builder for Lance reader options.

        Parameters
        ----------
        source : SourceInfo
            Source to read.

        Returns
        -------
        LanceReaderOptionsBuilder
        """
        cdef LanceReaderOptionsBuilder lance_builder = (
            LanceReaderOptionsBuilder.__new__(LanceReaderOptionsBuilder)
        )
        lance_builder.c_obj = lance_reader_options.builder(source.c_obj)
        lance_builder.source = source
        return lance_builder


cdef class LanceReaderOptionsBuilder:
    cpdef LanceReaderOptionsBuilder columns(self, list col_names):
        """
        Sets names of the columns to read.

        Parameters
        ----------
        col_names : list[str]
            List of column names.

        Returns
        -------
        LanceReaderOptionsBuilder
        """
        cdef vector[string] c_column_names
        c_column_names.reserve(len(col_names))
        for col in col_names:
            if not isinstance(col, str):
                raise TypeError("Column names must be strings!")
            c_column_names.push_back(col.encode())
        self.c_obj.columns(c_column_names)
        return self

    cpdef LanceReaderOptionsBuilder rows(self, list row_indices):
        """
        Sets zero-based row indices to read.

        Parameters
        ----------
        row_indices : list[int]
            Row ids to read, returned in the same order.

        Returns
        -------
        LanceReaderOptionsBuilder
        """
        cdef vector[size_type] c_rows
        c_rows.reserve(len(row_indices))
        for row in row_indices:
            c_rows.push_back(row)
        self.c_obj.rows(c_rows)
        return self

    cpdef LanceReaderOptions build(self):
        """Build Lance reader options."""
        cdef LanceReaderOptions lance_options = LanceReaderOptions.__new__(
            LanceReaderOptions
        )
        lance_options.c_obj = move(self.c_obj.build())
        lance_options.source = self.source
        return lance_options


cdef class LanceBulkReaderOptions:
    """Settings for reading sparse rows from multiple Lance data files."""

    @staticmethod
    def builder(SourceInfo source):
        """
        Create a builder for Lance bulk reader options.

        Parameters
        ----------
        source : SourceInfo
            Sources to read.

        Returns
        -------
        LanceBulkReaderOptionsBuilder
        """
        cdef LanceBulkReaderOptionsBuilder lance_builder = (
            LanceBulkReaderOptionsBuilder.__new__(LanceBulkReaderOptionsBuilder)
        )
        lance_builder.c_obj = lance_bulk_reader_options.builder(source.c_obj)
        lance_builder.source = source
        return lance_builder


cdef class LanceBulkReaderOptionsBuilder:
    cpdef LanceBulkReaderOptionsBuilder columns(self, list col_names):
        """
        Sets names of the columns to read.

        Parameters
        ----------
        col_names : list[str]
            List of column names.

        Returns
        -------
        LanceBulkReaderOptionsBuilder
        """
        cdef vector[string] c_column_names
        c_column_names.reserve(len(col_names))
        for col in col_names:
            if not isinstance(col, str):
                raise TypeError("Column names must be strings!")
            c_column_names.push_back(col.encode())
        self.c_obj.columns(c_column_names)
        return self

    cpdef LanceBulkReaderOptionsBuilder rows(self, list rows_per_source):
        """
        Sets zero-based row indices to read from each source.

        Parameters
        ----------
        rows_per_source : list[list[int]]
            Row ids to read, one row-id list per source.

        Returns
        -------
        LanceBulkReaderOptionsBuilder
        """
        cdef vector[vector[size_type]] c_rows_per_source
        cdef vector[size_type] c_rows
        c_rows_per_source.reserve(len(rows_per_source))
        for row_indices in rows_per_source:
            c_rows.clear()
            c_rows.reserve(len(row_indices))
            for row in row_indices:
                c_rows.push_back(row)
            c_rows_per_source.push_back(c_rows)
        self.c_obj.rows(c_rows_per_source)
        return self

    cpdef LanceBulkReaderOptionsBuilder row_locations(self, list locations):
        """
        Sets sparse source-row locations in global output order.

        Parameters
        ----------
        locations : list[tuple[int, int]]
            ``(source_index, row_index)`` pairs, one pair per output row.

        Returns
        -------
        LanceBulkReaderOptionsBuilder
        """
        cdef vector[lance_row_location] c_locations
        cdef lance_row_location location
        c_locations.reserve(len(locations))
        for source_index, row_index in locations:
            location.source_index = source_index
            location.row_index = row_index
            c_locations.push_back(location)
        self.c_obj.row_locations(c_locations)
        return self

    cpdef LanceBulkReaderOptionsBuilder read_coalesce_gap_bytes(self, uint64_t bytes):
        """
        Sets maximum byte gap to coalesce between selected file ranges.

        Parameters
        ----------
        bytes : int
            Maximum gap in bytes.

        Returns
        -------
        LanceBulkReaderOptionsBuilder
        """
        self.c_obj.read_coalesce_gap_bytes(bytes)
        return self

    cpdef LanceBulkReaderOptions build(self):
        """Build Lance bulk reader options."""
        cdef LanceBulkReaderOptions lance_options = LanceBulkReaderOptions.__new__(
            LanceBulkReaderOptions
        )
        lance_options.c_obj = move(self.c_obj.build())
        lance_options.source = self.source
        return lance_options


cpdef TableWithMetadata read_lance(
    LanceReaderOptions options, object stream = None, DeviceMemoryResource mr=None
):
    """
    Read from Lance format.

    Parameters
    ----------
    options : LanceReaderOptions
        Settings for controlling reading behavior.
    stream : Stream | None
        CUDA stream used for device memory operations and kernel launches.
    mr : DeviceMemoryResource, optional
        Device memory resource used to allocate the returned table's memory.
    """
    cdef table_with_metadata c_result
    cdef Stream s = _get_stream(stream)
    cdef cudaStream_t _cs = s.view().value()
    mr = _get_memory_resource(mr)
    with nogil:
        c_result = move(cpp_read_lance(options.c_obj, _cs, mr.get_mr()))

    return TableWithMetadata.from_libcudf(c_result, s, mr)


cpdef TableWithMetadata read_lance_bulk(
    LanceBulkReaderOptions options, object stream = None, DeviceMemoryResource mr=None
):
    """
    Read sparse rows from multiple Lance data files.

    Parameters
    ----------
    options : LanceBulkReaderOptions
        Settings for controlling bulk reading behavior.
    stream : Stream | None
        CUDA stream used for device memory operations and kernel launches.
    mr : DeviceMemoryResource, optional
        Device memory resource used to allocate the returned table's memory.
    """
    cdef lance_bulk_read_result c_result
    cdef Stream s = _get_stream(stream)
    cdef cudaStream_t _cs = s.view().value()
    mr = _get_memory_resource(mr)
    with nogil:
        c_result = move(cpp_read_lance_bulk(options.c_obj, _cs, mr.get_mr()))

    return TableWithMetadata.from_libcudf(c_result.data, s, mr)
