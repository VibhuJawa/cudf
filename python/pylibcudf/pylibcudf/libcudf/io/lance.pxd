# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

cimport pylibcudf.libcudf.io.types as cudf_io_types
cimport pylibcudf.libcudf.table.table_view as cudf_table_view

from cuda.bindings.cyruntime cimport cudaStream_t
from libcpp.optional cimport optional
from libcpp.string cimport string
from libcpp.vector cimport vector
from rmm.librmm.memory_resource cimport device_async_resource_ref

from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.types cimport size_type


cdef extern from "cudf/io/experimental/lance.hpp" \
        namespace "cudf::io::experimental" nogil:

    cdef cppclass lance_writer_options:
        lance_writer_options() except +libcudf_exception_handler

        cudf_io_types.sink_info get_sink() except +libcudf_exception_handler
        cudf_table_view.table_view get_table() except +libcudf_exception_handler
        cudf_io_types.table_metadata get_metadata() except +libcudf_exception_handler
        cudf_io_types.compression_type get_compression() \
            except +libcudf_exception_handler
        optional[size_type] get_max_rows_per_page() except +libcudf_exception_handler

        @staticmethod
        lance_writer_options_builder builder(
            const cudf_io_types.sink_info& sink,
            const cudf_table_view.table_view& table
        ) except +libcudf_exception_handler

    cdef cppclass lance_reader_options:
        lance_reader_options() except +libcudf_exception_handler

        @staticmethod
        lance_reader_options_builder builder(
            cudf_io_types.source_info source
        ) except +libcudf_exception_handler

    cdef cppclass lance_writer_options_builder:
        lance_writer_options_builder() except +libcudf_exception_handler

        lance_writer_options_builder& metadata(
            cudf_io_types.table_metadata metadata
        ) except +libcudf_exception_handler
        lance_writer_options_builder& compression(
            cudf_io_types.compression_type compression
        ) except +libcudf_exception_handler
        lance_writer_options_builder& max_rows_per_page(
            size_type rows
        ) except +libcudf_exception_handler

        lance_writer_options build() except +libcudf_exception_handler

    cdef cppclass lance_reader_options_builder:
        lance_reader_options_builder() except +libcudf_exception_handler

        lance_reader_options_builder& columns(
            vector[string] columns
        ) except +libcudf_exception_handler
        lance_reader_options_builder& rows(
            vector[size_type] rows
        ) except +libcudf_exception_handler

        lance_reader_options build() except +libcudf_exception_handler

    cdef void write_lance(
        const lance_writer_options& options,
        cudaStream_t stream,
    ) except +libcudf_exception_handler

    cdef cudf_io_types.table_with_metadata read_lance(
        const lance_reader_options& options,
        cudaStream_t stream,
        device_async_resource_ref mr,
    ) except +libcudf_exception_handler
