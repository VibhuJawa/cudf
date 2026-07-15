/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace CUDF_EXPORT cudf {
namespace io::experimental {

/**
 * @file
 * @brief Experimental Lance file writer APIs.
 */

/**
 * @addtogroup io_writers
 * @{
 */

class lance_writer_options_builder;

/**
 * @brief Settings for `write_lance()`.
 *
 * The initial experimental writer emits a self-described Lance v2.2 data file for non-null,
 * top-level fixed-width columns.  Page payloads use Lance v2.2 MiniBlock chunks so ZSTD
 * compression can be applied by cuDF's device compression path and decoded by Lance's existing
 * block-compression reader.
 */
class lance_writer_options {
  sink_info _sink;
  table_view _table;
  table_metadata _metadata;
  compression_type _compression = compression_type::ZSTD;
  std::optional<size_type> _max_rows_per_page{};

  friend lance_writer_options_builder;

  /**
   * @brief Constructor from sink and table.
   *
   * @param sink The sink used for writer output
   * @param table Table to be written to output
   */
  explicit lance_writer_options(sink_info sink, table_view table)
    : _sink(std::move(sink)), _table(std::move(table))
  {
  }

 public:
  /**
   * @brief Default constructor for interoperability.
   */
  lance_writer_options() = default;

  /**
   * @brief Create builder to create `lance_writer_options`.
   *
   * @param sink The sink used for writer output
   * @param table Table to be written to output
   *
   * @return Builder to build lance_writer_options
   */
  static lance_writer_options_builder builder(sink_info const& sink, table_view const& table);

  /**
   * @brief Returns sink used for writer output.
   *
   * @return Sink used for writer output
   */
  [[nodiscard]] sink_info const& get_sink() const noexcept { return _sink; }

  /**
   * @brief Returns table that will be written.
   *
   * @return Table to write
   */
  [[nodiscard]] table_view const& get_table() const noexcept { return _table; }

  /**
   * @brief Returns table metadata.
   *
   * Column names are taken from `metadata.schema_info` when provided.  Otherwise generated column
   * names are used.
   *
   * @return Table metadata
   */
  [[nodiscard]] table_metadata const& get_metadata() const noexcept { return _metadata; }

  /**
   * @brief Returns compression type.
   *
   * Supported values are `compression_type::NONE` and `compression_type::ZSTD`.
   *
   * @return Compression type
   */
  [[nodiscard]] compression_type get_compression() const noexcept { return _compression; }

  /**
   * @brief Returns optional maximum rows per Lance page.
   *
   * @return Optional maximum rows per page
   */
  [[nodiscard]] std::optional<size_type> get_max_rows_per_page() const noexcept
  {
    return _max_rows_per_page;
  }
};

/**
 * @brief Class to build `lance_writer_options`.
 */
class lance_writer_options_builder {
 public:
  /**
   * @brief Default constructor for interoperability.
   */
  lance_writer_options_builder() = default;

  /**
   * @brief Constructor from sink and table.
   *
   * @param sink The sink used for writer output
   * @param table Table to be written to output
   */
  explicit lance_writer_options_builder(sink_info const& sink, table_view const& table)
    : _options(sink, table)
  {
  }

  /**
   * @brief Set table metadata.
   *
   * @param metadata Table metadata
   * @return this for chaining
   */
  lance_writer_options_builder& metadata(table_metadata metadata) noexcept
  {
    _options._metadata = std::move(metadata);
    return *this;
  }

  /**
   * @brief Set compression type.
   *
   * @param compression Compression type
   * @return this for chaining
   */
  lance_writer_options_builder& compression(compression_type compression) noexcept
  {
    _options._compression = compression;
    return *this;
  }

  /**
   * @brief Set the maximum number of rows per Lance page.
   *
   * @param rows Maximum rows per page
   * @return this for chaining
   */
  lance_writer_options_builder& max_rows_per_page(size_type rows) noexcept
  {
    _options._max_rows_per_page = rows;
    return *this;
  }

  /**
   * @brief Build `lance_writer_options`.
   *
   * @return The constructed `lance_writer_options` object
   */
  [[nodiscard]] lance_writer_options build() const { return _options; }

 private:
  lance_writer_options _options;
};

/**
 * @brief Write a table as a self-described Lance data file.
 *
 * @param options Options specifying the sink, table, metadata, and compression
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void write_lance(lance_writer_options const& options,
                 rmm::cuda_stream_view stream = cudf::get_default_stream());

/** @} */  // end of io_writers group

/**
 * @addtogroup io_readers
 * @{
 */

class lance_reader_options_builder;

/**
 * @brief Settings for `read_lance()`.
 *
 * The initial experimental reader supports non-null fixed-width top-level columns in
 * self-described Lance v2.2 files, including projected reads from mixed schemas that also contain
 * unsupported column types. When row indices are supplied, the reader performs sparse page and
 * MiniBlock lookup and reads only chunks containing selected rows.
 */
class lance_reader_options {
  source_info _source;
  std::vector<std::string> _columns;
  std::vector<size_type> _rows;
  bool _has_row_selection = false;

  friend lance_reader_options_builder;

  /**
   * @brief Constructor from source info.
   *
   * @param source Source information used to read Lance file
   */
  explicit lance_reader_options(source_info source) : _source(std::move(source)) {}

 public:
  /**
   * @brief Default constructor for interoperability.
   */
  lance_reader_options() = default;

  /**
   * @brief Creates a builder to build `lance_reader_options`.
   *
   * @param source Source information used to read Lance file
   * @return Builder to build reader options
   */
  static lance_reader_options_builder builder(source_info source = source_info{});

  /**
   * @brief Returns source info.
   *
   * @return Source info
   */
  [[nodiscard]] source_info const& get_source() const noexcept { return _source; }

  /**
   * @brief Returns selected column names.
   *
   * Empty means all columns are selected.
   *
   * @return Selected column names
   */
  [[nodiscard]] std::vector<std::string> const& get_columns() const noexcept { return _columns; }

  /**
   * @brief Returns selected row indices.
   *
   * @return Selected row indices
   */
  [[nodiscard]] std::vector<size_type> const& get_rows() const noexcept { return _rows; }

  /**
   * @brief Returns whether sparse row selection is enabled.
   *
   * @return true if rows were supplied
   */
  [[nodiscard]] bool has_row_selection() const noexcept { return _has_row_selection; }
};

/**
 * @brief Class to build `lance_reader_options`.
 */
class lance_reader_options_builder {
 public:
  /**
   * @brief Default constructor for interoperability.
   */
  lance_reader_options_builder() = default;

  /**
   * @brief Constructor from source info.
   *
   * @param source Source information used to read Lance file
   */
  explicit lance_reader_options_builder(source_info source) : _options(std::move(source)) {}

  /**
   * @brief Select columns by name.
   *
   * @param columns Column names to read. Empty means all columns.
   * @return this for chaining
   */
  lance_reader_options_builder& columns(std::vector<std::string> columns)
  {
    _options._columns = std::move(columns);
    return *this;
  }

  /**
   * @brief Select top-level row indices to read.
   *
   * Rows are returned in the supplied order.
   *
   * @param rows Row indices to read
   * @return this for chaining
   */
  lance_reader_options_builder& rows(std::vector<size_type> rows)
  {
    _options._rows              = std::move(rows);
    _options._has_row_selection = true;
    return *this;
  }

  /**
   * @brief Build `lance_reader_options`.
   *
   * @return The constructed `lance_reader_options` object
   */
  [[nodiscard]] lance_reader_options build() const { return _options; }

 private:
  lance_reader_options _options;
};

/**
 * @brief Read a self-described Lance data file into a cuDF table.
 *
 * @param options Options specifying the source, selected columns, and sparse row selection
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate output columns
 * @return Table and associated metadata
 */
table_with_metadata read_lance(
  lance_reader_options const& options,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of io_readers

}  // namespace io::experimental
}  // namespace CUDF_EXPORT cudf
