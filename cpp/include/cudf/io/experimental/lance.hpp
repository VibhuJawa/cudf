/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/types.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstddef>
#include <optional>
#include <utility>

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
 * top-level fixed-width columns.  Page payloads use Lance MiniBlock layout so ZSTD compression can
 * be applied by cuDF's device compression path and decoded by Lance's existing block-compression
 * reader.
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

}  // namespace io::experimental
}  // namespace CUDF_EXPORT cudf
