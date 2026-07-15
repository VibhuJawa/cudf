/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/comp/compression.hpp"
#include "io/comp/nvcomp_adapter.hpp"
#include "io/utilities/hostdevice_vector.hpp"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/io/experimental/lance.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cudf::io::experimental {
namespace detail {

namespace {

constexpr std::size_t lance_buffer_alignment = 64;
constexpr std::uint8_t lance_pad_byte        = 72;
constexpr std::uint32_t default_rows_per_page = 64 * 1024;
constexpr std::array<std::uint8_t, 4> lance_magic{'L', 'A', 'N', 'C'};

enum class wire_type : std::uint8_t { varint = 0, fixed64 = 1, length_delimited = 2, fixed32 = 5 };

class proto_writer {
 public:
  [[nodiscard]] std::vector<std::uint8_t> const& buffer() const noexcept { return _buffer; }
  [[nodiscard]] std::vector<std::uint8_t> release() noexcept { return std::move(_buffer); }

  void uint_field(int field, std::uint64_t value)
  {
    key(field, wire_type::varint);
    varint(value);
  }

  void int32_field(int field, std::int32_t value)
  {
    key(field, wire_type::varint);
    varint(static_cast<std::uint64_t>(static_cast<std::int64_t>(value)));
  }

  void bool_field(int field, bool value)
  {
    key(field, wire_type::varint);
    varint(value ? 1 : 0);
  }

  void string_field(int field, std::string_view value)
  {
    key(field, wire_type::length_delimited);
    varint(value.size());
    bytes(value.data(), value.size());
  }

  void bytes_field(int field, std::vector<std::uint8_t> const& value)
  {
    key(field, wire_type::length_delimited);
    varint(value.size());
    bytes(value.data(), value.size());
  }

  void packed_uint64_field(int field, std::vector<std::uint64_t> const& values)
  {
    if (values.empty()) { return; }
    proto_writer packed;
    for (auto value : values) {
      packed.varint(value);
    }
    bytes_field(field, packed.buffer());
  }

  void message_field(int field, std::vector<std::uint8_t> const& message)
  {
    bytes_field(field, message);
  }

  void empty_message_field(int field)
  {
    key(field, wire_type::length_delimited);
    varint(0);
  }

  void fixed32(std::uint32_t value)
  {
    for (int i = 0; i < 4; ++i) {
      _buffer.push_back(static_cast<std::uint8_t>((value >> (i * 8)) & 0xff));
    }
  }

  void fixed64(std::uint64_t value)
  {
    for (int i = 0; i < 8; ++i) {
      _buffer.push_back(static_cast<std::uint8_t>((value >> (i * 8)) & 0xff));
    }
  }

 private:
  void key(int field, wire_type type)
  {
    varint((static_cast<std::uint64_t>(field) << 3) | static_cast<std::uint8_t>(type));
  }

  void varint(std::uint64_t value)
  {
    while (value > 0x7f) {
      _buffer.push_back(static_cast<std::uint8_t>((value & 0x7f) | 0x80));
      value >>= 7;
    }
    _buffer.push_back(static_cast<std::uint8_t>(value));
  }

  void bytes(void const* data, std::size_t size)
  {
    auto const* first = static_cast<std::uint8_t const*>(data);
    _buffer.insert(_buffer.end(), first, first + size);
  }

  std::vector<std::uint8_t> _buffer;
};

std::vector<std::uint8_t> make_any(std::string_view type_url,
                                   std::vector<std::uint8_t> const& value)
{
  proto_writer any;
  any.string_field(1, type_url);
  any.bytes_field(2, value);
  return any.release();
}

std::vector<std::uint8_t> make_flat_encoding(std::uint64_t bits_per_value)
{
  proto_writer flat;
  flat.uint_field(1, bits_per_value);

  proto_writer encoding;
  encoding.message_field(1, flat.buffer());
  return encoding.release();
}

std::vector<std::uint8_t> make_zstd_general_encoding(std::uint64_t bits_per_value)
{
  proto_writer buffer_compression;
  buffer_compression.uint_field(1, 2);  // COMPRESSION_ALGORITHM_ZSTD

  proto_writer general;
  general.message_field(1, buffer_compression.buffer());
  general.message_field(3, make_flat_encoding(bits_per_value));

  proto_writer encoding;
  encoding.message_field(10, general.buffer());
  return encoding.release();
}

std::vector<std::uint8_t> make_miniblock_page_layout(std::uint64_t bits_per_value,
                                                     std::uint64_t num_items,
                                                     compression_type compression)
{
  proto_writer miniblock;
  miniblock.message_field(3,
                          compression == compression_type::ZSTD
                            ? make_zstd_general_encoding(bits_per_value)
                            : make_flat_encoding(bits_per_value));
  miniblock.uint_field(7, 1);          // num_buffers: values only
  miniblock.uint_field(9, num_items);  // num_items
  miniblock.bool_field(10, true);      // v2.2 large miniblock metadata

  proto_writer page_layout;
  page_layout.message_field(1, miniblock.buffer());
  return page_layout.release();
}

std::vector<std::uint8_t> make_direct_encoding(std::vector<std::uint8_t> const& any_bytes)
{
  proto_writer direct;
  direct.bytes_field(1, any_bytes);

  proto_writer encoding;
  encoding.message_field(2, direct.buffer());
  return encoding.release();
}

std::vector<std::uint8_t> make_page_encoding(std::uint64_t bits_per_value,
                                             std::uint64_t num_items,
                                             compression_type compression)
{
  return make_direct_encoding(make_any("/lance.encodings21.PageLayout",
                                       make_miniblock_page_layout(
                                         bits_per_value, num_items, compression)));
}

std::vector<std::uint8_t> make_column_encoding()
{
  proto_writer column_encoding;
  column_encoding.empty_message_field(1);  // values = google.protobuf.Empty
  return make_direct_encoding(
    make_any("/lance.encodings.ColumnEncoding", column_encoding.release()));
}

std::string logical_type(data_type type)
{
  switch (type.id()) {
    case type_id::INT8: return "int8";
    case type_id::INT16: return "int16";
    case type_id::INT32: return "int32";
    case type_id::INT64: return "int64";
    case type_id::UINT8: return "uint8";
    case type_id::UINT16: return "uint16";
    case type_id::UINT32: return "uint32";
    case type_id::UINT64: return "uint64";
    case type_id::FLOAT32: return "float";
    case type_id::FLOAT64: return "double";
    default: CUDF_FAIL("Unsupported Lance writer type: " + cudf::type_to_name(type));
  }
}

bool is_supported_lance_type(data_type type)
{
  switch (type.id()) {
    case type_id::INT8:
    case type_id::INT16:
    case type_id::INT32:
    case type_id::INT64:
    case type_id::UINT8:
    case type_id::UINT16:
    case type_id::UINT32:
    case type_id::UINT64:
    case type_id::FLOAT32:
    case type_id::FLOAT64: return true;
    default: return false;
  }
}

std::string column_name(table_metadata const& metadata, size_type column_index)
{
  if (column_index < static_cast<size_type>(metadata.schema_info.size()) &&
      !metadata.schema_info[column_index].name.empty()) {
    return metadata.schema_info[column_index].name;
  }
  return "column_" + std::to_string(column_index);
}

std::vector<std::uint8_t> make_field(std::string const& name,
                                     std::int32_t id,
                                     data_type type,
                                     bool nullable)
{
  proto_writer field;
  field.string_field(2, name);
  field.int32_field(3, id);
  field.int32_field(4, -1);  // top-level field
  field.string_field(5, logical_type(type));
  if (nullable) { field.bool_field(6, true); }
  field.uint_field(7, 1);  // legacy PLAIN field encoding for compatibility
  return field.release();
}

std::vector<std::uint8_t> make_file_descriptor(table_view const& table,
                                               table_metadata const& metadata)
{
  proto_writer schema;
  for (size_type idx = 0; idx < table.num_columns(); ++idx) {
    auto const col = table.column(idx);
    schema.message_field(
      1, make_field(column_name(metadata, idx), idx, col.type(), col.nullable()));
  }

  proto_writer descriptor;
  descriptor.message_field(1, schema.buffer());
  descriptor.uint_field(2, static_cast<std::uint64_t>(table.num_rows()));
  return descriptor.release();
}

struct page_metadata {
  std::vector<std::uint64_t> buffer_offsets;
  std::vector<std::uint64_t> buffer_sizes;
  std::uint64_t length{};
  std::uint64_t priority{};
  std::vector<std::uint8_t> encoding;
};

struct column_metadata {
  std::vector<page_metadata> pages;
};

std::size_t pad_size(std::size_t size, std::size_t alignment = lance_buffer_alignment)
{
  auto const remainder = size % alignment;
  return remainder == 0 ? 0 : alignment - remainder;
}

void write_padding(data_sink* sink, std::size_t size)
{
  static constexpr std::array<std::uint8_t, lance_buffer_alignment> pad = [] {
    std::array<std::uint8_t, lance_buffer_alignment> data{};
    data.fill(lance_pad_byte);
    return data;
  }();

  while (size > 0) {
    auto const chunk = std::min(size, pad.size());
    sink->host_write(pad.data(), chunk);
    size -= chunk;
  }
}

void write_host_buffer(data_sink* sink, void const* data, std::size_t size)
{
  if (size > 0) { sink->host_write(data, size); }
}

void write_aligned_host_buffer(data_sink* sink, void const* data, std::size_t size)
{
  write_host_buffer(sink, data, size);
  write_padding(sink, pad_size(size));
}

void write_device_buffer(data_sink* sink,
                         void const* device_data,
                         std::size_t size,
                         rmm::cuda_stream_view stream)
{
  if (size == 0) { return; }

  if (sink->is_device_write_preferred(size)) {
    sink->device_write(device_data, size, stream);
  } else {
    auto host_buffer = cudf::detail::make_host_vector(
      device_span<std::uint8_t const>{static_cast<std::uint8_t const*>(device_data), size}, stream);
    sink->host_write(host_buffer.data(), size);
  }
}

void write_little_endian_u16(std::vector<std::uint8_t>& buffer, std::uint16_t value)
{
  buffer.push_back(static_cast<std::uint8_t>(value & 0xff));
  buffer.push_back(static_cast<std::uint8_t>((value >> 8) & 0xff));
}

void write_little_endian_u32(std::vector<std::uint8_t>& buffer, std::uint32_t value)
{
  for (int i = 0; i < 4; ++i) {
    buffer.push_back(static_cast<std::uint8_t>((value >> (i * 8)) & 0xff));
  }
}

std::vector<std::uint8_t> make_miniblock_metadata(std::uint64_t miniblock_buffer_size)
{
  CUDF_EXPECTS(miniblock_buffer_size > 0 && miniblock_buffer_size % 8 == 0,
               "Invalid Lance miniblock buffer size");
  auto const words = (miniblock_buffer_size / 8) - 1;
  CUDF_EXPECTS(words <= ((std::uint64_t{1} << 28) - 1),
               "Lance miniblock is too large to describe with v2.2 metadata");
  auto const metadata_word = static_cast<std::uint32_t>(words << 4);

  std::vector<std::uint8_t> metadata;
  metadata.reserve(sizeof(std::uint32_t));
  write_little_endian_u32(metadata, metadata_word);
  return metadata;
}

struct compressed_buffer {
  rmm::device_uvector<std::uint8_t> data;
  std::size_t bytes_written{};

  compressed_buffer(std::size_t max_size, rmm::cuda_stream_view stream)
    : data(max_size, stream)
  {
  }
};

compressed_buffer compress_zstd_device(std::uint8_t const* input,
                                       std::size_t input_size,
                                       rmm::cuda_stream_view stream)
{
  auto disabled = cudf::io::detail::nvcomp::is_compression_disabled(
    cudf::io::detail::nvcomp::compression_type::ZSTD);
  CUDF_EXPECTS(!disabled.has_value(), "nvCOMP ZSTD compression is disabled: " + disabled.value());

  auto const max_allowed = cudf::io::detail::nvcomp::compress_max_allowed_chunk_size(
    cudf::io::detail::nvcomp::compression_type::ZSTD);
  CUDF_EXPECTS(!max_allowed.has_value() || input_size <= *max_allowed,
               "Lance ZSTD chunk is larger than nvCOMP's maximum supported chunk size");

  auto const max_output = cudf::io::detail::nvcomp::compress_max_output_chunk_size(
    cudf::io::detail::nvcomp::compression_type::ZSTD, input_size);
  compressed_buffer compressed(max_output, stream);

  rmm::device_buffer aligned_input;
  auto const required_alignment = cudf::io::detail::nvcomp::compress_required_alignment(
    cudf::io::detail::nvcomp::compression_type::ZSTD);
  auto const input_addr = reinterpret_cast<std::uintptr_t>(input);
  auto const* compression_input = input;
  if (required_alignment > 1 && input_addr % required_alignment != 0) {
    aligned_input = rmm::device_buffer(input_size, stream);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      aligned_input.data(), input, input_size, cudaMemcpyDeviceToDevice, stream.value()));
    compression_input = static_cast<std::uint8_t const*>(aligned_input.data());
  }

  auto inputs = cudf::detail::hostdevice_vector<device_span<std::uint8_t const>>(1, stream);
  inputs[0]   = device_span<std::uint8_t const>{compression_input, input_size};
  inputs.host_to_device_async(stream);

  auto outputs = cudf::detail::hostdevice_vector<device_span<std::uint8_t>>(1, stream);
  outputs[0]   = device_span<std::uint8_t>{compressed.data.data(), compressed.data.size()};
  outputs.host_to_device_async(stream);

  auto results = cudf::detail::hostdevice_vector<cudf::io::detail::codec_exec_result>(1, stream);
  results[0]   = cudf::io::detail::codec_exec_result{0, cudf::io::detail::codec_status::FAILURE};
  results.host_to_device_async(stream);

  cudf::io::detail::nvcomp::batched_compress(
    cudf::io::detail::nvcomp::compression_type::ZSTD, inputs, outputs, results, stream);
  results.device_to_host(stream);

  CUDF_EXPECTS(results[0].status == cudf::io::detail::codec_status::SUCCESS,
               "nvCOMP ZSTD failed to compress a Lance page");
  compressed.bytes_written = results[0].bytes_written;
  return compressed;
}

page_metadata write_page(data_sink* sink,
                         column_view const& column,
                         size_type row_begin,
                         size_type num_rows,
                         compression_type compression,
                         rmm::cuda_stream_view stream)
{
  auto const type_width = cudf::size_of(column.type());
  auto const raw_size   = static_cast<std::size_t>(num_rows) * type_width;
  auto const* raw_begin = column.head<std::uint8_t>() +
                          (static_cast<std::size_t>(column.offset() + row_begin) * type_width);

  std::optional<compressed_buffer> compressed;
  auto const* payload     = raw_begin;
  std::size_t payload_len = raw_size;
  if (compression == compression_type::ZSTD && raw_size > 0) {
    compressed.emplace(compress_zstd_device(raw_begin, raw_size, stream));
    payload     = compressed->data.data();
    payload_len = compressed->bytes_written;
  }

  std::vector<std::uint8_t> miniblock_header;
  miniblock_header.reserve(sizeof(std::uint16_t) + sizeof(std::uint32_t));
  write_little_endian_u16(miniblock_header, 0);  // no rep/def levels
  CUDF_EXPECTS(payload_len <= std::numeric_limits<std::uint32_t>::max(),
               "Lance page payload is too large");
  write_little_endian_u32(miniblock_header, static_cast<std::uint32_t>(payload_len));

  auto const header_pad = pad_size(miniblock_header.size(), 8);
  auto const data_pad   = pad_size(payload_len, 8);
  auto const miniblock_buffer_size =
    static_cast<std::uint64_t>(miniblock_header.size() + header_pad + payload_len + data_pad);
  auto const metadata_buffer = make_miniblock_metadata(miniblock_buffer_size);

  page_metadata page;
  page.length   = num_rows;
  page.priority = row_begin;
  page.encoding =
    make_page_encoding(type_width * 8, static_cast<std::uint64_t>(num_rows), compression);

  page.buffer_offsets.push_back(sink->bytes_written());
  page.buffer_sizes.push_back(metadata_buffer.size());
  write_aligned_host_buffer(sink, metadata_buffer.data(), metadata_buffer.size());

  page.buffer_offsets.push_back(sink->bytes_written());
  page.buffer_sizes.push_back(miniblock_buffer_size);
  write_host_buffer(sink, miniblock_header.data(), miniblock_header.size());
  write_padding(sink, header_pad);
  write_device_buffer(sink, payload, payload_len, stream);
  write_padding(sink, data_pad);
  write_padding(sink, pad_size(miniblock_buffer_size));

  return page;
}

std::vector<std::uint8_t> serialize_page(page_metadata const& page)
{
  proto_writer out;
  out.packed_uint64_field(1, page.buffer_offsets);
  out.packed_uint64_field(2, page.buffer_sizes);
  out.uint_field(3, page.length);
  out.message_field(4, page.encoding);
  out.uint_field(5, page.priority);
  return out.release();
}

std::vector<std::uint8_t> serialize_column_metadata(column_metadata const& column)
{
  proto_writer out;
  out.message_field(1, make_column_encoding());
  for (auto const& page : column.pages) {
    out.message_field(2, serialize_page(page));
  }
  return out.release();
}

void write_u16_le(data_sink* sink, std::uint16_t value)
{
  std::array<std::uint8_t, 2> bytes{
    static_cast<std::uint8_t>(value & 0xff),
    static_cast<std::uint8_t>((value >> 8) & 0xff),
  };
  sink->host_write(bytes.data(), bytes.size());
}

void write_u32_le(data_sink* sink, std::uint32_t value)
{
  std::array<std::uint8_t, 4> bytes{};
  for (int i = 0; i < 4; ++i) {
    bytes[i] = static_cast<std::uint8_t>((value >> (i * 8)) & 0xff);
  }
  sink->host_write(bytes.data(), bytes.size());
}

void write_u64_le(data_sink* sink, std::uint64_t value)
{
  std::array<std::uint8_t, 8> bytes{};
  for (int i = 0; i < 8; ++i) {
    bytes[i] = static_cast<std::uint8_t>((value >> (i * 8)) & 0xff);
  }
  sink->host_write(bytes.data(), bytes.size());
}

std::vector<column_metadata> write_data_pages(data_sink* sink,
                                              lance_writer_options const& options,
                                              rmm::cuda_stream_view stream)
{
  auto const& table       = options.get_table();
  auto const compression  = options.get_compression();
  auto const page_rows_in = options.get_max_rows_per_page().value_or(default_rows_per_page);
  CUDF_EXPECTS(page_rows_in > 0, "max_rows_per_page must be greater than zero");

  std::vector<column_metadata> columns(table.num_columns());
  for (size_type col_idx = 0; col_idx < table.num_columns(); ++col_idx) {
    auto const column = table.column(col_idx);
    for (size_type row = 0; row < table.num_rows(); row += page_rows_in) {
      auto const rows = std::min(page_rows_in, table.num_rows() - row);
      columns[col_idx].pages.push_back(write_page(sink, column, row, rows, compression, stream));
    }
  }
  return columns;
}

void validate_options(lance_writer_options const& options)
{
  auto const& table = options.get_table();
  CUDF_EXPECTS(table.num_columns() > 0, "Lance writer requires at least one column");
  CUDF_EXPECTS(options.get_compression() == compression_type::NONE ||
                 options.get_compression() == compression_type::ZSTD,
               "Lance writer currently supports NONE and ZSTD compression");

  for (size_type idx = 0; idx < table.num_columns(); ++idx) {
    auto const column = table.column(idx);
    CUDF_EXPECTS(is_supported_lance_type(column.type()),
                 "Unsupported Lance writer type for column " + std::to_string(idx) + ": " +
                   cudf::type_to_name(column.type()));
    CUDF_EXPECTS(!column.has_nulls(),
                 "Lance writer currently supports only non-null fixed-width columns");
  }
}

}  // namespace

void write_lance(data_sink* sink, lance_writer_options const& options, rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();

  validate_options(options);

  auto const columns = write_data_pages(sink, options, stream);

  auto const file_descriptor = make_file_descriptor(options.get_table(), options.get_metadata());
  auto const file_descriptor_position = static_cast<std::uint64_t>(sink->bytes_written());
  write_host_buffer(sink, file_descriptor.data(), file_descriptor.size());
  auto const file_descriptor_length = static_cast<std::uint64_t>(file_descriptor.size());

  auto const column_metadata_start = static_cast<std::uint64_t>(sink->bytes_written());
  std::vector<std::pair<std::uint64_t, std::uint64_t>> column_metadata_positions;
  column_metadata_positions.reserve(columns.size());
  for (auto const& column : columns) {
    auto const encoded = serialize_column_metadata(column);
    auto const position = static_cast<std::uint64_t>(sink->bytes_written());
    write_host_buffer(sink, encoded.data(), encoded.size());
    column_metadata_positions.emplace_back(position, encoded.size());
  }

  auto const cmo_table_start = static_cast<std::uint64_t>(sink->bytes_written());
  for (auto const& [position, length] : column_metadata_positions) {
    write_u64_le(sink, position);
    write_u64_le(sink, length);
  }

  auto const gbo_table_start = static_cast<std::uint64_t>(sink->bytes_written());
  write_u64_le(sink, file_descriptor_position);
  write_u64_le(sink, file_descriptor_length);

  write_u64_le(sink, column_metadata_start);
  write_u64_le(sink, cmo_table_start);
  write_u64_le(sink, gbo_table_start);
  write_u32_le(sink, 1);  // schema/file descriptor global buffer
  write_u32_le(sink, static_cast<std::uint32_t>(columns.size()));
  write_u16_le(sink, 2);  // Lance file major version
  write_u16_le(sink, 2);  // Lance file minor version
  sink->host_write(lance_magic.data(), lance_magic.size());
  sink->flush();
}

}  // namespace detail

lance_writer_options_builder lance_writer_options::builder(sink_info const& sink,
                                                           table_view const& table)
{
  return lance_writer_options_builder(sink, table);
}

}  // namespace cudf::io::experimental
