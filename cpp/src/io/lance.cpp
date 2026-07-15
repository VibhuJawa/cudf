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
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/lance.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cudf::io::experimental {
namespace detail {

void copy_sparse_fixed_width(std::uint8_t const* source,
                             std::uint8_t* target,
                             size_type const* source_rows,
                             size_type const* target_rows,
                             size_type num_rows,
                             std::size_t type_width,
                             rmm::cuda_stream_view stream);

struct lance_pack_chunk {
  std::uint8_t const* payload{};
  std::size_t payload_size{};
  std::size_t output_offset{};
};

struct lance_miniblock_header {
  std::uint32_t num_levels{};
  std::uint32_t payload_size{};
};

struct lance_sparse_copy_chunk {
  std::uint8_t const* source{};
  std::uint8_t* target{};
  size_type const* source_rows{};
  size_type const* target_rows{};
  size_type num_rows{};
  std::size_t type_width{};
};

struct lance_sparse_copy_row {
  std::uint8_t const* source{};
  std::uint8_t* target{};
  size_type source_row{};
  size_type target_row{};
  std::size_t type_width{};
};

void pack_lance_miniblocks(lance_pack_chunk const* chunks,
                           std::size_t num_chunks,
                           std::uint8_t* output,
                           rmm::cuda_stream_view stream);

void decode_lance_miniblock_headers(std::uint8_t const* const* headers,
                                    lance_miniblock_header* decoded,
                                    std::size_t num_headers,
                                    rmm::cuda_stream_view stream);

void copy_sparse_fixed_width_batch(lance_sparse_copy_chunk const* chunks,
                                   std::size_t num_chunks,
                                   unsigned int blocks_per_chunk,
                                   rmm::cuda_stream_view stream);

void copy_sparse_fixed_width_single_row_batch(lance_sparse_copy_row const* rows,
                                              std::size_t num_rows,
                                              rmm::cuda_stream_view stream);

namespace {

constexpr std::size_t lance_buffer_alignment = 64;
constexpr std::uint8_t lance_pad_byte        = 72;
constexpr std::uint32_t default_rows_per_page = 64 * 1024;
constexpr std::uint8_t default_rows_per_page_log = 16;
constexpr std::uint32_t default_rows_per_miniblock = 4096;
constexpr std::uint8_t default_rows_per_miniblock_log = 12;
constexpr std::uint32_t writer_default_rows_per_miniblock = 1024;
constexpr std::uint32_t max_rows_per_miniblock = std::uint32_t{1} << 15;
constexpr std::size_t sparse_header_batch_min_reads = 8;
constexpr std::size_t sparse_multi_column_header_batch_min_reads = 2;
constexpr std::size_t sparse_copy_batch_min_chunks = 2;
constexpr std::size_t sparse_max_coalesced_miniblock_reads = 4;
constexpr std::size_t sparse_page_selected_span_min_chunks = 4;
constexpr std::size_t sparse_stack_miniblock_count = 64;
constexpr std::size_t sparse_page_span_min_pages = 16;
constexpr std::size_t sparse_page_span_min_chunks_per_page = 2;
constexpr std::size_t sparse_dense_min_rows_per_page = 4;
constexpr std::uint64_t sparse_page_span_max_bytes = std::uint64_t{64} << 20;
constexpr std::uint64_t sparse_page_span_max_overread_ratio = 2;
constexpr std::uint64_t sparse_page_selected_span_max_overread_ratio = 4;
constexpr std::size_t sparse_page_dense_selected_span_min_chunks_per_page = 8;
constexpr std::uint64_t sparse_page_dense_selected_span_max_overread_ratio = 8;
constexpr std::size_t sparse_uniform_miniblock_lookup_min_rows = 8;
constexpr std::size_t sparse_host_staging_min_reads = 32;
constexpr std::uint64_t sparse_host_staging_max_read_bytes = std::uint64_t{256} << 10;
constexpr std::uint64_t sparse_host_staging_max_total_bytes = std::uint64_t{64} << 20;
constexpr std::size_t miniblock_alignment = 8;
constexpr std::array<std::uint8_t, 4> lance_magic{'L', 'A', 'N', 'C'};

enum class wire_type : std::uint8_t { varint = 0, fixed64 = 1, length_delimited = 2, fixed32 = 5 };

std::size_t count_selected_miniblock_read_ranges(std::vector<bool> const& selected_chunks,
                                                 bool coalesce_adjacent)
{
  std::size_t range_count = 0;
  for (std::size_t idx = 0; idx < selected_chunks.size();) {
    if (!selected_chunks[idx]) {
      ++idx;
      continue;
    }
    ++range_count;
    auto range_chunks = std::size_t{1};
    ++idx;
    while (coalesce_adjacent && idx < selected_chunks.size() &&
           range_chunks < sparse_max_coalesced_miniblock_reads && selected_chunks[idx]) {
      ++range_chunks;
      ++idx;
    }
  }
  return range_count;
}

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

std::optional<data_type> data_type_from_lance_logical_type(std::string const& logical_type)
{
  if (logical_type == "int8") { return data_type{type_id::INT8}; }
  if (logical_type == "int16") { return data_type{type_id::INT16}; }
  if (logical_type == "int32") { return data_type{type_id::INT32}; }
  if (logical_type == "int64") { return data_type{type_id::INT64}; }
  if (logical_type == "uint8") { return data_type{type_id::UINT8}; }
  if (logical_type == "uint16") { return data_type{type_id::UINT16}; }
  if (logical_type == "uint32") { return data_type{type_id::UINT32}; }
  if (logical_type == "uint64") { return data_type{type_id::UINT64}; }
  if (logical_type == "float") { return data_type{type_id::FLOAT32}; }
  if (logical_type == "double") { return data_type{type_id::FLOAT64}; }
  return std::nullopt;
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

class proto_reader {
 public:
  proto_reader(std::uint8_t const* data, std::size_t size) : _data(data), _size(size) {}

  explicit proto_reader(std::vector<std::uint8_t> const& data)
    : proto_reader(data.data(), data.size())
  {
  }

  [[nodiscard]] bool eof() const noexcept { return _position == _size; }

  bool next(int& field, wire_type& type)
  {
    if (eof()) { return false; }
    auto const key = read_varint();
    field          = static_cast<int>(key >> 3);
    type           = static_cast<wire_type>(key & 0x7);
    return true;
  }

  std::uint64_t read_varint()
  {
    std::uint64_t value = 0;
    int shift           = 0;
    while (_position < _size) {
      auto const byte = _data[_position++];
      value |= static_cast<std::uint64_t>(byte & 0x7f) << shift;
      if ((byte & 0x80) == 0) { return value; }
      shift += 7;
      CUDF_EXPECTS(shift < 64, "Invalid Lance protobuf varint");
    }
    CUDF_FAIL("Unexpected end of Lance protobuf varint");
  }

  proto_reader read_message()
  {
    auto const size = read_varint();
    CUDF_EXPECTS(size <= _size - _position, "Invalid Lance protobuf message length");
    proto_reader child(_data + _position, static_cast<std::size_t>(size));
    _position += static_cast<std::size_t>(size);
    return child;
  }

  std::vector<std::uint8_t> read_bytes()
  {
    auto const size = read_varint();
    CUDF_EXPECTS(size <= _size - _position, "Invalid Lance protobuf bytes length");
    auto const begin = _data + _position;
    _position += static_cast<std::size_t>(size);
    return std::vector<std::uint8_t>(begin, begin + static_cast<std::size_t>(size));
  }

  std::string read_string()
  {
    auto bytes = read_bytes();
    return {reinterpret_cast<char const*>(bytes.data()), bytes.size()};
  }

  void skip(wire_type type)
  {
    switch (type) {
      case wire_type::varint: read_varint(); break;
      case wire_type::fixed64:
        CUDF_EXPECTS(_size - _position >= sizeof(std::uint64_t), "Invalid Lance protobuf fixed64");
        _position += sizeof(std::uint64_t);
        break;
      case wire_type::length_delimited: {
        auto const size = read_varint();
        CUDF_EXPECTS(size <= _size - _position, "Invalid Lance protobuf skipped field length");
        _position += static_cast<std::size_t>(size);
        break;
      }
      case wire_type::fixed32:
        CUDF_EXPECTS(_size - _position >= sizeof(std::uint32_t), "Invalid Lance protobuf fixed32");
        _position += sizeof(std::uint32_t);
        break;
      default: CUDF_FAIL("Unsupported Lance protobuf wire type");
    }
  }

 private:
  std::uint8_t const* _data{};
  std::size_t _size{};
  std::size_t _position{};
};

struct lance_field_info {
  std::string name;
  std::optional<data_type> type;
  bool nullable{};
  std::string logical_type;
  std::string unsupported_reason;

  [[nodiscard]] bool is_supported() const noexcept
  {
    return type.has_value() && unsupported_reason.empty();
  }
};

struct lance_page_layout {
  compression_type compression = compression_type::NONE;
  std::uint64_t bits_per_value{};
  std::uint64_t num_items{};
  std::uint64_t num_buffers{};
  bool has_large_chunk{};
};

struct lance_page_info {
  std::vector<std::uint64_t> buffer_offsets;
  std::vector<std::uint64_t> buffer_sizes;
  std::uint64_t length{};
  std::uint64_t priority{};
  lance_page_layout layout;
};

struct lance_column_info {
  std::vector<lance_page_info> pages;
};

struct lance_file_info {
  std::uint64_t num_rows{};
  std::vector<lance_field_info> fields;
  std::vector<lance_column_info> columns;
};

struct miniblock_chunk_info {
  std::uint64_t row_begin{};
  std::uint64_t num_rows{};
  std::uint64_t buffer_offset{};
  std::uint64_t buffer_size{};
};

struct miniblock_chunk_payload {
  std::uint64_t payload_offset{};
  std::uint32_t payload_size{};
};

std::size_t pad_size(std::size_t size, std::size_t alignment = lance_buffer_alignment)
{
  auto const remainder = size % alignment;
  return remainder == 0 ? 0 : alignment - remainder;
}

void write_host_buffer(data_sink* sink, void const* data, std::size_t size)
{
  if (size > 0) { sink->host_write(data, size); }
}

void append_aligned_host_buffer(std::vector<std::uint8_t>& output,
                                std::vector<std::uint8_t> const& data)
{
  output.insert(output.end(), data.begin(), data.end());
  output.insert(output.end(), pad_size(data.size()), lance_pad_byte);
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

void write_little_endian_u32(std::vector<std::uint8_t>& buffer, std::uint32_t value)
{
  for (int i = 0; i < 4; ++i) {
    buffer.push_back(static_cast<std::uint8_t>((value >> (i * 8)) & 0xff));
  }
}

std::uint16_t read_little_endian_u16(std::uint8_t const* data)
{
  return static_cast<std::uint16_t>(data[0]) |
         static_cast<std::uint16_t>(static_cast<std::uint16_t>(data[1]) << 8);
}

std::uint32_t read_little_endian_u32(std::uint8_t const* data)
{
  std::uint32_t value = 0;
  for (int i = 0; i < 4; ++i) {
    value |= static_cast<std::uint32_t>(data[i]) << (i * 8);
  }
  return value;
}

std::uint64_t read_little_endian_u64(std::uint8_t const* data)
{
  std::uint64_t value = 0;
  for (int i = 0; i < 8; ++i) {
    value |= static_cast<std::uint64_t>(data[i]) << (i * 8);
  }
  return value;
}

void expect_wire_type(wire_type actual, wire_type expected, std::string_view context)
{
  CUDF_EXPECTS(actual == expected, "Unexpected Lance protobuf wire type while reading " +
                                     std::string(context));
}

void append_repeated_uint64(proto_reader& reader,
                            wire_type type,
                            std::vector<std::uint64_t>& values)
{
  if (type == wire_type::length_delimited) {
    auto packed = reader.read_message();
    while (!packed.eof()) {
      values.push_back(packed.read_varint());
    }
  } else {
    expect_wire_type(type, wire_type::varint, "uint64 repeated field");
    values.push_back(reader.read_varint());
  }
}

lance_page_layout parse_compressive_encoding(proto_reader reader);

std::string parse_any_type_url(proto_reader reader, std::vector<std::uint8_t>& value)
{
  std::string type_url;
  int field{};
  wire_type type{};
  while (reader.next(field, type)) {
    switch (field) {
      case 1:
        expect_wire_type(type, wire_type::length_delimited, "Any.type_url");
        type_url = reader.read_string();
        break;
      case 2:
        expect_wire_type(type, wire_type::length_delimited, "Any.value");
        value = reader.read_bytes();
        break;
      default: reader.skip(type); break;
    }
  }
  return type_url;
}

std::vector<std::uint8_t> parse_direct_encoding_bytes(proto_reader reader)
{
  std::vector<std::uint8_t> direct_bytes;
  int field{};
  wire_type type{};
  while (reader.next(field, type)) {
    if (field == 2) {
      expect_wire_type(type, wire_type::length_delimited, "Encoding.direct");
      auto direct = reader.read_message();
      int direct_field{};
      wire_type direct_type{};
      while (direct.next(direct_field, direct_type)) {
        if (direct_field == 1) {
          expect_wire_type(direct_type, wire_type::length_delimited, "DirectEncoding.encoding");
          direct_bytes = direct.read_bytes();
        } else {
          direct.skip(direct_type);
        }
      }
    } else {
      reader.skip(type);
    }
  }
  CUDF_EXPECTS(!direct_bytes.empty(), "Lance encoding is missing direct bytes");
  return direct_bytes;
}

std::uint64_t parse_flat_bits(proto_reader reader)
{
  std::uint64_t bits{};
  int field{};
  wire_type type{};
  while (reader.next(field, type)) {
    if (field == 1) {
      expect_wire_type(type, wire_type::varint, "Flat.bits_per_value");
      bits = reader.read_varint();
    } else {
      reader.skip(type);
    }
  }
  CUDF_EXPECTS(bits > 0, "Lance flat encoding is missing bits_per_value");
  return bits;
}

compression_type parse_buffer_compression(proto_reader reader)
{
  compression_type compression = compression_type::NONE;
  int field{};
  wire_type type{};
  while (reader.next(field, type)) {
    if (field == 1) {
      expect_wire_type(type, wire_type::varint, "BufferCompression.scheme");
      auto const scheme = reader.read_varint();
      CUDF_EXPECTS(scheme == 2, "Only ZSTD Lance buffer compression is supported");
      compression = compression_type::ZSTD;
    } else {
      reader.skip(type);
    }
  }
  return compression;
}

lance_page_layout parse_general_encoding(proto_reader reader)
{
  lance_page_layout layout;
  int field{};
  wire_type type{};
  while (reader.next(field, type)) {
    switch (field) {
      case 1:
        expect_wire_type(type, wire_type::length_delimited, "General.compression");
        layout.compression = parse_buffer_compression(reader.read_message());
        break;
      case 3: {
        expect_wire_type(type, wire_type::length_delimited, "General.values");
        auto inner          = parse_compressive_encoding(reader.read_message());
        layout.bits_per_value = inner.bits_per_value;
        break;
      }
      default: reader.skip(type); break;
    }
  }
  CUDF_EXPECTS(layout.compression == compression_type::ZSTD,
               "Lance general encoding must specify ZSTD compression");
  CUDF_EXPECTS(layout.bits_per_value > 0, "Lance general encoding is missing inner flat encoding");
  return layout;
}

lance_page_layout parse_compressive_encoding(proto_reader reader)
{
  lance_page_layout layout;
  int field{};
  wire_type type{};
  while (reader.next(field, type)) {
    switch (field) {
      case 1:
        expect_wire_type(type, wire_type::length_delimited, "CompressiveEncoding.flat");
        layout.bits_per_value = parse_flat_bits(reader.read_message());
        layout.compression    = compression_type::NONE;
        break;
      case 10: {
        expect_wire_type(type, wire_type::length_delimited, "CompressiveEncoding.general");
        auto general          = parse_general_encoding(reader.read_message());
        layout.bits_per_value = general.bits_per_value;
        layout.compression    = general.compression;
        break;
      }
      default: reader.skip(type); break;
    }
  }
  CUDF_EXPECTS(layout.bits_per_value > 0, "Unsupported Lance compressive encoding");
  return layout;
}

lance_page_layout parse_miniblock_layout(proto_reader reader)
{
  lance_page_layout layout;
  int field{};
  wire_type type{};
  while (reader.next(field, type)) {
    switch (field) {
      case 3: {
        expect_wire_type(type, wire_type::length_delimited, "MiniBlockLayout.value_compression");
        auto value_layout     = parse_compressive_encoding(reader.read_message());
        layout.compression    = value_layout.compression;
        layout.bits_per_value = value_layout.bits_per_value;
        break;
      }
      case 7:
        expect_wire_type(type, wire_type::varint, "MiniBlockLayout.num_buffers");
        layout.num_buffers = reader.read_varint();
        break;
      case 9:
        expect_wire_type(type, wire_type::varint, "MiniBlockLayout.num_items");
        layout.num_items = reader.read_varint();
        break;
      case 10:
        expect_wire_type(type, wire_type::varint, "MiniBlockLayout.has_large_chunk");
        layout.has_large_chunk = reader.read_varint() != 0;
        break;
      default: reader.skip(type); break;
    }
  }
  CUDF_EXPECTS(layout.bits_per_value > 0, "Lance MiniBlock layout is missing value encoding");
  CUDF_EXPECTS(layout.num_buffers == 1, "Only single-buffer Lance MiniBlock pages are supported");
  CUDF_EXPECTS(layout.has_large_chunk, "Only Lance v2.2 large MiniBlock chunks are supported");
  return layout;
}

lance_page_layout parse_page_layout(proto_reader reader)
{
  lance_page_layout layout;
  int field{};
  wire_type type{};
  while (reader.next(field, type)) {
    if (field == 1) {
      expect_wire_type(type, wire_type::length_delimited, "PageLayout.mini_block_layout");
      layout = parse_miniblock_layout(reader.read_message());
    } else {
      reader.skip(type);
    }
  }
  CUDF_EXPECTS(layout.bits_per_value > 0, "Only Lance MiniBlock page layout is supported");
  return layout;
}

lance_page_layout parse_page_encoding(proto_reader reader)
{
  auto direct_bytes = parse_direct_encoding_bytes(reader);
  std::vector<std::uint8_t> any_value;
  auto const type_url = parse_any_type_url(proto_reader(direct_bytes), any_value);
  CUDF_EXPECTS(type_url == "/lance.encodings21.PageLayout",
               "Unsupported Lance page encoding: " + type_url);
  return parse_page_layout(proto_reader(any_value));
}

lance_field_info parse_field(proto_reader reader)
{
  std::string name;
  std::string logical;
  bool nullable = false;
  std::int32_t parent_id = -1;
  int field{};
  wire_type type{};
  while (reader.next(field, type)) {
    switch (field) {
      case 2:
        expect_wire_type(type, wire_type::length_delimited, "Field.name");
        name = reader.read_string();
        break;
      case 4:
        expect_wire_type(type, wire_type::varint, "Field.parent_id");
        {
          auto const raw_parent_id = static_cast<std::uint32_t>(reader.read_varint());
          parent_id                = raw_parent_id >
                              static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())
                            ? static_cast<std::int32_t>(
                                static_cast<std::int64_t>(raw_parent_id) -
                                (std::int64_t{1} << 32))
                            : static_cast<std::int32_t>(raw_parent_id);
        }
        break;
      case 5:
        expect_wire_type(type, wire_type::length_delimited, "Field.logical_type");
        logical = reader.read_string();
        break;
      case 6:
        expect_wire_type(type, wire_type::varint, "Field.nullable");
        nullable = reader.read_varint() != 0;
        break;
      default: reader.skip(type); break;
    }
  }
  CUDF_EXPECTS(parent_id == -1, "Only top-level Lance fields are supported");
  CUDF_EXPECTS(!name.empty(), "Lance field is missing a name");
  CUDF_EXPECTS(!logical.empty(), "Lance field is missing a logical type");
  auto dtype = data_type_from_lance_logical_type(logical);
  std::string unsupported_reason;
  if (nullable) { unsupported_reason = "nullable field"; }
  if (!dtype.has_value()) {
    if (!unsupported_reason.empty()) { unsupported_reason += "; "; }
    unsupported_reason += "unsupported logical type: " + logical;
  }
  if (!unsupported_reason.empty()) { dtype = std::nullopt; }
  return lance_field_info{name, dtype, nullable, logical, unsupported_reason};
}

std::vector<lance_field_info> parse_schema(proto_reader reader)
{
  std::vector<lance_field_info> fields;
  int field{};
  wire_type type{};
  while (reader.next(field, type)) {
    if (field == 1) {
      expect_wire_type(type, wire_type::length_delimited, "Schema.fields");
      fields.push_back(parse_field(reader.read_message()));
    } else {
      reader.skip(type);
    }
  }
  return fields;
}

lance_file_info parse_file_descriptor(std::vector<std::uint8_t> const& bytes)
{
  lance_file_info info;
  proto_reader reader(bytes);
  int field{};
  wire_type type{};
  while (reader.next(field, type)) {
    switch (field) {
      case 1:
        expect_wire_type(type, wire_type::length_delimited, "FileDescriptor.schema");
        info.fields = parse_schema(reader.read_message());
        break;
      case 2:
        expect_wire_type(type, wire_type::varint, "FileDescriptor.length");
        info.num_rows = reader.read_varint();
        break;
      default: reader.skip(type); break;
    }
  }
  CUDF_EXPECTS(!info.fields.empty(), "Lance file descriptor is missing schema fields");
  return info;
}

lance_page_info parse_page_metadata(proto_reader reader)
{
  lance_page_info page;
  int field{};
  wire_type type{};
  while (reader.next(field, type)) {
    switch (field) {
      case 1: append_repeated_uint64(reader, type, page.buffer_offsets); break;
      case 2: append_repeated_uint64(reader, type, page.buffer_sizes); break;
      case 3:
        expect_wire_type(type, wire_type::varint, "Page.length");
        page.length = reader.read_varint();
        break;
      case 4:
        expect_wire_type(type, wire_type::length_delimited, "Page.encoding");
        page.layout = parse_page_encoding(reader.read_message());
        break;
      case 5:
        expect_wire_type(type, wire_type::varint, "Page.priority");
        page.priority = reader.read_varint();
        break;
      default: reader.skip(type); break;
    }
  }
  CUDF_EXPECTS(page.buffer_offsets.size() == 2 && page.buffer_sizes.size() == 2,
               "Only two-buffer Lance MiniBlock pages are supported");
  CUDF_EXPECTS(page.length == page.layout.num_items,
               "Lance MiniBlock page length does not match num_items");
  return page;
}

lance_column_info parse_column_metadata(proto_reader reader, bool parse_pages)
{
  lance_column_info column;
  int field{};
  wire_type type{};
  while (reader.next(field, type)) {
    if (field == 2) {
      expect_wire_type(type, wire_type::length_delimited, "ColumnMetadata.pages");
      if (parse_pages) {
        column.pages.push_back(parse_page_metadata(reader.read_message()));
      } else {
        reader.skip(type);
      }
    } else {
      reader.skip(type);
    }
  }
  return column;
}

std::vector<std::uint8_t> read_host_bytes(datasource* source,
                                          std::uint64_t offset,
                                          std::uint64_t size)
{
  CUDF_EXPECTS(offset <= source->size() && size <= source->size() - offset,
               "Lance file read range is out of bounds");
  CUDF_EXPECTS(offset <= static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) &&
                 size <= static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()),
               "Lance host read range is too large");
  auto const offset_bytes = static_cast<std::size_t>(offset);
  auto const size_bytes = static_cast<std::size_t>(size);
  std::vector<std::uint8_t> bytes(size_bytes);
  if (size > 0) {
    auto const bytes_read = source->host_read(offset_bytes, size_bytes, bytes.data());
    CUDF_EXPECTS(bytes_read == size_bytes, "Failed to read expected Lance host bytes");
  }
  return bytes;
}

std::vector<std::pair<std::uint64_t, std::uint64_t>> read_offset_table(datasource* source,
                                                                       std::uint64_t offset,
                                                                       std::uint32_t count)
{
  auto const bytes = read_host_bytes(source, offset, static_cast<std::uint64_t>(count) * 16);
  std::vector<std::pair<std::uint64_t, std::uint64_t>> table;
  table.reserve(count);
  for (std::uint32_t idx = 0; idx < count; ++idx) {
    auto const* entry = bytes.data() + (static_cast<std::size_t>(idx) * 16);
    table.emplace_back(read_little_endian_u64(entry), read_little_endian_u64(entry + 8));
  }
  return table;
}

struct lance_footer {
  std::uint64_t column_metadata_start{};
  std::uint64_t cmo_table_start{};
  std::uint64_t gbo_table_start{};
  std::uint32_t num_global_buffers{};
  std::uint32_t num_columns{};
  std::uint16_t major{};
  std::uint16_t minor{};
};

lance_footer read_footer(datasource* source)
{
  constexpr std::size_t footer_size = 40;
  CUDF_EXPECTS(source->size() >= footer_size, "File too small to contain a Lance footer");
  auto const footer_bytes = read_host_bytes(source, source->size() - footer_size, footer_size);
  CUDF_EXPECTS(std::equal(lance_magic.begin(),
                         lance_magic.end(),
                         footer_bytes.data() + footer_size - lance_magic.size()),
               "Invalid Lance magic number");

  lance_footer footer;
  footer.column_metadata_start = read_little_endian_u64(footer_bytes.data());
  footer.cmo_table_start       = read_little_endian_u64(footer_bytes.data() + 8);
  footer.gbo_table_start       = read_little_endian_u64(footer_bytes.data() + 16);
  footer.num_global_buffers    = read_little_endian_u32(footer_bytes.data() + 24);
  footer.num_columns           = read_little_endian_u32(footer_bytes.data() + 28);
  footer.major                 = read_little_endian_u16(footer_bytes.data() + 32);
  footer.minor                 = read_little_endian_u16(footer_bytes.data() + 34);

  CUDF_EXPECTS(footer.major == 2 && footer.minor == 2,
               "Lance reader currently supports Lance file version 2.2");
  CUDF_EXPECTS(footer.num_global_buffers >= 1,
               "Lance file is missing schema global buffer");
  return footer;
}

struct lance_file_info_cache_key {
  std::uint64_t source_size{};
  lance_footer footer;
  std::vector<std::string> requested_columns;
};

bool same_footer(lance_footer const& lhs, lance_footer const& rhs)
{
  return lhs.column_metadata_start == rhs.column_metadata_start &&
         lhs.cmo_table_start == rhs.cmo_table_start &&
         lhs.gbo_table_start == rhs.gbo_table_start &&
         lhs.num_global_buffers == rhs.num_global_buffers &&
         lhs.num_columns == rhs.num_columns &&
         lhs.major == rhs.major &&
         lhs.minor == rhs.minor;
}

bool same_lance_file_info_cache_key(lance_file_info_cache_key const& lhs,
                                    lance_file_info_cache_key const& rhs)
{
  return lhs.source_size == rhs.source_size && same_footer(lhs.footer, rhs.footer) &&
         lhs.requested_columns == rhs.requested_columns;
}

struct lance_file_info_cache_entry {
  lance_file_info_cache_key key;
  lance_file_info info;
};

// Sparse lookup workloads repeatedly read the same Lance file metadata. Keep a tiny
// process-local cache keyed by immutable footer fields and the requested projection.
std::mutex& lance_file_info_cache_mutex()
{
  static std::mutex mutex;
  return mutex;
}

std::vector<lance_file_info_cache_entry>& lance_file_info_cache()
{
  static std::vector<lance_file_info_cache_entry> cache;
  return cache;
}

std::optional<lance_file_info> get_cached_lance_file_info(lance_file_info_cache_key const& key)
{
  std::lock_guard<std::mutex> lock(lance_file_info_cache_mutex());
  auto& cache = lance_file_info_cache();
  auto found = std::find_if(cache.begin(), cache.end(), [&](auto const& entry) {
    return same_lance_file_info_cache_key(entry.key, key);
  });
  if (found == cache.end()) { return std::nullopt; }
  return found->info;
}

void put_cached_lance_file_info(lance_file_info_cache_key key, lance_file_info const& info)
{
  constexpr std::size_t max_lance_file_info_cache_entries = 8;
  std::lock_guard<std::mutex> lock(lance_file_info_cache_mutex());
  auto& cache = lance_file_info_cache();
  auto found = std::find_if(cache.begin(), cache.end(), [&](auto const& entry) {
    return same_lance_file_info_cache_key(entry.key, key);
  });
  if (found != cache.end()) {
    found->info = info;
    return;
  }
  if (cache.size() >= max_lance_file_info_cache_entries) { cache.erase(cache.begin()); }
  cache.push_back(lance_file_info_cache_entry{std::move(key), info});
}

bool should_parse_column_pages(lance_field_info const& field,
                               std::vector<std::string> const& requested_columns)
{
  return field.is_supported() &&
         (requested_columns.empty() ||
          std::find(requested_columns.begin(), requested_columns.end(), field.name) !=
            requested_columns.end());
}

lance_file_info read_lance_file_info(datasource* source,
                                     std::vector<std::string> const& requested_columns)
{
  auto const footer = read_footer(source);
  lance_file_info_cache_key cache_key{
    static_cast<std::uint64_t>(source->size()), footer, requested_columns};
  if (auto cached = get_cached_lance_file_info(cache_key); cached.has_value()) {
    return std::move(*cached);
  }

  auto const global_buffers = read_offset_table(
    source, footer.gbo_table_start, footer.num_global_buffers);
  auto info = parse_file_descriptor(
    read_host_bytes(source, global_buffers[0].first, global_buffers[0].second));

  CUDF_EXPECTS(info.fields.size() == footer.num_columns,
               "Lance schema field count does not match footer column count");

  auto const column_offsets =
    read_offset_table(source, footer.cmo_table_start, footer.num_columns);
  info.columns.reserve(footer.num_columns);
  if (!column_offsets.empty()) {
    std::uint64_t begin = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t end   = 0;
    std::uint64_t total = 0;
    for (auto const& [offset, size] : column_offsets) {
      CUDF_EXPECTS(offset <= source->size() && size <= source->size() - offset,
                   "Lance column metadata range is out of bounds");
      begin = std::min(begin, offset);
      end   = std::max(end, offset + size);
      total += size;
    }

    auto const span_size = end - begin;
    auto const max_extra = std::max<std::uint64_t>(1 << 20, total / 8);
    if (span_size <= total + max_extra) {
      auto const metadata = read_host_bytes(source, begin, span_size);
      for (std::size_t idx = 0; idx < column_offsets.size(); ++idx) {
        auto const& [offset, size] = column_offsets[idx];
        info.columns.push_back(parse_column_metadata(
          proto_reader(metadata.data() + static_cast<std::size_t>(offset - begin),
                       static_cast<std::size_t>(size)),
          should_parse_column_pages(info.fields[idx], requested_columns)));
      }
    } else {
      for (std::size_t idx = 0; idx < column_offsets.size(); ++idx) {
        auto const& [offset, size] = column_offsets[idx];
        auto const metadata_bytes = read_host_bytes(source, offset, size);
        info.columns.push_back(parse_column_metadata(
          proto_reader(metadata_bytes),
          should_parse_column_pages(info.fields[idx], requested_columns)));
      }
    }
  }
  CUDF_EXPECTS(info.columns.size() == info.fields.size(),
               "Lance column metadata count does not match schema field count");
  put_cached_lance_file_info(std::move(cache_key), info);
  return info;
}

void read_device_bytes_into(datasource* source,
                            std::uint64_t offset,
                            std::uint64_t size,
                            std::uint8_t* output,
                            rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(size <= static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()),
               "Lance device read range is too large");
  if (size == 0) { return; }

  CUDF_EXPECTS(offset <= source->size() && size <= source->size() - offset,
               "Lance device read range is out of bounds");
  CUDF_EXPECTS(offset <= static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()),
               "Lance device read offset is too large");
  auto const offset_bytes = static_cast<std::size_t>(offset);
  auto const size_bytes = static_cast<std::size_t>(size);
  if (source->is_device_read_preferred(size_bytes)) {
    auto const bytes_read = source->device_read(offset_bytes, size_bytes, output, stream);
    CUDF_EXPECTS(bytes_read == size_bytes, "Failed to read expected Lance device bytes");
  } else {
    auto host_data = source->host_read(offset_bytes, size_bytes);
    CUDF_EXPECTS(host_data->size() == size_bytes, "Failed to read expected Lance host bytes");
    CUDF_CUDA_TRY(cudf::detail::memcpy_async(output, host_data->data(), size_bytes, stream));
    stream.synchronize();
  }
}

rmm::device_uvector<std::uint8_t> read_device_bytes(datasource* source,
                                                    std::uint64_t offset,
                                                    std::uint64_t size,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(size <= static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()),
               "Lance device read range is too large");
  rmm::device_uvector<std::uint8_t> data(static_cast<std::size_t>(size), stream, mr);
  read_device_bytes_into(source, offset, size, data.data(), stream);
  return data;
}

struct pending_device_read {
  std::uint64_t offset{};
  std::uint64_t size{};
  std::uint8_t* output{};
};

void read_device_byte_ranges_into(datasource* source,
                                  std::vector<pending_device_read> const& reads,
                                  rmm::cuda_stream_view stream)
{
  struct staged_read_info {
    std::size_t read_idx{};
    std::size_t staging_offset{};
    std::size_t size{};
  };

  std::size_t non_empty_reads = 0;
  std::uint64_t total_read_size = 0;
  auto can_stage_small_reads = reads.size() >= sparse_host_staging_min_reads;
  for (auto const& read : reads) {
    if (read.size == 0) { continue; }
    CUDF_EXPECTS(read.size <= static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()),
                 "Lance device read range is too large");
    CUDF_EXPECTS(read.offset <= source->size() && read.size <= source->size() - read.offset,
                 "Lance device read range is out of bounds");
    CUDF_EXPECTS(read.offset <= static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()),
                 "Lance device read offset is too large");
    auto const size_bytes = static_cast<std::size_t>(read.size);
    can_stage_small_reads =
      can_stage_small_reads && source->is_device_read_preferred(size_bytes) &&
      read.size <= sparse_host_staging_max_read_bytes &&
      total_read_size <= sparse_host_staging_max_total_bytes - read.size;
    total_read_size += read.size;
    ++non_empty_reads;
  }

  if (can_stage_small_reads && non_empty_reads >= sparse_host_staging_min_reads) {
    CUDF_EXPECTS(total_read_size <=
                   static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()),
                 "Lance staged read buffer is too large");
    auto staging =
      cudf::detail::make_pinned_vector<std::uint8_t>(static_cast<std::size_t>(total_read_size),
                                                     stream);
    std::vector<staged_read_info> staged_reads;
    std::vector<std::future<std::size_t>> futures;
    staged_reads.reserve(non_empty_reads);
    futures.reserve(non_empty_reads);

    std::size_t staging_offset = 0;
    for (std::size_t read_idx = 0; read_idx < reads.size(); ++read_idx) {
      auto const& read = reads[read_idx];
      if (read.size == 0) { continue; }
      auto const offset_bytes = static_cast<std::size_t>(read.offset);
      auto const size_bytes   = static_cast<std::size_t>(read.size);
      staged_reads.push_back(staged_read_info{read_idx, staging_offset, size_bytes});
      futures.push_back(
        source->host_read_async(offset_bytes, size_bytes, staging.data() + staging_offset));
      staging_offset += size_bytes;
    }

    for (std::size_t idx = 0; idx < futures.size(); ++idx) {
      auto const bytes_read = futures[idx].get();
      CUDF_EXPECTS(bytes_read == staged_reads[idx].size,
                   "Failed to read expected Lance host bytes");
    }

    for (std::size_t run_begin = 0; run_begin < staged_reads.size();) {
      auto const& first_staged_read = staged_reads[run_begin];
      auto const& first_read        = reads[first_staged_read.read_idx];
      auto const run_staging_begin  = first_staged_read.staging_offset;
      auto const run_output_begin   = reinterpret_cast<std::uintptr_t>(first_read.output);
      auto run_size                 = first_staged_read.size;
      auto run_end                  = run_begin + 1;
      while (run_end < staged_reads.size()) {
        auto const& staged_read = staged_reads[run_end];
        auto const& read        = reads[staged_read.read_idx];
        auto const output_offset = staged_read.staging_offset - run_staging_begin;
        CUDF_EXPECTS(run_output_begin <=
                       std::numeric_limits<std::uintptr_t>::max() - output_offset,
                     "Lance staged read output range is too large");
        if (reinterpret_cast<std::uintptr_t>(read.output) != run_output_begin + output_offset) {
          break;
        }
        run_size = staged_read.staging_offset + staged_read.size - run_staging_begin;
        ++run_end;
      }
      CUDF_CUDA_TRY(cudf::detail::memcpy_async(reinterpret_cast<void*>(run_output_begin),
                                               staging.data() + run_staging_begin,
                                               run_size,
                                               stream));
      run_begin = run_end;
    }
    stream.synchronize();
    return;
  }

  std::vector<std::future<std::size_t>> futures;
  std::vector<std::size_t> expected_sizes;
  futures.reserve(reads.size());
  expected_sizes.reserve(reads.size());

  for (auto const& read : reads) {
    if (read.size == 0) { continue; }
    CUDF_EXPECTS(read.size <= static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()),
                 "Lance device read range is too large");
    CUDF_EXPECTS(read.offset <= source->size() && read.size <= source->size() - read.offset,
                 "Lance device read range is out of bounds");
    CUDF_EXPECTS(read.offset <= static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()),
                 "Lance device read offset is too large");

    auto const offset_bytes = static_cast<std::size_t>(read.offset);
    auto const size_bytes   = static_cast<std::size_t>(read.size);
    if (source->is_device_read_preferred(size_bytes)) {
      futures.push_back(source->device_read_async(offset_bytes, size_bytes, read.output, stream));
      expected_sizes.push_back(size_bytes);
    } else {
      read_device_bytes_into(source, read.offset, read.size, read.output, stream);
    }
  }

  for (std::size_t idx = 0; idx < futures.size(); ++idx) {
    auto const bytes_read = futures[idx].get();
    CUDF_EXPECTS(bytes_read == expected_sizes[idx],
                 "Failed to read expected Lance device bytes");
  }
}

struct page_data_buffers {
  std::vector<rmm::device_uvector<std::uint8_t>> storage;
  std::vector<std::uint8_t const*> pages;
};

page_data_buffers read_page_data_buffers(datasource* source,
                                         std::vector<lance_page_info> const& pages,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  page_data_buffers result;
  result.pages.reserve(pages.size());
  if (pages.empty()) { return result; }

  std::uint64_t begin = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t end   = 0;
  std::uint64_t total = 0;
  for (auto const& page : pages) {
    auto const offset = page.buffer_offsets[1];
    auto const size   = page.buffer_sizes[1];
    CUDF_EXPECTS(offset <= source->size() && size <= source->size() - offset,
                 "Lance page data range is out of bounds");
    begin = std::min(begin, offset);
    end   = std::max(end, offset + size);
    total += size;
  }

  auto const span_size = end - begin;
  auto const max_extra = std::max<std::uint64_t>(1 << 20, total / 8);
  if (span_size <= total + max_extra) {
    result.storage.emplace_back(static_cast<std::size_t>(span_size), stream, mr);
    read_device_bytes_into(source, begin, span_size, result.storage.back().data(), stream);
    auto const* data = result.storage.back().data();
    for (auto const& page : pages) {
      result.pages.push_back(data + static_cast<std::size_t>(page.buffer_offsets[1] - begin));
    }
    return result;
  }

  result.storage.reserve(pages.size());
  for (auto const& page : pages) {
    result.storage.push_back(
      read_device_bytes(source, page.buffer_offsets[1], page.buffer_sizes[1], stream, mr));
    result.pages.push_back(result.storage.back().data());
  }
  return result;
}

void append_miniblock_metadata(std::vector<std::uint8_t>& metadata,
                               std::uint64_t miniblock_buffer_size,
                               std::uint8_t log_num_values)
{
  CUDF_EXPECTS(miniblock_buffer_size > 0 && miniblock_buffer_size % 8 == 0,
               "Invalid Lance miniblock buffer size");
  auto const words = (miniblock_buffer_size / 8) - 1;
  CUDF_EXPECTS(words <= ((std::uint64_t{1} << 28) - 1),
               "Lance miniblock is too large to describe with v2.2 metadata");
  CUDF_EXPECTS(log_num_values <= 15, "Lance miniblock row count is too large for metadata");
  auto const metadata_word =
    static_cast<std::uint32_t>((words << 4) | static_cast<std::uint64_t>(log_num_values));

  write_little_endian_u32(metadata, metadata_word);
}

std::vector<miniblock_chunk_info> parse_miniblock_chunks(std::uint8_t const* metadata,
                                                         std::size_t metadata_size,
                                                         lance_page_info const& page)
{
  CUDF_EXPECTS((metadata_size == 0 && page.length == 0) ||
                 (metadata_size > 0 && metadata_size % sizeof(std::uint32_t) == 0),
               "Invalid Lance MiniBlock metadata buffer");

  auto const num_chunks = metadata_size / sizeof(std::uint32_t);
  std::vector<miniblock_chunk_info> chunks;
  chunks.reserve(num_chunks);

  std::uint64_t rows_seen   = 0;
  std::uint64_t bytes_seen  = 0;
  auto const data_buf_begin = page.buffer_offsets[1];
  for (std::size_t idx = 0; idx < num_chunks; ++idx) {
    auto const metadata_word =
      read_little_endian_u32(metadata + (idx * sizeof(std::uint32_t)));
    auto const log_num_values = static_cast<std::uint8_t>(metadata_word & 0xf);
    auto const chunk_size =
      ((static_cast<std::uint64_t>(metadata_word) >> 4) + 1) * miniblock_alignment;

    CUDF_EXPECTS(bytes_seen <= page.buffer_sizes[1] &&
                   chunk_size <= page.buffer_sizes[1] - bytes_seen,
                 "Lance MiniBlock metadata exceeds data buffer size");
    CUDF_EXPECTS(rows_seen < page.length, "Lance MiniBlock metadata exceeds page length");

    std::uint64_t chunk_rows{};
    auto const rows_remaining = page.length - rows_seen;
    if (idx + 1 < num_chunks) {
      CUDF_EXPECTS(log_num_values > 0,
                   "Non-final Lance MiniBlock chunks must store a power-of-two row count");
      chunk_rows = std::uint64_t{1} << log_num_values;
      CUDF_EXPECTS(chunk_rows < rows_remaining,
                   "Lance MiniBlock row metadata exceeds page length");
    } else {
      if (log_num_values != 0) {
        auto const encoded_rows = std::uint64_t{1} << log_num_values;
        CUDF_EXPECTS(encoded_rows == rows_remaining,
                     "Final Lance MiniBlock row count does not match page length");
      }
      chunk_rows = rows_remaining;
    }

    chunks.push_back(
      miniblock_chunk_info{rows_seen, chunk_rows, data_buf_begin + bytes_seen, chunk_size});
    rows_seen += chunk_rows;
    bytes_seen += chunk_size;
  }

  CUDF_EXPECTS(rows_seen == page.length, "Lance MiniBlock row metadata does not match page length");
  CUDF_EXPECTS(bytes_seen == page.buffer_sizes[1],
               "Lance MiniBlock metadata does not match data buffer size");
  return chunks;
}

std::vector<miniblock_chunk_info> read_miniblock_chunks(datasource* source,
                                                        lance_page_info const& page)
{
  auto const metadata = read_host_bytes(source, page.buffer_offsets[0], page.buffer_sizes[0]);
  return parse_miniblock_chunks(metadata.data(), metadata.size(), page);
}

std::vector<std::vector<miniblock_chunk_info>> read_miniblock_chunks(
  datasource* source, std::vector<lance_page_info> const& pages)
{
  std::vector<std::vector<miniblock_chunk_info>> result(pages.size());
  if (pages.empty()) { return result; }

  std::uint64_t begin = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t end   = 0;
  std::uint64_t total = 0;
  for (auto const& page : pages) {
    auto const offset = page.buffer_offsets[0];
    auto const size   = page.buffer_sizes[0];
    CUDF_EXPECTS(offset <= source->size() && size <= source->size() - offset,
                 "Lance MiniBlock metadata range is out of bounds");
    begin = std::min(begin, offset);
    end   = std::max(end, offset + size);
    total += size;
  }

  auto const span_size = end - begin;
  auto const max_extra = std::max<std::uint64_t>(1 << 20, total / 8);
  if (span_size <= total + max_extra) {
    auto const metadata = read_host_bytes(source, begin, span_size);
    for (std::size_t page_idx = 0; page_idx < pages.size(); ++page_idx) {
      auto const& page = pages[page_idx];
      auto const offset =
        static_cast<std::size_t>(page.buffer_offsets[0] - begin);
      result[page_idx] = parse_miniblock_chunks(
        metadata.data() + offset, static_cast<std::size_t>(page.buffer_sizes[0]), page);
    }
    return result;
  }

  for (std::size_t page_idx = 0; page_idx < pages.size(); ++page_idx) {
    result[page_idx] = read_miniblock_chunks(source, pages[page_idx]);
  }
  return result;
}

std::vector<std::size_t> read_touched_miniblock_chunks(
  datasource* source,
  std::vector<lance_page_info> const& pages,
  std::vector<std::vector<std::pair<size_type, size_type>>> const& rows_by_page,
  std::vector<std::vector<miniblock_chunk_info>>& chunks_by_page)
{
  std::vector<std::size_t> touched_page_indices;

  for (std::size_t page_idx = 0; page_idx < rows_by_page.size(); ++page_idx) {
    if (rows_by_page[page_idx].empty()) { continue; }
    touched_page_indices.push_back(page_idx);
  }

  if (touched_page_indices.empty()) { return touched_page_indices; }

  if (touched_page_indices.size() == 1) {
    auto const page_idx   = touched_page_indices.front();
    chunks_by_page[page_idx] = read_miniblock_chunks(source, pages[page_idx]);
    return touched_page_indices;
  }

  std::vector<lance_page_info> touched_pages;
  touched_pages.reserve(touched_page_indices.size());
  for (auto page_idx : touched_page_indices) {
    touched_pages.push_back(pages[page_idx]);
  }

  auto touched_chunks = read_miniblock_chunks(source, touched_pages);
  for (std::size_t idx = 0; idx < touched_page_indices.size(); ++idx) {
    chunks_by_page[touched_page_indices[idx]] = std::move(touched_chunks[idx]);
  }
  return touched_page_indices;
}

struct pending_miniblock_chunk {
  std::uint64_t num_rows{};
  std::uint8_t log_num_values{};
  std::uint8_t const* payload{};
  std::size_t payload_size{};
  std::uint64_t buffer_size{};
};

struct pending_lance_page {
  size_type row_begin{};
  size_type num_rows{};
  std::size_t chunk_begin{};
  std::size_t chunk_count{};
};

struct pending_lance_column {
  std::size_t type_width{};
  std::vector<pending_lance_page> pages;
  std::vector<pending_miniblock_chunk> chunks;
};

void append_page_chunks(std::vector<pending_miniblock_chunk>& chunks,
                        column_view const& column,
                        size_type row_begin,
                        size_type num_rows,
                        size_type rows_per_miniblock)
{
  auto const type_width         = cudf::size_of(column.type());
  auto const miniblock_log      = static_cast<std::uint8_t>(
    std::bit_width(static_cast<std::uint32_t>(rows_per_miniblock)) - 1);
  chunks.reserve(chunks.size() +
                 static_cast<std::size_t>(
                   (num_rows + rows_per_miniblock - 1) / rows_per_miniblock));

  for (size_type chunk_row = 0; chunk_row < num_rows; chunk_row += rows_per_miniblock) {
    auto const rows = std::min(rows_per_miniblock, num_rows - chunk_row);
    auto const is_last_chunk = chunk_row + rows == num_rows;
    CUDF_EXPECTS(is_last_chunk || rows == rows_per_miniblock,
                 "Non-final Lance MiniBlock chunks must have the default row count");

    auto const raw_size   = static_cast<std::size_t>(rows) * type_width;
    auto const* raw_begin = column.head<std::uint8_t>() +
                            (static_cast<std::size_t>(column.offset() + row_begin + chunk_row) *
                             type_width);

    pending_miniblock_chunk chunk;
    chunk.num_rows       = rows;
    chunk.log_num_values = is_last_chunk ? 0 : miniblock_log;
    chunk.payload        = raw_begin;
    chunk.payload_size   = raw_size;
    chunks.push_back(std::move(chunk));
  }
}

void finalize_miniblock_chunks(std::vector<pending_miniblock_chunk>& chunks)
{
  for (auto& chunk : chunks) {
    CUDF_EXPECTS(chunk.payload_size <= std::numeric_limits<std::uint32_t>::max(),
                 "Lance MiniBlock payload is too large");

    auto const header_size = sizeof(std::uint16_t) + sizeof(std::uint32_t);
    auto const header_pad  = pad_size(header_size, miniblock_alignment);
    auto const data_pad    = pad_size(chunk.payload_size, miniblock_alignment);
    chunk.buffer_size =
      static_cast<std::uint64_t>(header_size + header_pad + chunk.payload_size + data_pad);
  }
}

rmm::device_uvector<std::uint8_t> compress_zstd_miniblocks(
  std::vector<pending_miniblock_chunk*>& chunks, rmm::cuda_stream_view stream)
{
  if (chunks.empty()) { return rmm::device_uvector<std::uint8_t>(0, stream); }

  auto disabled = cudf::io::detail::nvcomp::is_compression_disabled(
    cudf::io::detail::nvcomp::compression_type::ZSTD);
  CUDF_EXPECTS(!disabled.has_value(), "nvCOMP ZSTD compression is disabled: " + disabled.value());

  auto const max_allowed = cudf::io::detail::nvcomp::compress_max_allowed_chunk_size(
    cudf::io::detail::nvcomp::compression_type::ZSTD);
  auto const required_alignment = cudf::io::detail::nvcomp::compress_required_alignment(
    cudf::io::detail::nvcomp::compression_type::ZSTD);

  bool needs_aligned_inputs = false;
  std::size_t aligned_input_size = 0;
  std::vector<std::size_t> input_offsets(chunks.size());
  std::vector<std::size_t> output_offsets(chunks.size());
  std::vector<std::size_t> max_output_sizes(chunks.size());
  std::size_t compressed_buffer_size = 0;

  for (std::size_t idx = 0; idx < chunks.size(); ++idx) {
    auto const& chunk = *chunks[idx];
    CUDF_EXPECTS(!max_allowed.has_value() || chunk.payload_size <= *max_allowed,
                 "Lance ZSTD chunk is larger than nvCOMP's maximum supported chunk size");
    needs_aligned_inputs |= required_alignment > 1 &&
                            (reinterpret_cast<std::uintptr_t>(chunk.payload) %
                               required_alignment) != 0;

    aligned_input_size += pad_size(aligned_input_size, required_alignment);
    input_offsets[idx] = aligned_input_size;
    aligned_input_size += chunk.payload_size;

    compressed_buffer_size += pad_size(compressed_buffer_size, required_alignment);
    output_offsets[idx] = compressed_buffer_size;
    max_output_sizes[idx] = cudf::io::detail::nvcomp::compress_max_output_chunk_size(
      cudf::io::detail::nvcomp::compression_type::ZSTD, chunk.payload_size);
    compressed_buffer_size += max_output_sizes[idx];
  }

  rmm::device_buffer aligned_inputs;
  if (needs_aligned_inputs) {
    aligned_inputs = rmm::device_buffer(aligned_input_size, stream);
    for (std::size_t idx = 0; idx < chunks.size(); ++idx) {
      CUDF_CUDA_TRY(cudaMemcpyAsync(static_cast<std::uint8_t*>(aligned_inputs.data()) +
                                      input_offsets[idx],
                                    chunks[idx]->payload,
                                    chunks[idx]->payload_size,
                                    cudaMemcpyDeviceToDevice,
                                    stream.value()));
    }
  }

  rmm::device_uvector<std::uint8_t> compressed(compressed_buffer_size, stream);
  auto inputs =
    cudf::detail::hostdevice_vector<device_span<std::uint8_t const>>(chunks.size(), stream);
  for (std::size_t idx = 0; idx < chunks.size(); ++idx) {
    auto const* input = needs_aligned_inputs
                          ? static_cast<std::uint8_t const*>(aligned_inputs.data()) +
                              input_offsets[idx]
                          : chunks[idx]->payload;
    inputs[idx] = device_span<std::uint8_t const>{input, chunks[idx]->payload_size};
  }
  inputs.host_to_device_async(stream);

  auto outputs =
    cudf::detail::hostdevice_vector<device_span<std::uint8_t>>(chunks.size(), stream);
  for (std::size_t idx = 0; idx < chunks.size(); ++idx) {
    outputs[idx] =
      device_span<std::uint8_t>{compressed.data() + output_offsets[idx], max_output_sizes[idx]};
  }
  outputs.host_to_device_async(stream);

  auto results =
    cudf::detail::hostdevice_vector<cudf::io::detail::codec_exec_result>(chunks.size(), stream);
  for (std::size_t idx = 0; idx < chunks.size(); ++idx) {
    results[idx] =
      cudf::io::detail::codec_exec_result{0, cudf::io::detail::codec_status::FAILURE};
  }
  results.host_to_device_async(stream);

  cudf::io::detail::nvcomp::batched_compress(
    cudf::io::detail::nvcomp::compression_type::ZSTD, inputs, outputs, results, stream);
  results.device_to_host(stream);

  for (std::size_t idx = 0; idx < chunks.size(); ++idx) {
    CUDF_EXPECTS(results[idx].status == cudf::io::detail::codec_status::SUCCESS,
                 "nvCOMP ZSTD failed to compress a Lance page");
    chunks[idx]->payload      = compressed.data() + output_offsets[idx];
    chunks[idx]->payload_size = results[idx].bytes_written;
  }
  return compressed;
}

void decompress_zstd_device_batch_into(
  std::vector<device_span<std::uint8_t const>> const& input_spans,
  std::vector<device_span<std::uint8_t>> const& output_spans,
  std::vector<std::size_t> const& output_sizes,
  std::size_t max_output_size,
  std::size_t total_output_size,
  rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(!input_spans.empty(), "Lance ZSTD decompression batch must not be empty");
  CUDF_EXPECTS(input_spans.size() == output_spans.size(),
               "Lance ZSTD decompression input/output batch size mismatch");
  CUDF_EXPECTS(input_spans.size() == output_sizes.size(),
               "Lance ZSTD decompression result batch size mismatch");

  auto disabled = cudf::io::detail::nvcomp::is_decompression_disabled(
    cudf::io::detail::nvcomp::compression_type::ZSTD);
  CUDF_EXPECTS(!disabled.has_value(), "nvCOMP ZSTD decompression is disabled: " + disabled.value());

  auto inputs =
    cudf::detail::hostdevice_vector<device_span<std::uint8_t const>>(input_spans.size(), stream);
  for (std::size_t idx = 0; idx < input_spans.size(); ++idx) {
    inputs[idx] = input_spans[idx];
  }
  inputs.host_to_device_async(stream);

  auto outputs =
    cudf::detail::hostdevice_vector<device_span<std::uint8_t>>(output_spans.size(), stream);
  for (std::size_t idx = 0; idx < output_spans.size(); ++idx) {
    outputs[idx] = output_spans[idx];
  }
  outputs.host_to_device_async(stream);

  auto results =
    cudf::detail::hostdevice_vector<cudf::io::detail::codec_exec_result>(input_spans.size(),
                                                                         stream);
  for (std::size_t idx = 0; idx < input_spans.size(); ++idx) {
    results[idx] =
      cudf::io::detail::codec_exec_result{0, cudf::io::detail::codec_status::FAILURE};
  }
  results.host_to_device_async(stream);

  cudf::io::detail::nvcomp::batched_decompress(cudf::io::detail::nvcomp::compression_type::ZSTD,
                                               inputs,
                                               outputs,
                                               results,
                                               max_output_size,
                                               total_output_size,
                                               stream);
  results.device_to_host(stream);

  for (std::size_t idx = 0; idx < input_spans.size(); ++idx) {
    CUDF_EXPECTS(results[idx].status == cudf::io::detail::codec_status::SUCCESS &&
                   results[idx].bytes_written == output_sizes[idx],
                 "nvCOMP ZSTD failed to decompress a Lance page");
  }
}

miniblock_chunk_payload read_miniblock_chunk_payload(datasource* source,
                                                     miniblock_chunk_info const& chunk)
{
  constexpr std::size_t miniblock_header_size = 8;
  CUDF_EXPECTS(chunk.buffer_size >= miniblock_header_size, "Invalid Lance MiniBlock data buffer");

  auto const header = read_host_bytes(source, chunk.buffer_offset, miniblock_header_size);
  auto const num_levels = read_little_endian_u16(header.data());
  CUDF_EXPECTS(num_levels == 0, "Lance reader does not yet support repetition/definition levels");
  auto const payload_size = read_little_endian_u32(header.data() + sizeof(std::uint16_t));
  auto const expected_chunk_size =
    miniblock_header_size + payload_size + pad_size(payload_size, miniblock_alignment);
  CUDF_EXPECTS(expected_chunk_size == chunk.buffer_size, "Invalid Lance MiniBlock payload size");
  return miniblock_chunk_payload{chunk.buffer_offset + miniblock_header_size, payload_size};
}

struct selected_miniblock_payload {
  miniblock_chunk_payload payload;
  std::size_t raw_size{};
  std::size_t output_offset{};
};

struct selected_miniblock_chunk {
  miniblock_chunk_info chunk;
  std::size_t raw_size{};
  std::size_t chunk_idx{};
};

struct sparse_miniblock_read {
  std::uint8_t const* header{};
  std::uint8_t* output{};
  std::size_t raw_size{};
  std::uint64_t buffer_size{};
  compression_type compression{};
};

struct sparse_page_copy {
  std::uint8_t const* page_values{};
  std::uint8_t* output{};
  std::size_t row_map_idx{};
  std::size_t type_width{};
};

struct sparse_flat_page_copy {
  std::uint8_t const* page_values{};
  std::uint8_t* output{};
  std::size_t row_map_offset{};
  size_type num_rows{};
  std::size_t type_width{};
};

struct dense_miniblock_read {
  miniblock_chunk_info chunk;
  std::uint8_t const* page_data{};
  std::uint8_t* output{};
  std::uint64_t page_data_size{};
  std::size_t header_offset{};
  std::size_t raw_size{};
  std::size_t output_offset{};
  compression_type compression{};
};

struct dense_page_read {
  lance_page_info const* page{};
  data_type type{type_id::EMPTY};
  std::uint8_t* output{};
};

void append_sparse_miniblock_reads(
  std::vector<sparse_miniblock_read> const& reads,
  std::vector<device_span<std::uint8_t const>>& inputs,
  std::vector<device_span<std::uint8_t>>& outputs,
  std::vector<std::size_t>& output_sizes,
  std::size_t& total_raw_size,
  std::size_t& max_raw_size,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  if (reads.empty()) { return; }

  static_cast<void>(mr);

  for (auto const& read : reads) {
    CUDF_EXPECTS(read.buffer_size >= miniblock_alignment, "Invalid Lance MiniBlock payload size");
    auto const* payload = read.header + miniblock_alignment;
    auto const payload_size =
      static_cast<std::size_t>(read.buffer_size - miniblock_alignment);
    if (read.compression == compression_type::NONE) {
      CUDF_EXPECTS(payload_size >= read.raw_size,
                   "Uncompressed Lance MiniBlock has unexpected payload size");
      CUDF_CUDA_TRY(cudaMemcpyAsync(read.output,
                                    payload,
                                    read.raw_size,
                                    cudaMemcpyDeviceToDevice,
                                    stream.value()));
      continue;
    }

    CUDF_EXPECTS(read.compression == compression_type::ZSTD,
                 "Unsupported Lance page compression");
    inputs.push_back(device_span<std::uint8_t const>{payload, payload_size});
    outputs.push_back(device_span<std::uint8_t>{read.output, read.raw_size});
    output_sizes.push_back(read.raw_size);
    total_raw_size += read.raw_size;
    max_raw_size = std::max(max_raw_size, read.raw_size);
  }
}

void read_dense_page_values_into(datasource* source,
                                 std::vector<dense_page_read> const& page_reads,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  std::vector<lance_page_info> pages;
  pages.reserve(page_reads.size());
  std::transform(page_reads.begin(),
                 page_reads.end(),
                 std::back_inserter(pages),
                 [](auto const& read) { return *read.page; });
  auto page_buffers = read_page_data_buffers(source, pages, stream, mr);
  auto chunks_by_page = read_miniblock_chunks(source, pages);

  std::vector<device_span<std::uint8_t const>> inputs;
  std::vector<device_span<std::uint8_t>> outputs;
  std::vector<std::size_t> output_sizes;
  std::vector<dense_miniblock_read> reads;
  std::vector<std::uint8_t const*> header_ptrs;
  std::size_t total_raw_size = 0;
  std::size_t max_raw_size   = 0;

  for (std::size_t page_idx = 0; page_idx < page_reads.size(); ++page_idx) {
    auto const& page_read = page_reads[page_idx];
    auto const type_width = cudf::size_of(page_read.type);
    auto const& page = pages[page_idx];
    CUDF_EXPECTS(page.layout.bits_per_value == static_cast<std::uint64_t>(type_width * 8),
                 "Lance page encoding does not match schema type width");

    auto const& chunks = chunks_by_page[page_idx];
    auto const* page_data = page_buffers.pages[page_idx];

    for (auto const& chunk : chunks) {
      CUDF_EXPECTS(chunk.num_rows <=
                     static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) /
                       type_width,
                   "Lance MiniBlock is too large to read");
      auto const raw_size = static_cast<std::size_t>(chunk.num_rows) * type_width;
      auto const output_offset =
        static_cast<std::size_t>(page.priority + chunk.row_begin) * type_width;
      auto const header_offset = chunk.buffer_offset - page.buffer_offsets[1];
      CUDF_EXPECTS(header_offset <= page.buffer_sizes[1] && chunk.buffer_size <= page.buffer_sizes[1] &&
                     header_offset <= page.buffer_sizes[1] - chunk.buffer_size,
                   "Lance MiniBlock is outside the page data buffer");

      header_ptrs.push_back(page_data + static_cast<std::size_t>(header_offset));
      reads.push_back(dense_miniblock_read{chunk,
                                           page_data,
                                           page_read.output,
                                           page.buffer_sizes[1],
                                           static_cast<std::size_t>(header_offset),
                                           raw_size,
                                           output_offset,
                                           page.layout.compression});
    }
  }

  if (!reads.empty()) {
    auto device_header_ptrs = cudf::detail::make_device_uvector(header_ptrs, stream, mr);
    rmm::device_uvector<lance_miniblock_header> device_headers(reads.size(), stream, mr);
    decode_lance_miniblock_headers(
      device_header_ptrs.data(), device_headers.data(), device_headers.size(), stream);
    auto const headers = cudf::detail::make_host_vector(
      device_span<lance_miniblock_header const>{device_headers.data(), device_headers.size()},
      stream);

    for (std::size_t idx = 0; idx < reads.size(); ++idx) {
      auto const& read   = reads[idx];
      auto const& header = headers[idx];
      CUDF_EXPECTS(header.num_levels == 0,
                   "Lance reader does not yet support repetition/definition levels");
      auto const payload_size = static_cast<std::size_t>(header.payload_size);
      auto const expected_chunk_size =
        miniblock_alignment + payload_size + pad_size(payload_size, miniblock_alignment);
      CUDF_EXPECTS(expected_chunk_size == read.chunk.buffer_size,
                   "Invalid Lance MiniBlock payload size");
      auto const payload_offset = read.header_offset + miniblock_alignment;
      CUDF_EXPECTS(payload_offset <= read.page_data_size &&
                     payload_size <= read.page_data_size - payload_offset,
                   "Lance MiniBlock payload is outside the page data buffer");

      if (read.compression == compression_type::NONE) {
        CUDF_EXPECTS(payload_size == read.raw_size,
                     "Uncompressed Lance MiniBlock has unexpected payload size");
        CUDF_CUDA_TRY(cudaMemcpyAsync(read.output + read.output_offset,
                                      read.page_data + payload_offset,
                                      read.raw_size,
                                      cudaMemcpyDeviceToDevice,
                                      stream.value()));
        continue;
      }

      CUDF_EXPECTS(read.compression == compression_type::ZSTD, "Unsupported Lance page compression");
      inputs.push_back(device_span<std::uint8_t const>{read.page_data + payload_offset,
                                                       payload_size});
      outputs.push_back(device_span<std::uint8_t>{read.output + read.output_offset, read.raw_size});
      output_sizes.push_back(read.raw_size);
      total_raw_size += read.raw_size;
      max_raw_size = std::max(max_raw_size, read.raw_size);
    }
  }

  if (!inputs.empty()) {
    decompress_zstd_device_batch_into(
      inputs, outputs, output_sizes, max_raw_size, total_raw_size, stream);
  }
}

void read_dense_pages_values_into(datasource* source,
                                  std::vector<lance_page_info> const& pages,
                                  data_type type,
                                  std::uint8_t* output,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  std::vector<dense_page_read> page_reads;
  page_reads.reserve(pages.size());
  for (auto const& page : pages) {
    page_reads.push_back(dense_page_read{&page, type, output});
  }
  read_dense_page_values_into(source, page_reads, stream, mr);
}

std::vector<page_metadata> write_column_pages(
  std::vector<std::uint8_t>& metadata_section,
  std::uint64_t metadata_begin,
  std::vector<pending_miniblock_chunk> const& chunks,
  std::vector<pending_lance_page> const& pending_pages,
  std::size_t type_width,
  compression_type compression,
  std::vector<lance_pack_chunk>& pack_chunks,
  std::uint64_t& data_size)
{
  std::vector<page_metadata> pages;
  pages.reserve(pending_pages.size());
  static constexpr auto miniblock_header_size = sizeof(std::uint16_t) + sizeof(std::uint32_t);
  static constexpr auto miniblock_header_pad  = miniblock_alignment - miniblock_header_size;
  static_assert(miniblock_header_size + miniblock_header_pad == miniblock_alignment);

  for (auto const& pending_page : pending_pages) {
    CUDF_EXPECTS(pending_page.chunk_begin <= chunks.size() &&
                   pending_page.chunk_count <= chunks.size() - pending_page.chunk_begin,
                 "Invalid Lance pending page chunk range");

    std::vector<std::uint8_t> metadata_buffer;
    metadata_buffer.reserve(pending_page.chunk_count * sizeof(std::uint32_t));
    std::uint64_t miniblock_buffer_size = 0;
    auto const chunk_end = pending_page.chunk_begin + pending_page.chunk_count;
    for (auto chunk_idx = pending_page.chunk_begin; chunk_idx < chunk_end; ++chunk_idx) {
      auto const& chunk = chunks[chunk_idx];
      append_miniblock_metadata(metadata_buffer, chunk.buffer_size, chunk.log_num_values);
      miniblock_buffer_size += chunk.buffer_size;
    }

    page_metadata page;
    page.length   = pending_page.num_rows;
    page.priority = pending_page.row_begin;
    page.encoding = make_page_encoding(
      type_width * 8, static_cast<std::uint64_t>(pending_page.num_rows), compression);

    page.buffer_offsets.push_back(metadata_begin + metadata_section.size());
    page.buffer_sizes.push_back(metadata_buffer.size());
    append_aligned_host_buffer(metadata_section, metadata_buffer);

    CUDF_EXPECTS(data_size <= std::numeric_limits<std::size_t>::max(),
                 "Lance data buffer is too large");
    auto const page_data_begin = static_cast<std::size_t>(data_size);
    page.buffer_offsets.push_back(page_data_begin);
    page.buffer_sizes.push_back(miniblock_buffer_size);

    std::size_t page_data_offset = page_data_begin;
    for (std::size_t idx = 0; idx < pending_page.chunk_count; ++idx) {
      auto const& chunk = chunks[pending_page.chunk_begin + idx];
      pack_chunks.push_back(lance_pack_chunk{chunk.payload, chunk.payload_size, page_data_offset});
      page_data_offset += miniblock_header_size + miniblock_header_pad;
      page_data_offset += chunk.payload_size + pad_size(chunk.payload_size, miniblock_alignment);
    }
    CUDF_EXPECTS(page_data_offset == page_data_begin + miniblock_buffer_size,
                 "Packed Lance MiniBlock size mismatch");
    data_size += miniblock_buffer_size + pad_size(miniblock_buffer_size);
    pages.push_back(std::move(page));
  }

  return pages;
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
  auto const miniblock_rows_in =
    options.get_max_rows_per_miniblock().value_or(writer_default_rows_per_miniblock);
  CUDF_EXPECTS(page_rows_in > 0, "max_rows_per_page must be greater than zero");

  std::vector<column_metadata> columns(table.num_columns());
  std::vector<pending_lance_column> pending_columns(table.num_columns());
  for (size_type col_idx = 0; col_idx < table.num_columns(); ++col_idx) {
    auto const column = table.column(col_idx);
    auto& pending_column = pending_columns[static_cast<std::size_t>(col_idx)];
    pending_column.type_width = cudf::size_of(column.type());

    for (size_type row = 0; row < table.num_rows(); row += page_rows_in) {
      auto const rows = std::min(page_rows_in, table.num_rows() - row);
      auto const chunk_begin = pending_column.chunks.size();
      append_page_chunks(pending_column.chunks, column, row, rows, miniblock_rows_in);
      pending_column.pages.push_back(pending_lance_page{
        row, rows, chunk_begin, pending_column.chunks.size() - chunk_begin});
    }
  }

  std::optional<rmm::device_uvector<std::uint8_t>> compressed_payloads;
  if (compression == compression_type::ZSTD) {
    std::vector<pending_miniblock_chunk*> chunks_to_compress;
    for (auto& pending_column : pending_columns) {
      chunks_to_compress.reserve(chunks_to_compress.size() + pending_column.chunks.size());
      for (auto& chunk : pending_column.chunks) {
        chunks_to_compress.push_back(&chunk);
      }
    }
    compressed_payloads.emplace(compress_zstd_miniblocks(chunks_to_compress, stream));
  }

  std::size_t total_chunks = 0;
  for (auto const& pending_column : pending_columns) {
    total_chunks += pending_column.chunks.size();
  }
  std::vector<lance_pack_chunk> pack_chunks;
  pack_chunks.reserve(total_chunks);
  std::uint64_t data_size = 0;
  auto const metadata_begin = sink->bytes_written();
  std::vector<std::uint8_t> metadata_section;
  for (size_type col_idx = 0; col_idx < table.num_columns(); ++col_idx) {
    auto& pending_column = pending_columns[static_cast<std::size_t>(col_idx)];
    finalize_miniblock_chunks(pending_column.chunks);
    columns[col_idx].pages =
      write_column_pages(metadata_section,
                         metadata_begin,
                         pending_column.chunks,
                         pending_column.pages,
                         pending_column.type_width,
                         compression,
                         pack_chunks,
                         data_size);
  }

  write_host_buffer(sink, metadata_section.data(), metadata_section.size());
  auto const data_begin = sink->bytes_written();
  for (auto& column : columns) {
    for (auto& page : column.pages) {
      CUDF_EXPECTS(page.buffer_offsets.size() == 2,
                   "Invalid Lance page buffer offsets while writing");
      page.buffer_offsets[1] += data_begin;
    }
  }

  CUDF_EXPECTS(data_size <= std::numeric_limits<std::size_t>::max(),
               "Lance data buffer is too large");
  rmm::device_uvector<std::uint8_t> data(static_cast<std::size_t>(data_size), stream);
  if (data.size() > 0) {
    CUDF_CUDA_TRY(cudaMemsetAsync(data.data(), lance_pad_byte, data.size(), stream.value()));
  }
  if (!pack_chunks.empty()) {
    rmm::device_uvector<lance_pack_chunk> device_pack_chunks(pack_chunks.size(), stream);
    CUDF_CUDA_TRY(cudaMemcpyAsync(device_pack_chunks.data(),
                                  pack_chunks.data(),
                                  pack_chunks.size() * sizeof(lance_pack_chunk),
                                  cudaMemcpyHostToDevice,
                                  stream.value()));
    pack_lance_miniblocks(device_pack_chunks.data(), device_pack_chunks.size(), data.data(), stream);
  }
  write_device_buffer(sink, data.data(), data.size(), stream);

  return columns;
}

void validate_options(lance_writer_options const& options)
{
  auto const& table = options.get_table();
  CUDF_EXPECTS(table.num_columns() > 0, "Lance writer requires at least one column");
  CUDF_EXPECTS(options.get_compression() == compression_type::NONE ||
                 options.get_compression() == compression_type::ZSTD,
               "Lance writer currently supports NONE and ZSTD compression");
  if (options.get_max_rows_per_page().has_value()) {
    CUDF_EXPECTS(options.get_max_rows_per_page().value() > 0,
                 "max_rows_per_page must be greater than zero");
  }
  if (options.get_max_rows_per_miniblock().has_value()) {
    auto const rows = options.get_max_rows_per_miniblock().value();
    CUDF_EXPECTS(rows >= 2, "max_rows_per_miniblock must be at least 2");
    CUDF_EXPECTS(rows <= static_cast<size_type>(max_rows_per_miniblock),
                 "max_rows_per_miniblock exceeds Lance MiniBlock metadata limits");
    CUDF_EXPECTS(std::has_single_bit(static_cast<std::uint32_t>(rows)),
                 "max_rows_per_miniblock must be a power of two");
  }

  for (size_type idx = 0; idx < table.num_columns(); ++idx) {
    auto const column = table.column(idx);
    CUDF_EXPECTS(is_supported_lance_type(column.type()),
                 "Unsupported Lance writer type for column " + std::to_string(idx) + ": " +
                   cudf::type_to_name(column.type()));
    CUDF_EXPECTS(!column.has_nulls(),
                 "Lance writer currently supports only non-null fixed-width columns");
  }
}

std::vector<size_type> selected_rows(lance_file_info const& file_info,
                                     lance_reader_options const& options)
{
  std::vector<size_type> rows;
  if (options.has_row_selection()) {
    rows = options.get_rows();
    CUDF_EXPECTS(rows.size() <= static_cast<std::size_t>(std::numeric_limits<size_type>::max()),
                 "Lance row selection contains too many rows for cuDF");
  } else {
    CUDF_EXPECTS(file_info.num_rows <=
                   static_cast<std::uint64_t>(std::numeric_limits<size_type>::max()),
                 "Lance file contains too many rows for cuDF");
    rows.resize(static_cast<std::size_t>(file_info.num_rows));
    std::iota(rows.begin(), rows.end(), 0);
  }

  for (auto row : rows) {
    CUDF_EXPECTS(row >= 0 && static_cast<std::uint64_t>(row) < file_info.num_rows,
                 "Lance row selection contains an out-of-bounds row");
  }
  return rows;
}

std::vector<size_type> selected_columns(lance_file_info const& file_info,
                                        lance_reader_options const& options)
{
  std::vector<size_type> columns;
  auto const& requested = options.get_columns();
  if (requested.empty()) {
    CUDF_EXPECTS(file_info.fields.size() <=
                   static_cast<std::size_t>(std::numeric_limits<size_type>::max()),
                 "Lance file contains too many columns for cuDF");
    for (auto const& field : file_info.fields) {
      CUDF_EXPECTS(field.is_supported(),
                   "Lance column requires explicit projection or unsupported column handling: " +
                     field.name + " (" + field.unsupported_reason + ")");
    }
    columns.resize(file_info.fields.size());
    std::iota(columns.begin(), columns.end(), 0);
    return columns;
  }

  constexpr std::size_t linear_column_lookup_max_fields = 32;
  if (file_info.fields.size() <= linear_column_lookup_max_fields) {
    columns.reserve(requested.size());
    for (auto const& name : requested) {
      auto found = std::find_if(file_info.fields.begin(),
                                file_info.fields.end(),
                                [&](auto const& field) { return field.name == name; });
      CUDF_EXPECTS(found != file_info.fields.end(), "Requested Lance column not found: " + name);
      auto const column_idx =
        static_cast<std::size_t>(std::distance(file_info.fields.begin(), found));
      CUDF_EXPECTS(column_idx <= static_cast<std::size_t>(std::numeric_limits<size_type>::max()),
                   "Lance file contains too many columns for cuDF");
      CUDF_EXPECTS(found->is_supported(),
                   "Requested Lance column is not supported: " + found->name + " (" +
                     found->unsupported_reason + ")");
      columns.push_back(static_cast<size_type>(column_idx));
    }
    return columns;
  }

  std::unordered_map<std::string, size_type> field_indices;
  for (std::size_t idx = 0; idx < file_info.fields.size(); ++idx) {
    CUDF_EXPECTS(idx <= static_cast<std::size_t>(std::numeric_limits<size_type>::max()),
                 "Lance file contains too many columns for cuDF");
    field_indices.emplace(file_info.fields[idx].name, static_cast<size_type>(idx));
  }

  columns.reserve(requested.size());
  for (auto const& name : requested) {
    auto found = field_indices.find(name);
    CUDF_EXPECTS(found != field_indices.end(), "Requested Lance column not found: " + name);
    auto const& field = file_info.fields[found->second];
    CUDF_EXPECTS(field.is_supported(),
                 "Requested Lance column is not supported: " + field.name + " (" +
                   field.unsupported_reason + ")");
    columns.push_back(found->second);
  }
  return columns;
}

std::size_t find_page_for_row(std::vector<lance_page_info> const& pages, size_type row)
{
  auto const row_u64 = static_cast<std::uint64_t>(row);
  auto const default_idx = static_cast<std::size_t>(row_u64 >> default_rows_per_page_log);
  if (default_idx < pages.size()) {
    auto const& page = pages[default_idx];
    if (row_u64 >= page.priority && row_u64 < page.priority + page.length) {
      return default_idx;
    }
  }

  auto const found = std::upper_bound(
    pages.begin(), pages.end(), row_u64, [](auto value, auto const& page) {
      return value < page.priority;
    });
  if (found != pages.begin()) {
    auto const idx  = static_cast<std::size_t>(std::distance(pages.begin(), found - 1));
    auto const& page = pages[idx];
    if (row_u64 < page.priority + page.length) { return idx; }
  }
  CUDF_FAIL("Lance row selection references a row without a data page");
}

bool should_use_dense_for_sparse_selection(datasource* source,
                                           lance_file_info const& file_info,
                                           std::vector<size_type> const& columns,
                                           std::vector<size_type> const& rows)
{
  if (columns.size() < 2 || rows.empty()) { return false; }

  auto const& column_info = file_info.columns[columns.front()];
  if (rows.size() < column_info.pages.size() * sparse_dense_min_rows_per_page) {
    return false;
  }

  std::vector<std::uint64_t> page_miniblock_begins(column_info.pages.size());
  std::uint64_t total_miniblocks = 0;
  for (std::size_t page_idx = 0; page_idx < column_info.pages.size(); ++page_idx) {
    auto const& page                 = column_info.pages[page_idx];
    page_miniblock_begins[page_idx] = total_miniblocks;
    auto const page_miniblock_count = (page.length + default_rows_per_miniblock - 1) /
                                      default_rows_per_miniblock;
    total_miniblocks += page_miniblock_count;
  }
  if (total_miniblocks == 0) { return false; }

  auto const dense_miniblock_threshold = (total_miniblocks + 1) / 2;
  if (static_cast<std::uint64_t>(rows.size()) < dense_miniblock_threshold) { return false; }

  auto const [min_row, max_row]        = std::minmax_element(rows.begin(), rows.end());
  auto const row_span =
    static_cast<std::uint64_t>(*max_row) - static_cast<std::uint64_t>(*min_row);
  auto const max_span_miniblocks = (row_span / default_rows_per_miniblock) + 1;
  if (max_span_miniblocks < dense_miniblock_threshold) { return false; }

  auto chunks_by_page = read_miniblock_chunks(source, column_info.pages);
  total_miniblocks    = 0;
  auto total_data_buffer_size = std::uint64_t{0};
  for (std::size_t page_idx = 0; page_idx < chunks_by_page.size(); ++page_idx) {
    page_miniblock_begins[page_idx] = total_miniblocks;
    total_miniblocks += chunks_by_page[page_idx].size();
    for (auto const& chunk : chunks_by_page[page_idx]) {
      total_data_buffer_size += chunk.buffer_size;
    }
  }
  if (total_miniblocks == 0 || total_data_buffer_size == 0) { return false; }

  auto const dense_actual_miniblock_threshold = (total_miniblocks + 1) / 2;
  if (static_cast<std::uint64_t>(rows.size()) < dense_actual_miniblock_threshold) {
    return false;
  }

  CUDF_EXPECTS(total_miniblocks <=
                 static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()),
               "Lance file contains too many MiniBlocks");
  std::vector<std::uint8_t> selected_miniblocks(static_cast<std::size_t>(total_miniblocks), 0);
  std::uint64_t selected_miniblock_count = 0;
  auto selected_data_buffer_size         = std::uint64_t{0};
  for (auto row : rows) {
    auto const page_idx    = find_page_for_row(column_info.pages, row);
    auto const& page       = column_info.pages[page_idx];
    auto const row_in_page = static_cast<std::uint64_t>(row) - page.priority;
    auto const& chunks     = chunks_by_page[page_idx];
    auto const found = std::upper_bound(
      chunks.begin(), chunks.end(), row_in_page, [](auto value, auto const& chunk) {
        return value < chunk.row_begin;
      });
    CUDF_EXPECTS(found != chunks.begin(),
                 "Lance row selection references a row without a MiniBlock chunk");
    auto const chunk_idx = static_cast<std::size_t>(std::distance(chunks.begin(), found - 1));
    CUDF_EXPECTS(row_in_page < chunks[chunk_idx].row_begin + chunks[chunk_idx].num_rows,
                 "Lance row selection references a row without a MiniBlock chunk");
    auto const miniblock_idx = page_miniblock_begins[page_idx] + chunk_idx;
    auto& selected = selected_miniblocks[static_cast<std::size_t>(miniblock_idx)];
    if (selected == 0) {
      selected = 1;
      selected_data_buffer_size += chunks[chunk_idx].buffer_size;
      ++selected_miniblock_count;
      if (selected_miniblock_count >= dense_actual_miniblock_threshold &&
          selected_data_buffer_size >= (total_data_buffer_size + 1) / 2) {
        return true;
      }
    }
  }

  return false;
}

bool can_batch_sparse_zstd_columns(lance_file_info const& file_info,
                                   std::vector<size_type> const& columns)
{
  for (auto column_idx : columns) {
    for (auto const& page : file_info.columns[column_idx].pages) {
      if (page.layout.compression != compression_type::ZSTD) { return false; }
    }
  }
  return true;
}

bool should_batch_single_column_sparse_selection(lance_file_info const& file_info,
                                                 size_type column_idx,
                                                 std::vector<size_type> const& rows)
{
  if (rows.size() < sparse_header_batch_min_reads) { return false; }

  auto const [min_row, max_row] = std::minmax_element(rows.begin(), rows.end());
  auto const row_span =
    static_cast<std::uint64_t>(*max_row) - static_cast<std::uint64_t>(*min_row);
  if (row_span < (sparse_header_batch_min_reads - 1) * default_rows_per_page) {
    return false;
  }

  auto const& column_info = file_info.columns[column_idx];
  std::vector<std::uint8_t> selected_pages(column_info.pages.size(), 0);
  std::size_t selected_page_count = 0;
  for (auto row : rows) {
    auto const page_idx = find_page_for_row(column_info.pages, row);
    if (selected_pages[page_idx] == 0) {
      selected_pages[page_idx] = 1;
      ++selected_page_count;
      if (selected_page_count >= sparse_header_batch_min_reads) { return true; }
    }
  }

  return false;
}

bool have_same_page_rows(lance_column_info const& lhs, lance_column_info const& rhs)
{
  if (lhs.pages.size() != rhs.pages.size()) { return false; }
  for (std::size_t idx = 0; idx < lhs.pages.size(); ++idx) {
    if (lhs.pages[idx].priority != rhs.pages[idx].priority ||
        lhs.pages[idx].length != rhs.pages[idx].length) {
      return false;
    }
  }
  return true;
}

std::size_t find_miniblock_chunk_for_row(std::vector<miniblock_chunk_info> const& chunks,
                                         size_type row,
                                         bool prefer_uniform_stride_lookup = false)
{
  auto const row_u64 = static_cast<std::uint64_t>(row);

  if (prefer_uniform_stride_lookup && !chunks.empty()) {
    auto const chunk_rows = chunks.front().num_rows;
    auto const chunk_idx =
      chunk_rows == 0
        ? chunks.size()
        : static_cast<std::size_t>(
            std::has_single_bit(chunk_rows)
              ? row_u64 >> std::countr_zero(chunk_rows)
              : row_u64 / chunk_rows);
    if (chunk_idx < chunks.size()) {
      auto const& chunk = chunks[chunk_idx];
      if (row_u64 >= chunk.row_begin && row_u64 < chunk.row_begin + chunk.num_rows) {
        return chunk_idx;
      }
    }
  }

  auto const lance_default_idx =
    static_cast<std::size_t>(row_u64 >> default_rows_per_miniblock_log);
  if (lance_default_idx < chunks.size()) {
    auto const& chunk = chunks[lance_default_idx];
    if (row_u64 >= chunk.row_begin && row_u64 < chunk.row_begin + chunk.num_rows) {
      return lance_default_idx;
    }
  }

  auto const found = std::upper_bound(
    chunks.begin(), chunks.end(), row_u64, [](auto value, auto const& chunk) {
      return value < chunk.row_begin;
    });
  if (found != chunks.begin()) {
    auto const idx   = static_cast<std::size_t>(std::distance(chunks.begin(), found - 1));
    auto const& chunk = chunks[idx];
    if (row_u64 < chunk.row_begin + chunk.num_rows) { return idx; }
  }
  CUDF_FAIL("Lance row selection references a row without a MiniBlock chunk");
}

std::unique_ptr<column> read_lance_column(datasource* source,
                                          lance_field_info const& field,
                                          lance_column_info const& column_info,
                                          std::uint64_t num_rows,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(field.type.has_value(),
               "Unsupported Lance column selected: " + field.name + " (" +
                 field.unsupported_reason + ")");
  auto const field_type = *field.type;
  CUDF_EXPECTS(num_rows <= static_cast<std::uint64_t>(std::numeric_limits<size_type>::max()),
               "Lance file contains too many rows for cuDF");
  if (num_rows == 0) {
    return cudf::make_fixed_width_column(
      field_type, 0, mask_state::UNALLOCATED, stream, mr);
  }

  auto output = cudf::make_fixed_width_column(
    field_type, static_cast<size_type>(num_rows), mask_state::UNALLOCATED, stream, mr);
  auto* output_data = output->mutable_view().head<std::uint8_t>();

  std::uint64_t next_row = 0;
  for (auto const& page : column_info.pages) {
    CUDF_EXPECTS(page.priority == next_row,
                 "Lance pages are not contiguous for full-column read");
    CUDF_EXPECTS(page.length <=
                   static_cast<std::uint64_t>(std::numeric_limits<size_type>::max()),
                 "Lance page contains too many rows for cuDF");

    next_row += page.length;
  }

  CUDF_EXPECTS(next_row == num_rows,
               "Lance page lengths do not match file row count");
  read_dense_pages_values_into(source, column_info.pages, field_type, output_data, stream, mr);
  return output;
}

std::vector<std::unique_ptr<column>> read_lance_columns(datasource* source,
                                                        lance_file_info const& file_info,
                                                        std::vector<size_type> const& columns,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(file_info.num_rows <=
                 static_cast<std::uint64_t>(std::numeric_limits<size_type>::max()),
               "Lance file contains too many rows for cuDF");

  std::vector<std::unique_ptr<column>> output_columns;
  output_columns.reserve(columns.size());

  if (file_info.num_rows == 0) {
    for (auto column_idx : columns) {
      auto const& field = file_info.fields[column_idx];
      CUDF_EXPECTS(field.type.has_value(),
                   "Unsupported Lance column selected: " + field.name + " (" +
                     field.unsupported_reason + ")");
      output_columns.push_back(
        cudf::make_fixed_width_column(*field.type, 0, mask_state::UNALLOCATED, stream, mr));
    }
    return output_columns;
  }

  std::vector<dense_page_read> page_reads;
  for (auto column_idx : columns) {
    auto const& field       = file_info.fields[column_idx];
    auto const& column_info = file_info.columns[column_idx];
    CUDF_EXPECTS(field.type.has_value(),
                 "Unsupported Lance column selected: " + field.name + " (" +
                   field.unsupported_reason + ")");
    auto const field_type = *field.type;
    auto output = cudf::make_fixed_width_column(field_type,
                                                static_cast<size_type>(file_info.num_rows),
                                                mask_state::UNALLOCATED,
                                                stream,
                                                mr);
    auto* output_data = output->mutable_view().head<std::uint8_t>();

    std::uint64_t next_row = 0;
    for (auto const& page : column_info.pages) {
      CUDF_EXPECTS(page.priority == next_row,
                   "Lance pages are not contiguous for full-column read");
      CUDF_EXPECTS(page.length <=
                     static_cast<std::uint64_t>(std::numeric_limits<size_type>::max()),
                   "Lance page contains too many rows for cuDF");

      next_row += page.length;
      page_reads.push_back(dense_page_read{&page, field_type, output_data});
    }

    CUDF_EXPECTS(next_row == file_info.num_rows,
                 "Lance page lengths do not match file row count");
    output_columns.push_back(std::move(output));
  }

  read_dense_page_values_into(source, page_reads, stream, mr);
  return output_columns;
}

std::unique_ptr<column> read_lance_column(datasource* source,
                                          lance_field_info const& field,
                                          lance_column_info const& column_info,
                                          std::vector<size_type> const& rows,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(field.type.has_value(),
               "Unsupported Lance column selected: " + field.name + " (" +
                 field.unsupported_reason + ")");
  auto const field_type = *field.type;
  auto output = cudf::make_fixed_width_column(
    field_type, static_cast<size_type>(rows.size()), mask_state::UNALLOCATED, stream, mr);
  if (rows.empty()) { return output; }

  std::vector<std::vector<std::pair<size_type, size_type>>> rows_by_page(column_info.pages.size());
  for (size_type output_row = 0; output_row < static_cast<size_type>(rows.size()); ++output_row) {
    auto const page_idx = find_page_for_row(column_info.pages, rows[output_row]);
    auto const& page    = column_info.pages[page_idx];
    auto const row_in_page =
      static_cast<size_type>(static_cast<std::uint64_t>(rows[output_row]) - page.priority);
    rows_by_page[page_idx].emplace_back(output_row, row_in_page);
  }

  auto const type_width = cudf::size_of(field_type);
  auto* output_data = output->mutable_view().head<std::uint8_t>();

  std::uint64_t total_data_buffer_size    = 0;
  std::uint64_t selected_data_buffer_size = 0;
  std::vector<std::vector<miniblock_chunk_info>> chunks_by_page(column_info.pages.size());
  for (auto const& page : column_info.pages) {
    total_data_buffer_size += page.buffer_sizes[1];
  }
  auto const touched_page_indices =
    read_touched_miniblock_chunks(source, column_info.pages, rows_by_page, chunks_by_page);
  for (auto page_idx : touched_page_indices) {

    auto const& chunks       = chunks_by_page[page_idx];
    std::vector<bool> selected_chunks(chunks.size(), false);
    auto const prefer_uniform_stride_lookup =
      rows_by_page[page_idx].size() >= sparse_uniform_miniblock_lookup_min_rows;
    for (auto const& [_, row_in_page] : rows_by_page[page_idx]) {
      selected_chunks[find_miniblock_chunk_for_row(
        chunks, row_in_page, prefer_uniform_stride_lookup)] = true;
    }
    for (std::size_t chunk_idx = 0; chunk_idx < chunks.size(); ++chunk_idx) {
      if (selected_chunks[chunk_idx]) { selected_data_buffer_size += chunks[chunk_idx].buffer_size; }
    }
  }

  // Random row selections can touch most MiniBlocks. Once that happens, the coalesced dense path is
  // cheaper than issuing many small sparse reads and then gathering out of page-sized buffers.
  if (selected_data_buffer_size >= (total_data_buffer_size + 1) / 2) {
    std::uint64_t dense_rows = 0;
    for (auto const& page : column_info.pages) {
      dense_rows += page.length;
    }

    auto dense_column = read_lance_column(source, field, column_info, dense_rows, stream, mr);
    auto source_map   = cudf::detail::make_device_uvector(rows, stream, mr);
    std::vector<size_type> target_rows(rows.size());
    std::iota(target_rows.begin(), target_rows.end(), 0);
    auto target_map = cudf::detail::make_device_uvector(target_rows, stream, mr);
    copy_sparse_fixed_width(dense_column->view().head<std::uint8_t>(),
                            output_data,
                            source_map.data(),
                            target_map.data(),
                            static_cast<size_type>(source_map.size()),
                            type_width,
                            stream);
    return output;
  }

  std::vector<rmm::device_uvector<std::uint8_t>> page_value_buffers;
  std::vector<rmm::device_uvector<std::uint8_t>> input_buffers;
  std::vector<std::vector<size_type>> source_rows_by_page;
  std::vector<std::vector<size_type>> target_rows_by_page;
  page_value_buffers.reserve(column_info.pages.size());
  input_buffers.reserve(column_info.pages.size());
  source_rows_by_page.reserve(column_info.pages.size());
  target_rows_by_page.reserve(column_info.pages.size());

  std::vector<device_span<std::uint8_t const>> inputs;
  std::vector<device_span<std::uint8_t>> outputs;
  std::vector<std::size_t> output_sizes;
  std::size_t total_raw_size = 0;
  std::size_t max_raw_size   = 0;

  for (auto page_idx : touched_page_indices) {
    auto const& page = column_info.pages[page_idx];
    CUDF_EXPECTS(page.length <= static_cast<std::uint64_t>(std::numeric_limits<size_type>::max()),
                 "Lance page contains too many rows for cuDF");
    CUDF_EXPECTS(page.length <=
                   static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) /
                     type_width,
                 "Lance page is too large to read");
    auto const& chunks = chunks_by_page[page_idx];
    std::vector<std::vector<std::pair<size_type, size_type>>> rows_by_chunk(chunks.size());
    auto const prefer_uniform_stride_lookup =
      rows_by_page[page_idx].size() >= sparse_uniform_miniblock_lookup_min_rows;
    for (auto const& [output_row, row_in_page] : rows_by_page[page_idx]) {
      auto const chunk_idx =
        find_miniblock_chunk_for_row(chunks, row_in_page, prefer_uniform_stride_lookup);
      auto const& chunk    = chunks[chunk_idx];
      auto const row_in_chunk =
        static_cast<size_type>(static_cast<std::uint64_t>(row_in_page) - chunk.row_begin);
      rows_by_chunk[chunk_idx].emplace_back(output_row, row_in_chunk);
    }

    std::vector<selected_miniblock_payload> selected;
    selected.reserve(chunks.size());
    std::uint64_t selected_buffer_size = 0;
    std::size_t total_payload_size     = 0;
    for (std::size_t chunk_idx = 0; chunk_idx < chunks.size(); ++chunk_idx) {
      if (rows_by_chunk[chunk_idx].empty()) { continue; }
      auto const& chunk = chunks[chunk_idx];
      CUDF_EXPECTS(chunk.num_rows <=
                     static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) /
                       type_width,
                   "Lance MiniBlock is too large to read");
      auto const raw_size = static_cast<std::size_t>(chunk.num_rows) * type_width;
      auto const payload  = read_miniblock_chunk_payload(source, chunk);
      if (page.layout.compression == compression_type::NONE) {
        CUDF_EXPECTS(payload.payload_size == raw_size,
                     "Uncompressed Lance MiniBlock has unexpected payload size");
      }

      auto const output_offset = static_cast<std::size_t>(chunk.row_begin) * type_width;
      selected.push_back(selected_miniblock_payload{payload, raw_size, output_offset});
      selected_buffer_size += chunk.buffer_size;
      total_payload_size += payload.payload_size;
    }

    auto const page_raw_size = static_cast<std::size_t>(page.length) * type_width;
    page_value_buffers.emplace_back(page_raw_size, stream, mr);
    auto* page_values = page_value_buffers.back().data();

    auto const read_full_page =
      selected_buffer_size >= ((page.buffer_sizes[1] + 1) / 2) || selected.size() == chunks.size();
    std::vector<std::size_t> input_offsets(selected.size());
    if (read_full_page) {
      input_buffers.push_back(
        read_device_bytes(source, page.buffer_offsets[1], page.buffer_sizes[1], stream, mr));
      for (std::size_t idx = 0; idx < selected.size(); ++idx) {
        auto const& payload = selected[idx].payload;
        CUDF_EXPECTS(payload.payload_offset >= page.buffer_offsets[1] &&
                       payload.payload_size <= page.buffer_sizes[1] &&
                       payload.payload_offset - page.buffer_offsets[1] <=
                         page.buffer_sizes[1] - payload.payload_size,
                     "Lance MiniBlock payload is outside the page data buffer");
        input_offsets[idx] =
          static_cast<std::size_t>(payload.payload_offset - page.buffer_offsets[1]);
      }
    } else {
      input_buffers.emplace_back(total_payload_size, stream, mr);
      std::size_t input_offset = 0;
      for (std::size_t idx = 0; idx < selected.size(); ++idx) {
        auto const& payload = selected[idx].payload;
        input_offsets[idx]  = input_offset;
        read_device_bytes_into(source,
                               payload.payload_offset,
                               payload.payload_size,
                               input_buffers.back().data() + input_offset,
                               stream);
        input_offset += payload.payload_size;
      }
    }
    auto const* input_data = input_buffers.back().data();

    if (page.layout.compression == compression_type::NONE) {
      for (std::size_t idx = 0; idx < selected.size(); ++idx) {
        CUDF_CUDA_TRY(cudaMemcpyAsync(page_values + selected[idx].output_offset,
                                      input_data + input_offsets[idx],
                                      selected[idx].raw_size,
                                      cudaMemcpyDeviceToDevice,
                                      stream.value()));
      }
    } else {
      CUDF_EXPECTS(page.layout.compression == compression_type::ZSTD,
                   "Unsupported Lance page compression");
      for (std::size_t idx = 0; idx < selected.size(); ++idx) {
        auto const& payload = selected[idx].payload;
        inputs.push_back(
          device_span<std::uint8_t const>{input_data + input_offsets[idx],
                                          static_cast<std::size_t>(payload.payload_size)});
        outputs.push_back(device_span<std::uint8_t>{page_values + selected[idx].output_offset,
                                                    selected[idx].raw_size});
        output_sizes.push_back(selected[idx].raw_size);
        total_raw_size += selected[idx].raw_size;
        max_raw_size = std::max(max_raw_size, selected[idx].raw_size);
      }
    }

    std::vector<size_type> source_rows;
    std::vector<size_type> target_rows;
    source_rows.reserve(rows_by_page[page_idx].size());
    target_rows.reserve(rows_by_page[page_idx].size());
    for (auto const& [output_row, row_in_page] : rows_by_page[page_idx]) {
      target_rows.push_back(output_row);
      source_rows.push_back(row_in_page);
    }
    source_rows_by_page.push_back(std::move(source_rows));
    target_rows_by_page.push_back(std::move(target_rows));
  }

  if (!inputs.empty()) {
    decompress_zstd_device_batch_into(
      inputs, outputs, output_sizes, max_raw_size, total_raw_size, stream);
  }

  for (std::size_t idx = 0; idx < page_value_buffers.size(); ++idx) {
    auto source_map = cudf::detail::make_device_uvector(source_rows_by_page[idx], stream, mr);
    auto target_map = cudf::detail::make_device_uvector(target_rows_by_page[idx], stream, mr);
    copy_sparse_fixed_width(page_value_buffers[idx].data(),
                            output_data,
                            source_map.data(),
                            target_map.data(),
                            static_cast<size_type>(source_map.size()),
                            type_width,
                            stream);
  }
  return output;
}

std::vector<std::unique_ptr<column>> read_lance_columns(datasource* source,
                                                        lance_file_info const& file_info,
                                                        std::vector<size_type> const& columns,
                                                        std::vector<size_type> const& rows,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<column>> output_columns;
  output_columns.reserve(columns.size());

  if (rows.empty()) {
    for (auto column_idx : columns) {
      auto const& field = file_info.fields[column_idx];
      CUDF_EXPECTS(field.type.has_value(),
                   "Unsupported Lance column selected: " + field.name + " (" +
                     field.unsupported_reason + ")");
      output_columns.push_back(
        cudf::make_fixed_width_column(*field.type, 0, mask_state::UNALLOCATED, stream, mr));
    }
    return output_columns;
  }

  std::size_t total_pages = 0;
  for (auto column_idx : columns) {
    total_pages += file_info.columns[column_idx].pages.size();
  }
  auto const can_share_page_rows = std::all_of(columns.begin() + 1,
                                               columns.end(),
                                               [&](auto column_idx) {
                                                 return have_same_page_rows(
                                                   file_info.columns[columns.front()],
                                                   file_info.columns[column_idx]);
                                               });

  std::vector<rmm::device_uvector<std::uint8_t>> page_value_buffers;
  std::vector<rmm::device_uvector<std::uint8_t>> input_buffers;
  std::vector<std::vector<size_type>> source_rows_by_page;
  std::vector<std::vector<size_type>> target_rows_by_page;
  std::vector<sparse_page_copy> page_copies;
  std::vector<sparse_flat_page_copy> flat_page_copies;
  std::vector<size_type> flat_source_rows;
  std::vector<size_type> flat_target_rows;
  std::vector<lance_sparse_copy_row> single_row_page_copies;
  std::vector<pending_device_read> pending_sparse_device_reads;
  page_value_buffers.reserve(total_pages);
  input_buffers.reserve(total_pages);
  source_rows_by_page.reserve(total_pages);
  target_rows_by_page.reserve(total_pages);
  page_copies.reserve(total_pages);
  flat_page_copies.reserve(total_pages);
  flat_source_rows.reserve(rows.size() * columns.size());
  flat_target_rows.reserve(rows.size() * columns.size());
  single_row_page_copies.reserve(total_pages);
  pending_sparse_device_reads.reserve(total_pages);

  std::vector<device_span<std::uint8_t const>> inputs;
  std::vector<device_span<std::uint8_t>> outputs;
  std::vector<std::size_t> output_sizes;
  std::vector<sparse_miniblock_read> sparse_miniblock_reads;
  std::size_t total_raw_size = 0;
  std::size_t max_raw_size   = 0;

  std::vector<std::vector<std::pair<size_type, size_type>>> shared_rows_by_page;
  std::vector<std::size_t> shared_row_map_indices;
  std::size_t shared_pages_touched = 0;
  if (can_share_page_rows) {
    auto const& pages = file_info.columns[columns.front()].pages;
    shared_rows_by_page.resize(pages.size());
    for (size_type output_row = 0; output_row < static_cast<size_type>(rows.size());
         ++output_row) {
      auto const page_idx = find_page_for_row(pages, rows[output_row]);
      auto const& page    = pages[page_idx];
      auto const row_in_page =
        static_cast<size_type>(static_cast<std::uint64_t>(rows[output_row]) - page.priority);
      shared_rows_by_page[page_idx].emplace_back(output_row, row_in_page);
    }
    for (std::size_t page_idx = 0; page_idx < shared_rows_by_page.size(); ++page_idx) {
      if (shared_rows_by_page[page_idx].empty()) { continue; }
      ++shared_pages_touched;
    }
  }
  auto const use_batched_sparse_headers =
    can_share_page_rows &&
    shared_pages_touched * columns.size() >= sparse_multi_column_header_batch_min_reads;
  if (can_share_page_rows && !use_batched_sparse_headers) {
    shared_row_map_indices.resize(shared_rows_by_page.size(),
                                  std::numeric_limits<std::size_t>::max());
    for (std::size_t page_idx = 0; page_idx < shared_rows_by_page.size(); ++page_idx) {
      if (shared_rows_by_page[page_idx].empty()) { continue; }
      std::vector<size_type> source_rows;
      std::vector<size_type> target_rows;
      source_rows.reserve(shared_rows_by_page[page_idx].size());
      target_rows.reserve(shared_rows_by_page[page_idx].size());
      for (auto const& [output_row, row_in_page] : shared_rows_by_page[page_idx]) {
        target_rows.push_back(output_row);
        source_rows.push_back(row_in_page);
      }
      shared_row_map_indices[page_idx] = source_rows_by_page.size();
      source_rows_by_page.push_back(std::move(source_rows));
      target_rows_by_page.push_back(std::move(target_rows));
    }
  }

  for (auto column_idx : columns) {
    auto const& field       = file_info.fields[column_idx];
    auto const& column_info = file_info.columns[column_idx];
    CUDF_EXPECTS(field.type.has_value(),
                 "Unsupported Lance column selected: " + field.name + " (" +
                   field.unsupported_reason + ")");
    auto const field_type = *field.type;
    auto const type_width = cudf::size_of(field_type);

    std::vector<std::vector<std::pair<size_type, size_type>>> local_rows_by_page;
    auto const* rows_by_page_ptr = &shared_rows_by_page;
    if (!can_share_page_rows) {
      local_rows_by_page.resize(column_info.pages.size());
      for (size_type output_row = 0; output_row < static_cast<size_type>(rows.size());
           ++output_row) {
        auto const page_idx = find_page_for_row(column_info.pages, rows[output_row]);
        auto const& page    = column_info.pages[page_idx];
        auto const row_in_page =
          static_cast<size_type>(static_cast<std::uint64_t>(rows[output_row]) - page.priority);
        local_rows_by_page[page_idx].emplace_back(output_row, row_in_page);
      }
      rows_by_page_ptr = &local_rows_by_page;
    }
    auto const& rows_by_page = *rows_by_page_ptr;

    std::uint64_t total_data_buffer_size    = 0;
    std::uint64_t selected_data_buffer_size = 0;
    std::size_t selected_miniblock_count    = 0;
    std::size_t selected_miniblock_read_range_count = 0;
    std::size_t selected_raw_value_size     = 0;
    std::vector<std::vector<miniblock_chunk_info>> chunks_by_page(column_info.pages.size());
    std::vector<std::size_t> selected_chunk_counts_by_page(column_info.pages.size(), 0);
    for (auto const& page : column_info.pages) {
      total_data_buffer_size += page.buffer_sizes[1];
    }
    std::vector<std::size_t> touched_page_indices;
    if (shared_pages_touched > 1) {
      touched_page_indices =
        read_touched_miniblock_chunks(source, column_info.pages, rows_by_page, chunks_by_page);
    }
    for (auto page_idx : touched_page_indices) {

      auto const& chunks       = chunks_by_page[page_idx];
      std::vector<bool> selected_chunks(chunks.size(), false);
      auto const prefer_uniform_stride_lookup =
        rows_by_page[page_idx].size() >= sparse_uniform_miniblock_lookup_min_rows;
      for (auto const& [_, row_in_page] : rows_by_page[page_idx]) {
        selected_chunks[find_miniblock_chunk_for_row(
          chunks, row_in_page, prefer_uniform_stride_lookup)] = true;
      }
      auto selected_chunk_count = std::size_t{0};
      selected_miniblock_read_range_count +=
        count_selected_miniblock_read_ranges(selected_chunks, true);
      for (std::size_t chunk_idx = 0; chunk_idx < chunks.size(); ++chunk_idx) {
        if (selected_chunks[chunk_idx]) {
          auto const& chunk = chunks[chunk_idx];
          CUDF_EXPECTS(chunk.num_rows <=
                         static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) /
                           type_width,
                       "Lance MiniBlock is too large to read");
          selected_data_buffer_size += chunks[chunk_idx].buffer_size;
          selected_raw_value_size += static_cast<std::size_t>(chunk.num_rows) * type_width;
          ++selected_miniblock_count;
          ++selected_chunk_count;
        }
      }
      selected_chunk_counts_by_page[page_idx] = selected_chunk_count;
    }
    if (shared_pages_touched <= 1) {
      for (std::size_t page_idx = 0; page_idx < column_info.pages.size(); ++page_idx) {
        if (rows_by_page[page_idx].empty()) { continue; }
        touched_page_indices.push_back(page_idx);
        chunks_by_page[page_idx] = read_miniblock_chunks(source, column_info.pages[page_idx]);
        auto const& chunks       = chunks_by_page[page_idx];
        std::vector<bool> selected_chunks(chunks.size(), false);
        auto const prefer_uniform_stride_lookup =
          rows_by_page[page_idx].size() >= sparse_uniform_miniblock_lookup_min_rows;
        for (auto const& [_, row_in_page] : rows_by_page[page_idx]) {
          selected_chunks[find_miniblock_chunk_for_row(
            chunks, row_in_page, prefer_uniform_stride_lookup)] = true;
        }
        auto selected_chunk_count = std::size_t{0};
        selected_miniblock_read_range_count +=
          count_selected_miniblock_read_ranges(selected_chunks, true);
        for (std::size_t chunk_idx = 0; chunk_idx < chunks.size(); ++chunk_idx) {
          if (selected_chunks[chunk_idx]) {
            auto const& chunk = chunks[chunk_idx];
            CUDF_EXPECTS(chunk.num_rows <=
                           static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) /
                             type_width,
                         "Lance MiniBlock is too large to read");
            selected_data_buffer_size += chunks[chunk_idx].buffer_size;
            selected_raw_value_size += static_cast<std::size_t>(chunk.num_rows) * type_width;
            ++selected_miniblock_count;
            ++selected_chunk_count;
          }
        }
        selected_chunk_counts_by_page[page_idx] = selected_chunk_count;
      }
    }

    if (selected_data_buffer_size >= (total_data_buffer_size + 1) / 2) {
      std::uint64_t dense_rows = 0;
      for (auto const& page : column_info.pages) {
        dense_rows += page.length;
      }

      auto dense_column = read_lance_column(source, field, column_info, dense_rows, stream, mr);
      auto output = cudf::make_fixed_width_column(field_type,
                                                  static_cast<size_type>(rows.size()),
                                                  mask_state::UNALLOCATED,
                                                  stream,
                                                  mr);
      auto source_map = cudf::detail::make_device_uvector(rows, stream, mr);
      std::vector<size_type> target_rows(rows.size());
      std::iota(target_rows.begin(), target_rows.end(), 0);
      auto target_map = cudf::detail::make_device_uvector(target_rows, stream, mr);
      copy_sparse_fixed_width(dense_column->view().head<std::uint8_t>(),
                              output->mutable_view().head<std::uint8_t>(),
                              source_map.data(),
                              target_map.data(),
                              static_cast<size_type>(source_map.size()),
                              type_width,
                              stream);
      output_columns.push_back(std::move(output));
      continue;
    }

    if (use_batched_sparse_headers && selected_miniblock_count > 0) {
      sparse_miniblock_reads.reserve(sparse_miniblock_reads.size() + selected_miniblock_count);
      inputs.reserve(inputs.size() + selected_miniblock_count);
      outputs.reserve(outputs.size() + selected_miniblock_count);
      output_sizes.reserve(output_sizes.size() + selected_miniblock_count);
      page_copies.reserve(page_copies.size() + selected_miniblock_count);
      flat_page_copies.reserve(flat_page_copies.size() + selected_miniblock_count);
      source_rows_by_page.reserve(source_rows_by_page.size() + selected_miniblock_count);
      target_rows_by_page.reserve(target_rows_by_page.size() + selected_miniblock_count);
      single_row_page_copies.reserve(single_row_page_copies.size() + selected_miniblock_count);
    }

    std::uint64_t sparse_page_span_begin = 0;
    std::uint8_t const* sparse_page_span_data = nullptr;
    auto const use_selected_narrow_ranges =
      selected_miniblock_read_range_count <= touched_page_indices.size();
    if (use_batched_sparse_headers &&
        touched_page_indices.size() >= sparse_page_span_min_pages &&
        selected_miniblock_count >=
          touched_page_indices.size() * sparse_page_span_min_chunks_per_page &&
        !use_selected_narrow_ranges) {
      std::uint64_t span_begin = std::numeric_limits<std::uint64_t>::max();
      std::uint64_t span_end   = 0;
      for (auto page_idx : touched_page_indices) {
        auto const& page = column_info.pages[page_idx];
        CUDF_EXPECTS(page.buffer_offsets.size() > 1 && page.buffer_sizes.size() > 1,
                     "Lance sparse page is missing data buffer metadata");
        auto const offset = page.buffer_offsets[1];
        auto const size   = page.buffer_sizes[1];
        CUDF_EXPECTS(offset <= source->size() && size <= source->size() - offset,
                     "Lance sparse page data range is out of bounds");
        span_begin = std::min(span_begin, offset);
        span_end   = std::max(span_end, offset + size);
      }
      auto const span_size = span_end - span_begin;
      // Avoid trading hundreds of small selected MiniBlock reads for one huge sparse span.
      if (span_size <= sparse_page_span_max_bytes &&
          (span_size + sparse_page_span_max_overread_ratio - 1) /
              sparse_page_span_max_overread_ratio <=
            selected_data_buffer_size) {
        input_buffers.emplace_back(static_cast<std::size_t>(span_size), stream, mr);
        pending_sparse_device_reads.push_back(
          pending_device_read{span_begin, span_size, input_buffers.back().data()});
        sparse_page_span_begin = span_begin;
        sparse_page_span_data  = input_buffers.back().data();
      }
    }

    std::uint64_t selected_input_buffer_size = selected_data_buffer_size;
    std::vector<std::uint64_t> selected_input_sizes_by_page;
    std::vector<std::uint64_t> selected_span_offsets_by_page;
    std::vector<std::uint8_t> use_selected_span_by_page;
    // Patterned sparse reads can be faster as one bounded span per page, but random lookups should
    // keep the narrower MiniBlock ranges. Only enable spans when most touched pages can use them.
    auto selected_span_candidate_pages = std::size_t{0};
    auto const may_use_selected_page_spans =
      use_batched_sparse_headers && sparse_page_span_data == nullptr &&
      selected_data_buffer_size > 0 &&
      selected_miniblock_count / sparse_page_selected_span_min_chunks >=
        touched_page_indices.size();
    if (may_use_selected_page_spans) {
      selected_span_candidate_pages = static_cast<std::size_t>(std::count_if(
        touched_page_indices.begin(), touched_page_indices.end(), [&](auto page_idx) {
          return selected_chunk_counts_by_page[page_idx] >= sparse_page_selected_span_min_chunks;
        }));
    }
    auto const has_selected_span_candidate =
      may_use_selected_page_spans && selected_span_candidate_pages > 0 &&
      selected_span_candidate_pages * 2 >= touched_page_indices.size();
    if (has_selected_span_candidate) {
      selected_input_buffer_size = 0;
      selected_input_sizes_by_page.assign(column_info.pages.size(), 0);
      selected_span_offsets_by_page.assign(column_info.pages.size(), 0);
      use_selected_span_by_page.assign(column_info.pages.size(), 0);
      auto const selected_span_max_overread_ratio =
        selected_miniblock_count >=
            touched_page_indices.size() * sparse_page_dense_selected_span_min_chunks_per_page
          ? sparse_page_dense_selected_span_max_overread_ratio
          : sparse_page_selected_span_max_overread_ratio;

      for (auto page_idx : touched_page_indices) {
        auto const& page   = column_info.pages[page_idx];
        auto const& chunks = chunks_by_page[page_idx];
        std::vector<bool> selected_chunks(chunks.size(), false);
        auto const prefer_uniform_stride_lookup =
          rows_by_page[page_idx].size() >= sparse_uniform_miniblock_lookup_min_rows;
        for (auto const& [_, row_in_page] : rows_by_page[page_idx]) {
          selected_chunks[find_miniblock_chunk_for_row(
            chunks, row_in_page, prefer_uniform_stride_lookup)] = true;
        }

        auto page_selected_buffer_size = std::uint64_t{0};
        auto span_begin                = std::numeric_limits<std::uint64_t>::max();
        auto span_end                  = std::uint64_t{0};
        auto selected_chunk_count      = std::size_t{0};
        for (std::size_t chunk_idx = 0; chunk_idx < chunks.size(); ++chunk_idx) {
          if (!selected_chunks[chunk_idx]) { continue; }
          auto const& chunk = chunks[chunk_idx];
          page_selected_buffer_size += chunk.buffer_size;
          span_begin = std::min(span_begin, chunk.buffer_offset);
          span_end   = std::max(span_end, chunk.buffer_offset + chunk.buffer_size);
          ++selected_chunk_count;
        }

        auto page_input_size = page_selected_buffer_size;
        if (selected_chunk_count >= sparse_page_selected_span_min_chunks) {
          auto const span_size = span_end - span_begin;
          if (span_size <= page.buffer_sizes[1] &&
              (span_size + selected_span_max_overread_ratio - 1) /
                  selected_span_max_overread_ratio <=
                page_selected_buffer_size) {
            use_selected_span_by_page[page_idx]     = 1;
            selected_span_offsets_by_page[page_idx] = span_begin;
            page_input_size                         = span_size;
          }
        }

        selected_input_sizes_by_page[page_idx] = page_input_size;
        selected_input_buffer_size += page_input_size;
      }
    }

    auto output = cudf::make_fixed_width_column(field_type,
                                                static_cast<size_type>(rows.size()),
                                                mask_state::UNALLOCATED,
                                                stream,
                                                mr);
    auto* output_data = output->mutable_view().head<std::uint8_t>();
    output_columns.push_back(std::move(output));

    std::uint8_t* sparse_value_data = nullptr;
    if (use_batched_sparse_headers && selected_raw_value_size > 0) {
      page_value_buffers.emplace_back(selected_raw_value_size, stream, mr);
      sparse_value_data = page_value_buffers.back().data();
    }
    std::size_t sparse_value_offset = 0;
    std::uint8_t* sparse_input_data = nullptr;
    std::size_t sparse_input_offset = 0;
    if (use_batched_sparse_headers && sparse_page_span_data == nullptr &&
        selected_input_buffer_size > 0) {
      CUDF_EXPECTS(selected_input_buffer_size <=
                     static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()),
                   "Lance sparse input buffer is too large");
      input_buffers.emplace_back(static_cast<std::size_t>(selected_input_buffer_size), stream, mr);
      sparse_input_data = input_buffers.back().data();
    }

    for (auto page_idx : touched_page_indices) {
      auto const& page = column_info.pages[page_idx];
      CUDF_EXPECTS(page.length <=
                     static_cast<std::uint64_t>(std::numeric_limits<size_type>::max()),
                   "Lance page contains too many rows for cuDF");
      CUDF_EXPECTS(page.length <=
                     static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) /
                       type_width,
                   "Lance page is too large to read");
      auto const& chunks = chunks_by_page[page_idx];
      auto const& page_rows_for_copy = rows_by_page[page_idx];
      auto const prefer_uniform_stride_lookup =
        page_rows_for_copy.size() >= sparse_uniform_miniblock_lookup_min_rows;
      auto has_distinct_chunk_rows = use_batched_sparse_headers &&
                                     selected_chunk_counts_by_page[page_idx] ==
                                       page_rows_for_copy.size();
      auto has_single_multi_row_chunk = false;
      std::size_t single_multi_row_chunk_idx = std::numeric_limits<std::size_t>::max();
      std::array<std::pair<size_type, size_type>, sparse_stack_miniblock_count>
        stack_multi_rows_in_chunk{};
      std::array<std::uint8_t, sparse_stack_miniblock_count> stack_chunk_row_counts{};
      std::array<std::pair<size_type, size_type>, sparse_stack_miniblock_count>
        stack_single_rows_by_chunk{};
      std::vector<std::uint8_t> heap_chunk_row_counts;
      std::vector<std::pair<size_type, size_type>> heap_single_rows_by_chunk;
      auto chunk_row_counts = [&](std::size_t chunk_idx) -> std::uint8_t& {
        return chunks.size() <= sparse_stack_miniblock_count ? stack_chunk_row_counts[chunk_idx]
                                                             : heap_chunk_row_counts[chunk_idx];
      };
      auto single_rows_by_chunk =
        [&](std::size_t chunk_idx) -> std::pair<size_type, size_type>& {
        return chunks.size() <= sparse_stack_miniblock_count
                 ? stack_single_rows_by_chunk[chunk_idx]
                 : heap_single_rows_by_chunk[chunk_idx];
      };
      if (has_distinct_chunk_rows) {
        if (chunks.size() > sparse_stack_miniblock_count) {
          heap_chunk_row_counts.assign(chunks.size(), 0);
          heap_single_rows_by_chunk.resize(chunks.size());
        }
        for (auto const& [output_row, row_in_page] : page_rows_for_copy) {
          auto const chunk_idx =
            find_miniblock_chunk_for_row(chunks, row_in_page, prefer_uniform_stride_lookup);
          if (chunk_row_counts(chunk_idx) != 0) {
            has_distinct_chunk_rows = false;
            break;
          }
          auto const& chunk = chunks[chunk_idx];
          auto const row_in_chunk =
            static_cast<size_type>(static_cast<std::uint64_t>(row_in_page) - chunk.row_begin);
          chunk_row_counts(chunk_idx) = 1;
          single_rows_by_chunk(chunk_idx) = std::pair{output_row, row_in_chunk};
        }
      }
      if (!has_distinct_chunk_rows && selected_chunk_counts_by_page[page_idx] == 1 &&
          page_rows_for_copy.size() <= sparse_stack_miniblock_count) {
        has_single_multi_row_chunk = true;
        for (std::size_t row_idx = 0; row_idx < page_rows_for_copy.size(); ++row_idx) {
          auto const& [output_row, row_in_page] = page_rows_for_copy[row_idx];
          auto const chunk_idx =
            find_miniblock_chunk_for_row(chunks, row_in_page, prefer_uniform_stride_lookup);
          if (row_idx == 0) {
            single_multi_row_chunk_idx = chunk_idx;
          } else {
            CUDF_EXPECTS(chunk_idx == single_multi_row_chunk_idx,
                         "Lance sparse MiniBlock selection count is inconsistent");
          }
          auto const& chunk = chunks[chunk_idx];
          auto const row_in_chunk =
            static_cast<size_type>(static_cast<std::uint64_t>(row_in_page) - chunk.row_begin);
          stack_multi_rows_in_chunk[row_idx] = std::pair{output_row, row_in_chunk};
        }
      }

      std::vector<std::vector<std::pair<size_type, size_type>>> rows_by_chunk;
      if (!has_distinct_chunk_rows && !has_single_multi_row_chunk) {
        rows_by_chunk.resize(chunks.size());
        for (auto const& [output_row, row_in_page] : page_rows_for_copy) {
          auto const chunk_idx =
            find_miniblock_chunk_for_row(chunks, row_in_page, prefer_uniform_stride_lookup);
          auto const& chunk    = chunks[chunk_idx];
          auto const row_in_chunk =
            static_cast<size_type>(static_cast<std::uint64_t>(row_in_page) - chunk.row_begin);
          rows_by_chunk[chunk_idx].emplace_back(output_row, row_in_chunk);
        }
      }

      std::vector<selected_miniblock_payload> selected;
      std::vector<selected_miniblock_chunk> selected_chunks;
      selected.reserve(chunks.size());
      selected_chunks.reserve(chunks.size());
      std::uint64_t selected_buffer_size = 0;
      std::size_t total_payload_size     = 0;
      std::size_t total_miniblock_size   = 0;
      for (std::size_t chunk_idx = 0; chunk_idx < chunks.size(); ++chunk_idx) {
        auto const chunk_selected =
          has_distinct_chunk_rows ? chunk_row_counts(chunk_idx) != 0
          : has_single_multi_row_chunk ? chunk_idx == single_multi_row_chunk_idx
                                  : !rows_by_chunk[chunk_idx].empty();
        if (!chunk_selected) { continue; }
        auto const& chunk = chunks[chunk_idx];
        CUDF_EXPECTS(chunk.num_rows <=
                       static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) /
                         type_width,
                     "Lance MiniBlock is too large to read");
        auto const raw_size = static_cast<std::size_t>(chunk.num_rows) * type_width;
        auto const output_offset = static_cast<std::size_t>(chunk.row_begin) * type_width;
        selected_buffer_size += chunk.buffer_size;
        if (use_batched_sparse_headers) {
          CUDF_EXPECTS(chunk.buffer_size <=
                         static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()),
                       "Lance MiniBlock is too large to read");
          selected_chunks.push_back(selected_miniblock_chunk{chunk, raw_size, chunk_idx});
          total_miniblock_size += static_cast<std::size_t>(chunk.buffer_size);
        } else {
          auto const payload = read_miniblock_chunk_payload(source, chunk);
          if (page.layout.compression == compression_type::NONE) {
            CUDF_EXPECTS(payload.payload_size == raw_size,
                         "Uncompressed Lance MiniBlock has unexpected payload size");
          }
          selected.push_back(selected_miniblock_payload{payload, raw_size, output_offset});
          total_payload_size += payload.payload_size;
        }
      }

      std::uint8_t* page_values = nullptr;
      if (!use_batched_sparse_headers) {
        auto const page_raw_size = static_cast<std::size_t>(page.length) * type_width;
        page_value_buffers.emplace_back(page_raw_size, stream, mr);
        page_values = page_value_buffers.back().data();
      }
      auto const num_selected =
        use_batched_sparse_headers ? selected_chunks.size() : selected.size();

      auto const read_full_page =
        selected_buffer_size >= ((page.buffer_sizes[1] + 1) / 2) ||
        num_selected == chunks.size();
      std::vector<std::size_t> input_offsets(num_selected);
      std::uint8_t const* input_data = nullptr;
      if (sparse_page_span_data != nullptr) {
        CUDF_EXPECTS(page.buffer_offsets[1] >= sparse_page_span_begin,
                     "Lance sparse page is before coalesced page span");
        input_data = sparse_page_span_data +
                     static_cast<std::size_t>(page.buffer_offsets[1] - sparse_page_span_begin);
        if (use_batched_sparse_headers) {
          for (std::size_t idx = 0; idx < selected_chunks.size(); ++idx) {
            auto const& chunk = selected_chunks[idx].chunk;
            CUDF_EXPECTS(chunk.buffer_offset >= page.buffer_offsets[1] &&
                           chunk.buffer_size <= page.buffer_sizes[1] &&
                           chunk.buffer_offset - page.buffer_offsets[1] <=
                             page.buffer_sizes[1] - chunk.buffer_size,
                         "Lance MiniBlock is outside the page data buffer");
            input_offsets[idx] =
              static_cast<std::size_t>(chunk.buffer_offset - page.buffer_offsets[1]);
          }
        } else {
          for (std::size_t idx = 0; idx < selected.size(); ++idx) {
            auto const& payload = selected[idx].payload;
            CUDF_EXPECTS(payload.payload_offset >= page.buffer_offsets[1] &&
                           payload.payload_size <= page.buffer_sizes[1] &&
                           payload.payload_offset - page.buffer_offsets[1] <=
                             page.buffer_sizes[1] - payload.payload_size,
                         "Lance MiniBlock payload is outside the page data buffer");
            input_offsets[idx] =
              static_cast<std::size_t>(payload.payload_offset - page.buffer_offsets[1]);
          }
        }
      } else if (read_full_page) {
        input_buffers.emplace_back(static_cast<std::size_t>(page.buffer_sizes[1]), stream, mr);
        auto* page_input_data = input_buffers.back().data();
        input_data            = page_input_data;
        if (!use_batched_sparse_headers && page.layout.compression == compression_type::NONE) {
          read_device_bytes_into(source,
                                 page.buffer_offsets[1],
                                 page.buffer_sizes[1],
                                 page_input_data,
                                 stream);
        } else {
          pending_sparse_device_reads.push_back(pending_device_read{
            page.buffer_offsets[1], page.buffer_sizes[1], page_input_data});
        }
        if (use_batched_sparse_headers) {
          for (std::size_t idx = 0; idx < selected_chunks.size(); ++idx) {
            auto const& chunk = selected_chunks[idx].chunk;
            CUDF_EXPECTS(chunk.buffer_offset >= page.buffer_offsets[1] &&
                           chunk.buffer_size <= page.buffer_sizes[1] &&
                           chunk.buffer_offset - page.buffer_offsets[1] <=
                             page.buffer_sizes[1] - chunk.buffer_size,
                         "Lance MiniBlock is outside the page data buffer");
            input_offsets[idx] =
              static_cast<std::size_t>(chunk.buffer_offset - page.buffer_offsets[1]);
          }
        } else {
          for (std::size_t idx = 0; idx < selected.size(); ++idx) {
            auto const& payload = selected[idx].payload;
            CUDF_EXPECTS(payload.payload_offset >= page.buffer_offsets[1] &&
                           payload.payload_size <= page.buffer_sizes[1] &&
                           payload.payload_offset - page.buffer_offsets[1] <=
                             page.buffer_sizes[1] - payload.payload_size,
                         "Lance MiniBlock payload is outside the page data buffer");
            input_offsets[idx] =
              static_cast<std::size_t>(payload.payload_offset - page.buffer_offsets[1]);
          }
        }
      } else {
        auto const use_selected_span =
          !use_selected_span_by_page.empty() && use_selected_span_by_page[page_idx] != 0;
        auto const input_buffer_size =
          use_batched_sparse_headers
            ? (!selected_input_sizes_by_page.empty()
                 ? static_cast<std::size_t>(selected_input_sizes_by_page[page_idx])
                 : total_miniblock_size)
            : total_payload_size;
        std::uint8_t* page_input_data = nullptr;
        if (use_batched_sparse_headers && sparse_input_data != nullptr) {
          CUDF_EXPECTS(input_buffer_size <=
                         static_cast<std::size_t>(selected_input_buffer_size) -
                           sparse_input_offset,
                       "Invalid Lance sparse input buffer offset");
          page_input_data = sparse_input_data + sparse_input_offset;
          input_data      = page_input_data;
        } else {
          input_buffers.emplace_back(input_buffer_size, stream, mr);
          page_input_data = input_buffers.back().data();
          input_data      = page_input_data;
        }
        std::size_t input_offset = 0;
        if (use_batched_sparse_headers) {
          if (use_selected_span) {
            auto const range_offset = selected_span_offsets_by_page[page_idx];
            for (std::size_t idx = 0; idx < selected_chunks.size(); ++idx) {
              auto const& chunk = selected_chunks[idx].chunk;
              CUDF_EXPECTS(chunk.buffer_offset >= range_offset &&
                             chunk.buffer_size <= input_buffer_size &&
                             chunk.buffer_offset - range_offset <=
                               input_buffer_size - chunk.buffer_size,
                           "Lance MiniBlock is outside the selected sparse page span");
              input_offsets[idx] = static_cast<std::size_t>(chunk.buffer_offset - range_offset);
            }
            pending_sparse_device_reads.push_back(
              pending_device_read{range_offset, input_buffer_size, page_input_data});
          } else if (selected_chunks.size() > 1) {
            for (std::size_t idx = 0; idx < selected_chunks.size();) {
              auto const range_begin_idx = idx;
              auto range_offset          = selected_chunks[idx].chunk.buffer_offset;
              auto range_size            = selected_chunks[idx].chunk.buffer_size;
              input_offsets[idx]         = input_offset;
              std::size_t range_chunks   = 1;
              ++idx;
              while (idx < selected_chunks.size() &&
                     range_chunks < sparse_max_coalesced_miniblock_reads &&
                     selected_chunks[idx].chunk.buffer_offset == range_offset + range_size) {
                input_offsets[idx] = input_offset + static_cast<std::size_t>(range_size);
                range_size += selected_chunks[idx].chunk.buffer_size;
                ++range_chunks;
                ++idx;
              }
              pending_sparse_device_reads.push_back(pending_device_read{
                range_offset,
                range_size,
                page_input_data + input_offsets[range_begin_idx]});
              input_offset += static_cast<std::size_t>(range_size);
            }
          } else {
            for (std::size_t idx = 0; idx < selected_chunks.size(); ++idx) {
              auto const& chunk = selected_chunks[idx].chunk;
              input_offsets[idx] = input_offset;
              pending_sparse_device_reads.push_back(pending_device_read{
                chunk.buffer_offset,
                chunk.buffer_size,
                page_input_data + input_offset});
              input_offset += static_cast<std::size_t>(chunk.buffer_size);
            }
          }
        } else {
          for (std::size_t idx = 0; idx < selected.size(); ++idx) {
            auto const& payload = selected[idx].payload;
            input_offsets[idx]  = input_offset;
            if (page.layout.compression == compression_type::NONE) {
              read_device_bytes_into(source,
                                     payload.payload_offset,
                                     payload.payload_size,
                                     page_input_data + input_offset,
                                     stream);
            } else {
              pending_sparse_device_reads.push_back(
                pending_device_read{payload.payload_offset,
                                    payload.payload_size,
                                    page_input_data + input_offset});
            }
            input_offset += payload.payload_size;
          }
        }
        if (use_batched_sparse_headers && sparse_input_data != nullptr) {
          sparse_input_offset += input_buffer_size;
        }
      }

      if (use_batched_sparse_headers) {
        for (std::size_t idx = 0; idx < selected_chunks.size(); ++idx) {
          auto const& selected_chunk = selected_chunks[idx];
          CUDF_EXPECTS(sparse_value_data != nullptr &&
                         selected_chunk.raw_size <= selected_raw_value_size - sparse_value_offset,
                       "Invalid Lance sparse value buffer offset");
          auto* chunk_values = sparse_value_data + sparse_value_offset;
          sparse_value_offset += selected_chunk.raw_size;
          sparse_miniblock_reads.push_back(
            sparse_miniblock_read{input_data + input_offsets[idx],
                                  chunk_values,
                                  selected_chunk.raw_size,
                                  selected_chunk.chunk.buffer_size,
                                  page.layout.compression});

          if (has_distinct_chunk_rows) {
            auto const& [output_row, row_in_chunk] =
              single_rows_by_chunk(selected_chunk.chunk_idx);
            single_row_page_copies.push_back(lance_sparse_copy_row{
              chunk_values, output_data, row_in_chunk, output_row, type_width});
          } else if (has_single_multi_row_chunk) {
            auto const row_map_offset = flat_source_rows.size();
            for (std::size_t row_idx = 0; row_idx < page_rows_for_copy.size(); ++row_idx) {
              auto const& [output_row, row_in_chunk] = stack_multi_rows_in_chunk[row_idx];
              flat_target_rows.push_back(output_row);
              flat_source_rows.push_back(row_in_chunk);
            }
            flat_page_copies.push_back(
              sparse_flat_page_copy{chunk_values,
                                    output_data,
                                    row_map_offset,
                                    static_cast<size_type>(page_rows_for_copy.size()),
                                    type_width});
          } else {
            auto const& chunk_rows = rows_by_chunk[selected_chunk.chunk_idx];
            auto const row_map_offset = flat_source_rows.size();
            for (auto const& [output_row, row_in_chunk] : chunk_rows) {
              flat_target_rows.push_back(output_row);
              flat_source_rows.push_back(row_in_chunk);
            }
            flat_page_copies.push_back(
              sparse_flat_page_copy{chunk_values,
                                    output_data,
                                    row_map_offset,
                                    static_cast<size_type>(chunk_rows.size()),
                                    type_width});
          }
        }
        continue;
      } else if (page.layout.compression == compression_type::NONE) {
        for (std::size_t idx = 0; idx < selected.size(); ++idx) {
          CUDF_CUDA_TRY(cudaMemcpyAsync(page_values + selected[idx].output_offset,
                                        input_data + input_offsets[idx],
                                        selected[idx].raw_size,
                                        cudaMemcpyDeviceToDevice,
                                        stream.value()));
        }
      } else {
        CUDF_EXPECTS(page.layout.compression == compression_type::ZSTD,
                     "Unsupported Lance page compression");
        for (std::size_t idx = 0; idx < selected.size(); ++idx) {
          auto const& payload = selected[idx].payload;
          inputs.push_back(
            device_span<std::uint8_t const>{input_data + input_offsets[idx],
                                            static_cast<std::size_t>(payload.payload_size)});
          outputs.push_back(device_span<std::uint8_t>{page_values + selected[idx].output_offset,
                                                      selected[idx].raw_size});
          output_sizes.push_back(selected[idx].raw_size);
          total_raw_size += selected[idx].raw_size;
          max_raw_size = std::max(max_raw_size, selected[idx].raw_size);
        }
      }

      std::size_t row_map_idx{};
      if (can_share_page_rows) {
        row_map_idx = shared_row_map_indices[page_idx];
        CUDF_EXPECTS(row_map_idx != std::numeric_limits<std::size_t>::max(),
                     "Missing shared Lance sparse row map");
      } else {
        std::vector<size_type> source_rows;
        std::vector<size_type> target_rows;
        source_rows.reserve(rows_by_page[page_idx].size());
        target_rows.reserve(rows_by_page[page_idx].size());
        for (auto const& [output_row, row_in_page] : rows_by_page[page_idx]) {
          target_rows.push_back(output_row);
          source_rows.push_back(row_in_page);
        }
        row_map_idx = source_rows_by_page.size();
        source_rows_by_page.push_back(std::move(source_rows));
        target_rows_by_page.push_back(std::move(target_rows));
      }
      page_copies.push_back(sparse_page_copy{page_values, output_data, row_map_idx, type_width});
    }
    if (use_batched_sparse_headers) {
      CUDF_EXPECTS(sparse_value_offset == selected_raw_value_size,
                   "Lance sparse value buffer was not fully assigned");
    }
  }

  read_device_byte_ranges_into(source, pending_sparse_device_reads, stream);

  append_sparse_miniblock_reads(sparse_miniblock_reads,
                                inputs,
                                outputs,
                                output_sizes,
                                total_raw_size,
                                max_raw_size,
                                stream,
                                mr);

  if (!inputs.empty()) {
    decompress_zstd_device_batch_into(
      inputs, outputs, output_sizes, max_raw_size, total_raw_size, stream);
  }

  if (!single_row_page_copies.empty()) {
    auto device_single_row_page_copies =
      cudf::detail::make_device_uvector(single_row_page_copies, stream, mr);
    copy_sparse_fixed_width_single_row_batch(
      device_single_row_page_copies.data(), device_single_row_page_copies.size(), stream);
  }

  if (!flat_page_copies.empty()) {
    constexpr size_type block_size = 256;
    constexpr auto max_blocks_per_chunk = 65535u;
    unsigned int blocks_per_chunk       = 1;

    auto source_map = cudf::detail::make_device_uvector(flat_source_rows, stream, mr);
    auto target_map = cudf::detail::make_device_uvector(flat_target_rows, stream, mr);

    std::vector<lance_sparse_copy_chunk> copy_chunks;
    copy_chunks.reserve(flat_page_copies.size());
    for (auto const& copy : flat_page_copies) {
      auto const chunk_blocks =
        static_cast<unsigned int>(std::min<size_type>(
          (copy.num_rows + block_size - 1) / block_size, max_blocks_per_chunk));
      blocks_per_chunk = std::max(blocks_per_chunk, chunk_blocks);
      copy_chunks.push_back(lance_sparse_copy_chunk{copy.page_values,
                                                    copy.output,
                                                    source_map.data() + copy.row_map_offset,
                                                    target_map.data() + copy.row_map_offset,
                                                    copy.num_rows,
                                                    copy.type_width});
    }

    auto device_copy_chunks = cudf::detail::make_device_uvector(copy_chunks, stream, mr);
    copy_sparse_fixed_width_batch(
      device_copy_chunks.data(), device_copy_chunks.size(), blocks_per_chunk, stream);
  }

  if (!page_copies.empty()) {
    if (page_copies.size() < sparse_copy_batch_min_chunks) {
      std::vector<rmm::device_uvector<size_type>> source_maps;
      std::vector<rmm::device_uvector<size_type>> target_maps;
      source_maps.reserve(source_rows_by_page.size());
      target_maps.reserve(target_rows_by_page.size());
      for (std::size_t idx = 0; idx < source_rows_by_page.size(); ++idx) {
        source_maps.push_back(
          cudf::detail::make_device_uvector(source_rows_by_page[idx], stream, mr));
        target_maps.push_back(
          cudf::detail::make_device_uvector(target_rows_by_page[idx], stream, mr));
      }

      for (auto const& copy : page_copies) {
        copy_sparse_fixed_width(copy.page_values,
                                copy.output,
                                source_maps[copy.row_map_idx].data(),
                                target_maps[copy.row_map_idx].data(),
                                static_cast<size_type>(source_maps[copy.row_map_idx].size()),
                                copy.type_width,
                                stream);
      }
    } else {
      constexpr size_type block_size = 256;
      constexpr auto max_blocks_per_chunk = 65535u;
      unsigned int blocks_per_chunk       = 1;

      std::size_t total_map_rows = 0;
      for (auto const& source_rows : source_rows_by_page) {
        total_map_rows += source_rows.size();
      }
      std::vector<std::size_t> row_map_offsets;
      std::vector<size_type> flat_source_rows;
      std::vector<size_type> flat_target_rows;
      row_map_offsets.reserve(source_rows_by_page.size());
      flat_source_rows.reserve(total_map_rows);
      flat_target_rows.reserve(total_map_rows);
      for (std::size_t idx = 0; idx < source_rows_by_page.size(); ++idx) {
        row_map_offsets.push_back(flat_source_rows.size());
        flat_source_rows.insert(flat_source_rows.end(),
                                source_rows_by_page[idx].begin(),
                                source_rows_by_page[idx].end());
        flat_target_rows.insert(flat_target_rows.end(),
                                target_rows_by_page[idx].begin(),
                                target_rows_by_page[idx].end());
      }

      auto source_map = cudf::detail::make_device_uvector(flat_source_rows, stream, mr);
      auto target_map = cudf::detail::make_device_uvector(flat_target_rows, stream, mr);

      std::vector<lance_sparse_copy_chunk> copy_chunks;
      copy_chunks.reserve(page_copies.size());
      for (auto const& copy : page_copies) {
        auto const num_rows = static_cast<size_type>(source_rows_by_page[copy.row_map_idx].size());
        auto const chunk_blocks =
          static_cast<unsigned int>(std::min<size_type>(
            (num_rows + block_size - 1) / block_size, max_blocks_per_chunk));
        blocks_per_chunk = std::max(blocks_per_chunk, chunk_blocks);
        auto const row_map_offset = row_map_offsets[copy.row_map_idx];
        copy_chunks.push_back(lance_sparse_copy_chunk{copy.page_values,
                                                      copy.output,
                                                      source_map.data() + row_map_offset,
                                                      target_map.data() + row_map_offset,
                                                      num_rows,
                                                      copy.type_width});
      }

      auto device_copy_chunks = cudf::detail::make_device_uvector(copy_chunks, stream, mr);
      copy_sparse_fixed_width_batch(
        device_copy_chunks.data(), device_copy_chunks.size(), blocks_per_chunk, stream);
    }
  }

  return output_columns;
}

table_metadata make_table_metadata(lance_file_info const& file_info,
                                   std::vector<size_type> const& columns,
                                   std::size_t num_rows)
{
  table_metadata metadata;
  metadata.schema_info.reserve(columns.size());
  for (auto column_idx : columns) {
    metadata.schema_info.push_back(column_name_info{file_info.fields[column_idx].name, {}});
  }
  metadata.num_rows_per_source.push_back(num_rows);
  return metadata;
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

table_with_metadata read_lance(datasource* source,
                               lance_reader_options const& options,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  auto const file_info = read_lance_file_info(source, options.get_columns());
  auto const columns   = selected_columns(file_info, options);
  auto const rows      = options.has_row_selection() ? selected_rows(file_info, options)
                                                     : std::vector<size_type>{};
  auto const num_rows  = options.has_row_selection() ? rows.size()
                                                     : static_cast<std::size_t>(file_info.num_rows);

  std::vector<std::unique_ptr<column>> output_columns;
  if (options.has_row_selection()) {
    if (should_use_dense_for_sparse_selection(source, file_info, columns, rows)) {
      auto dense_columns = read_lance_columns(source, file_info, columns, stream, mr);
      auto source_map    = cudf::detail::make_device_uvector(rows, stream, mr);
      std::vector<size_type> target_rows(rows.size());
      std::iota(target_rows.begin(), target_rows.end(), 0);
      auto target_map = cudf::detail::make_device_uvector(target_rows, stream, mr);

      output_columns.reserve(dense_columns.size());
      for (auto& dense_column : dense_columns) {
        auto const field_type = dense_column->type();
        auto output = cudf::make_fixed_width_column(field_type,
                                                    static_cast<size_type>(rows.size()),
                                                    mask_state::UNALLOCATED,
                                                    stream,
                                                    mr);
        copy_sparse_fixed_width(dense_column->view().head<std::uint8_t>(),
                                output->mutable_view().head<std::uint8_t>(),
                                source_map.data(),
                                target_map.data(),
                                static_cast<size_type>(source_map.size()),
                                cudf::size_of(field_type),
                                stream);
        output_columns.push_back(std::move(output));
      }
    } else {
      output_columns.reserve(columns.size());
      if (columns.size() > 1 && can_batch_sparse_zstd_columns(file_info, columns)) {
        output_columns = read_lance_columns(source, file_info, columns, rows, stream, mr);
      } else if (columns.size() == 1 &&
                 should_batch_single_column_sparse_selection(file_info, columns.front(), rows) &&
                 can_batch_sparse_zstd_columns(file_info, columns)) {
        output_columns = read_lance_columns(source, file_info, columns, rows, stream, mr);
      } else {
        for (auto column_idx : columns) {
          output_columns.push_back(read_lance_column(source,
                                                     file_info.fields[column_idx],
                                                     file_info.columns[column_idx],
                                                     rows,
                                                     stream,
                                                     mr));
        }
      }
    }
  } else {
    output_columns = read_lance_columns(source, file_info, columns, stream, mr);
  }

  table_with_metadata result;
  result.tbl      = std::make_unique<table>(std::move(output_columns));
  result.metadata = make_table_metadata(file_info, columns, num_rows);
  return result;
}

}  // namespace detail

lance_writer_options_builder lance_writer_options::builder(sink_info const& sink,
                                                           table_view const& table)
{
  return lance_writer_options_builder(sink, table);
}

lance_reader_options_builder lance_reader_options::builder(source_info source)
{
  return lance_reader_options_builder(std::move(source));
}

}  // namespace cudf::io::experimental
