/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/comp/compression.hpp"
#include "io/comp/nvcomp_adapter.hpp"
#include "io/utilities/hostdevice_vector.hpp"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/scatter.hpp>
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
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
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

data_type data_type_from_lance_logical_type(std::string const& logical_type)
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
  CUDF_FAIL("Unsupported Lance logical type: " + logical_type);
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
  data_type type;
  bool nullable{};
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
  CUDF_EXPECTS(!nullable, "Lance reader currently supports only non-nullable fields");
  return lance_field_info{name, data_type_from_lance_logical_type(logical), nullable};
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

lance_column_info parse_column_metadata(std::vector<std::uint8_t> const& bytes)
{
  lance_column_info column;
  proto_reader reader(bytes);
  int field{};
  wire_type type{};
  while (reader.next(field, type)) {
    if (field == 2) {
      expect_wire_type(type, wire_type::length_delimited, "ColumnMetadata.pages");
      column.pages.push_back(parse_page_metadata(reader.read_message()));
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

lance_file_info read_lance_file_info(datasource* source)
{
  auto const footer         = read_footer(source);
  auto const global_buffers = read_offset_table(
    source, footer.gbo_table_start, footer.num_global_buffers);
  auto info = parse_file_descriptor(
    read_host_bytes(source, global_buffers[0].first, global_buffers[0].second));

  CUDF_EXPECTS(info.fields.size() == footer.num_columns,
               "Lance schema field count does not match footer column count");

  auto const column_offsets =
    read_offset_table(source, footer.cmo_table_start, footer.num_columns);
  info.columns.reserve(footer.num_columns);
  for (auto const& [offset, size] : column_offsets) {
    info.columns.push_back(parse_column_metadata(read_host_bytes(source, offset, size)));
  }
  CUDF_EXPECTS(info.columns.size() == info.fields.size(),
               "Lance column metadata count does not match schema field count");
  return info;
}

rmm::device_uvector<std::uint8_t> read_device_bytes(datasource* source,
                                                    std::uint64_t offset,
                                                    std::uint64_t size,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(size <= static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()),
               "Lance device read range is too large");
  auto const allocation_size = static_cast<std::size_t>(size);
  rmm::device_uvector<std::uint8_t> data(allocation_size, stream, mr);
  if (size == 0) { return data; }

  CUDF_EXPECTS(offset <= source->size() && size <= source->size() - offset,
               "Lance device read range is out of bounds");
  CUDF_EXPECTS(offset <= static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()),
               "Lance device read offset is too large");
  auto const offset_bytes = static_cast<std::size_t>(offset);
  auto const size_bytes = static_cast<std::size_t>(size);
  if (source->is_device_read_preferred(size_bytes)) {
    auto const bytes_read = source->device_read(offset_bytes, size_bytes, data.data(), stream);
    CUDF_EXPECTS(bytes_read == size_bytes, "Failed to read expected Lance device bytes");
  } else {
    auto host_data = source->host_read(offset_bytes, size_bytes);
    CUDF_EXPECTS(host_data->size() == size_bytes, "Failed to read expected Lance host bytes");
    CUDF_CUDA_TRY(cudf::detail::memcpy_async(data.data(), host_data->data(), size_bytes, stream));
    stream.synchronize();
  }
  return data;
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

rmm::device_uvector<std::uint8_t> decompress_zstd_device(std::uint8_t const* input,
                                                         std::size_t input_size,
                                                         std::size_t output_size,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr)
{
  auto disabled = cudf::io::detail::nvcomp::is_decompression_disabled(
    cudf::io::detail::nvcomp::compression_type::ZSTD);
  CUDF_EXPECTS(!disabled.has_value(), "nvCOMP ZSTD decompression is disabled: " + disabled.value());

  rmm::device_uvector<std::uint8_t> output(output_size, stream, mr);
  auto inputs = cudf::detail::hostdevice_vector<device_span<std::uint8_t const>>(1, stream);
  inputs[0]   = device_span<std::uint8_t const>{input, input_size};
  inputs.host_to_device_async(stream);

  auto outputs = cudf::detail::hostdevice_vector<device_span<std::uint8_t>>(1, stream);
  outputs[0]   = device_span<std::uint8_t>{output.data(), output.size()};
  outputs.host_to_device_async(stream);

  auto results = cudf::detail::hostdevice_vector<cudf::io::detail::codec_exec_result>(1, stream);
  results[0]   = cudf::io::detail::codec_exec_result{0, cudf::io::detail::codec_status::FAILURE};
  results.host_to_device_async(stream);

  cudf::io::detail::nvcomp::batched_decompress(cudf::io::detail::nvcomp::compression_type::ZSTD,
                                               inputs,
                                               outputs,
                                               results,
                                               output_size,
                                               output_size,
                                               stream);
  results.device_to_host(stream);

  CUDF_EXPECTS(results[0].status == cudf::io::detail::codec_status::SUCCESS &&
                 results[0].bytes_written == output_size,
               "nvCOMP ZSTD failed to decompress a Lance page");
  return output;
}

rmm::device_uvector<std::uint8_t> read_page_values(datasource* source,
                                                   lance_page_info const& page,
                                                   data_type type,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  constexpr std::size_t miniblock_header_size = 8;
  auto const type_width = cudf::size_of(type);
  auto const raw_size   = page.length * type_width;
  CUDF_EXPECTS(raw_size <= static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()),
               "Lance page is too large to read");
  CUDF_EXPECTS(page.layout.bits_per_value == static_cast<std::uint64_t>(type_width * 8),
               "Lance page encoding does not match schema type width");
  CUDF_EXPECTS(page.buffer_sizes[1] >= miniblock_header_size,
               "Invalid Lance MiniBlock data buffer");

  auto const header = read_host_bytes(source, page.buffer_offsets[1], miniblock_header_size);
  auto const num_levels = read_little_endian_u16(header.data());
  CUDF_EXPECTS(num_levels == 0, "Lance reader does not yet support repetition/definition levels");
  auto const payload_size = read_little_endian_u32(header.data() + sizeof(std::uint16_t));
  CUDF_EXPECTS(miniblock_header_size + payload_size <= page.buffer_sizes[1],
               "Invalid Lance MiniBlock payload size");

  auto payload = read_device_bytes(source,
                                   page.buffer_offsets[1] + miniblock_header_size,
                                   payload_size,
                                   stream,
                                   mr);
  if (page.layout.compression == compression_type::NONE) {
    CUDF_EXPECTS(payload_size == raw_size, "Uncompressed Lance page has unexpected payload size");
    return payload;
  }
  CUDF_EXPECTS(page.layout.compression == compression_type::ZSTD,
               "Unsupported Lance page compression");
  return decompress_zstd_device(
    payload.data(), payload.size(), static_cast<std::size_t>(raw_size), stream, mr);
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
    columns.resize(file_info.fields.size());
    std::iota(columns.begin(), columns.end(), 0);
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
    columns.push_back(found->second);
  }
  return columns;
}

std::size_t find_page_for_row(std::vector<lance_page_info> const& pages, size_type row)
{
  auto const row_u64 = static_cast<std::uint64_t>(row);
  for (std::size_t idx = 0; idx < pages.size(); ++idx) {
    auto const& page = pages[idx];
    if (row_u64 >= page.priority && row_u64 < page.priority + page.length) { return idx; }
  }
  CUDF_FAIL("Lance row selection references a row without a data page");
}

std::unique_ptr<column> read_lance_column(datasource* source,
                                          lance_field_info const& field,
                                          lance_column_info const& column_info,
                                          std::vector<size_type> const& rows,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  auto output = cudf::make_fixed_width_column(
    field.type, static_cast<size_type>(rows.size()), mask_state::UNALLOCATED, stream, mr);
  if (rows.empty()) { return output; }

  std::vector<std::vector<std::pair<size_type, size_type>>> rows_by_page(column_info.pages.size());
  for (size_type output_row = 0; output_row < static_cast<size_type>(rows.size()); ++output_row) {
    auto const page_idx = find_page_for_row(column_info.pages, rows[output_row]);
    auto const& page    = column_info.pages[page_idx];
    auto const row_in_page =
      static_cast<size_type>(static_cast<std::uint64_t>(rows[output_row]) - page.priority);
    rows_by_page[page_idx].emplace_back(output_row, row_in_page);
  }

  auto const type_width = cudf::size_of(field.type);
  CUDF_CUDA_TRY(cudaMemsetAsync(output->mutable_view().head<std::uint8_t>(),
                                0,
                                rows.size() * type_width,
                                stream.value()));
  for (std::size_t page_idx = 0; page_idx < column_info.pages.size(); ++page_idx) {
    if (rows_by_page[page_idx].empty()) { continue; }
    auto const& page = column_info.pages[page_idx];
    CUDF_EXPECTS(page.length <= static_cast<std::uint64_t>(std::numeric_limits<size_type>::max()),
                 "Lance page contains too many rows for cuDF");
    auto page_values = read_page_values(source, page, field.type, stream, mr);

    std::vector<size_type> source_rows;
    std::vector<size_type> target_rows;
    source_rows.reserve(rows_by_page[page_idx].size());
    target_rows.reserve(rows_by_page[page_idx].size());
    for (auto const& [output_row, row_in_page] : rows_by_page[page_idx]) {
      target_rows.push_back(output_row);
      source_rows.push_back(row_in_page);
    }

    auto source_map = cudf::detail::make_device_uvector(source_rows, stream, mr);
    auto target_map = cudf::detail::make_device_uvector(target_rows, stream, mr);
    auto page_view  = column_view{
      field.type, static_cast<size_type>(page.length), page_values.data(), nullptr, 0};
    auto gathered = cudf::detail::gather(table_view{{page_view}},
                                         device_span<size_type const>{
                                           source_map.data(), source_map.size()},
                                         out_of_bounds_policy::DONT_CHECK,
                                         negative_index_policy::NOT_ALLOWED,
                                         stream,
                                         mr);
    auto scattered = cudf::detail::scatter(gathered->view(),
                                           device_span<size_type const>{
                                             target_map.data(), target_map.size()},
                                           table_view{{output->view()}},
                                           stream,
                                           mr);
    output         = std::move(scattered->release()[0]);
  }
  return output;
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

  auto const file_info = read_lance_file_info(source);
  auto const rows      = selected_rows(file_info, options);
  auto const columns   = selected_columns(file_info, options);

  std::vector<std::unique_ptr<column>> output_columns;
  output_columns.reserve(columns.size());
  for (auto column_idx : columns) {
    output_columns.push_back(read_lance_column(source,
                                               file_info.fields[column_idx],
                                               file_info.columns[column_idx],
                                               rows,
                                               stream,
                                               mr));
  }

  table_with_metadata result;
  result.tbl      = std::make_unique<table>(std::move(output_columns));
  result.metadata = make_table_metadata(file_info, columns, rows.size());
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
