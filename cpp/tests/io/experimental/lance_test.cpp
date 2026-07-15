/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/io/experimental/lance.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace {

std::uint16_t read_u16_le(std::vector<char> const& buffer, std::size_t offset)
{
  return static_cast<std::uint16_t>(
    static_cast<std::uint8_t>(buffer[offset]) |
    (static_cast<std::uint16_t>(static_cast<std::uint8_t>(buffer[offset + 1])) << 8));
}

}  // namespace

struct LanceWriterTest : public cudf::test::BaseFixture {};

TEST_F(LanceWriterTest, WritesDefaultZstdFileFooter)
{
  cudf::test::fixed_width_column_wrapper<int32_t> keys({1, 2, 3, 4, 5, 6, 7, 8});
  cudf::test::fixed_width_column_wrapper<double> values({0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5});
  cudf::table_view table({keys, values});

  cudf::io::table_metadata metadata;
  metadata.schema_info.push_back(cudf::io::column_name_info{"key", {}});
  metadata.schema_info.push_back(cudf::io::column_name_info{"value", {}});

  std::vector<char> buffer;
  auto options = cudf::io::experimental::lance_writer_options::builder(
                   cudf::io::sink_info{&buffer}, table)
                   .metadata(std::move(metadata))
                   .max_rows_per_page(3)
                   .build();
  cudf::io::experimental::write_lance(options);

  ASSERT_GT(buffer.size(), 40);
  EXPECT_EQ(read_u16_le(buffer, buffer.size() - 8), 2);
  EXPECT_EQ(read_u16_le(buffer, buffer.size() - 6), 2);
  EXPECT_EQ(std::string(buffer.end() - 4, buffer.end()), "LANC");
}

TEST_F(LanceWriterTest, RejectsUnsupportedCompression)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3});
  cudf::table_view table({col});

  std::vector<char> buffer;
  auto options = cudf::io::experimental::lance_writer_options::builder(
                   cudf::io::sink_info{&buffer}, table)
                   .compression(cudf::io::compression_type::SNAPPY)
                   .build();

  EXPECT_THROW(cudf::io::experimental::write_lance(options), cudf::logic_error);
}

TEST_F(LanceWriterTest, RejectsInvalidPageRows)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3});
  cudf::table_view table({col});

  std::vector<char> buffer;
  auto options = cudf::io::experimental::lance_writer_options::builder(
                   cudf::io::sink_info{&buffer}, table)
                   .max_rows_per_page(0)
                   .build();

  EXPECT_THROW(cudf::io::experimental::write_lance(options), cudf::logic_error);
}

TEST_F(LanceWriterTest, RejectsNulls)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3}, {true, false, true});
  cudf::table_view table({col});

  std::vector<char> buffer;
  auto options = cudf::io::experimental::lance_writer_options::builder(
                   cudf::io::sink_info{&buffer}, table)
                   .build();

  EXPECT_THROW(cudf::io::experimental::write_lance(options), cudf::logic_error);
}

TEST_F(LanceWriterTest, RejectsStrings)
{
  cudf::test::strings_column_wrapper col({"a", "b", "c"});
  cudf::table_view table({col});

  std::vector<char> buffer;
  auto options = cudf::io::experimental::lance_writer_options::builder(
                   cudf::io::sink_info{&buffer}, table)
                   .build();

  EXPECT_THROW(cudf::io::experimental::write_lance(options), cudf::logic_error);
}
