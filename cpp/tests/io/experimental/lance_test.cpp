/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/io/experimental/lance.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <algorithm>
#include <cstdint>
#include <numeric>
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

void replace_first(std::vector<char>& buffer,
                   std::string const& needle,
                   std::string const& replacement)
{
  ASSERT_EQ(needle.size(), replacement.size());
  auto const found = std::search(buffer.begin(), buffer.end(), needle.begin(), needle.end());
  ASSERT_NE(found, buffer.end());
  std::copy(replacement.begin(), replacement.end(), found);
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

  auto read_options = cudf::io::experimental::lance_reader_options::builder(
                        cudf::io::source_info{cudf::host_span<char>{buffer.data(), buffer.size()}})
                        .build();
  auto result = cudf::io::experimental::read_lance(read_options);
  CUDF_TEST_EXPECT_TABLES_EQUAL(table, result.tbl->view());
}

TEST_F(LanceWriterTest, ReadsSparseRowsFromSelectedColumns)
{
  cudf::test::fixed_width_column_wrapper<int32_t> keys({1, 2, 3, 4, 5, 6, 7, 8});
  cudf::test::fixed_width_column_wrapper<double> values({0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5});
  cudf::test::fixed_width_column_wrapper<int64_t> offsets({10, 20, 30, 40, 50, 60, 70, 80});
  cudf::table_view table({keys, values, offsets});

  cudf::io::table_metadata metadata;
  metadata.schema_info.push_back(cudf::io::column_name_info{"key", {}});
  metadata.schema_info.push_back(cudf::io::column_name_info{"value", {}});
  metadata.schema_info.push_back(cudf::io::column_name_info{"offset", {}});

  std::vector<char> buffer;
  auto write_options = cudf::io::experimental::lance_writer_options::builder(
                         cudf::io::sink_info{&buffer}, table)
                         .metadata(std::move(metadata))
                         .compression(cudf::io::compression_type::NONE)
                         .max_rows_per_page(3)
                         .build();
  cudf::io::experimental::write_lance(write_options);

  auto read_options = cudf::io::experimental::lance_reader_options::builder(
                        cudf::io::source_info{cudf::host_span<char>{buffer.data(), buffer.size()}})
                        .columns({"value", "key"})
                        .rows({4, 1, 4, 6})
                        .build();
  auto result = cudf::io::experimental::read_lance(read_options);

  cudf::test::fixed_width_column_wrapper<double> expected_values({4.5, 1.5, 4.5, 6.5});
  cudf::test::fixed_width_column_wrapper<int32_t> expected_keys({5, 2, 5, 7});
  cudf::table_view expected({expected_values, expected_keys});

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
  ASSERT_EQ(result.metadata.schema_info.size(), 2);
  EXPECT_EQ(result.metadata.schema_info[0].name, "value");
  EXPECT_EQ(result.metadata.schema_info[1].name, "key");
  ASSERT_EQ(result.metadata.num_rows_per_source.size(), 1);
  EXPECT_EQ(result.metadata.num_rows_per_source[0], 4);
}

TEST_F(LanceWriterTest, ReadsAllRowsWithoutSparseSelection)
{
  cudf::test::fixed_width_column_wrapper<int32_t> keys({1, 2, 3, 4, 5, 6, 7, 8});
  cudf::test::fixed_width_column_wrapper<double> values({0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5});
  cudf::table_view table({keys, values});

  cudf::io::table_metadata metadata;
  metadata.schema_info.push_back(cudf::io::column_name_info{"key", {}});
  metadata.schema_info.push_back(cudf::io::column_name_info{"value", {}});

  std::vector<char> buffer;
  auto write_options = cudf::io::experimental::lance_writer_options::builder(
                         cudf::io::sink_info{&buffer}, table)
                         .metadata(std::move(metadata))
                         .compression(cudf::io::compression_type::NONE)
                         .max_rows_per_page(3)
                         .build();
  cudf::io::experimental::write_lance(write_options);

  auto read_options = cudf::io::experimental::lance_reader_options::builder(
                        cudf::io::source_info{cudf::host_span<char>{buffer.data(), buffer.size()}})
                        .build();
  auto result = cudf::io::experimental::read_lance(read_options);

  CUDF_TEST_EXPECT_TABLES_EQUAL(table, result.tbl->view());
  ASSERT_EQ(result.metadata.schema_info.size(), 2);
  EXPECT_EQ(result.metadata.schema_info[0].name, "key");
  EXPECT_EQ(result.metadata.schema_info[1].name, "value");
  ASSERT_EQ(result.metadata.num_rows_per_source.size(), 1);
  EXPECT_EQ(result.metadata.num_rows_per_source[0], 8);
}

TEST_F(LanceWriterTest, ReadsProjectedColumnFromMixedSchema)
{
  cudf::test::fixed_width_column_wrapper<int32_t> unsupported({1, 2, 3, 4});
  cudf::test::fixed_width_column_wrapper<int32_t> keep({10, 20, 30, 40});
  cudf::table_view table({unsupported, keep});

  cudf::io::table_metadata metadata;
  metadata.schema_info.push_back(cudf::io::column_name_info{"unsupported", {}});
  metadata.schema_info.push_back(cudf::io::column_name_info{"keep", {}});

  std::vector<char> buffer;
  auto write_options = cudf::io::experimental::lance_writer_options::builder(
                         cudf::io::sink_info{&buffer}, table)
                         .metadata(std::move(metadata))
                         .compression(cudf::io::compression_type::NONE)
                         .max_rows_per_page(2)
                         .build();
  cudf::io::experimental::write_lance(write_options);

  replace_first(buffer, "int32", "strng");

  auto all_columns_options = cudf::io::experimental::lance_reader_options::builder(
                               cudf::io::source_info{
                                 cudf::host_span<char>{buffer.data(), buffer.size()}})
                               .build();
  EXPECT_THROW(cudf::io::experimental::read_lance(all_columns_options), cudf::logic_error);

  auto projected_options = cudf::io::experimental::lance_reader_options::builder(
                             cudf::io::source_info{
                               cudf::host_span<char>{buffer.data(), buffer.size()}})
                             .columns({"keep"})
                             .build();
  auto result = cudf::io::experimental::read_lance(projected_options);

  cudf::table_view expected({keep});
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
  ASSERT_EQ(result.metadata.schema_info.size(), 1);
  EXPECT_EQ(result.metadata.schema_info[0].name, "keep");
}

TEST_F(LanceWriterTest, ReadsEmptySparseSelection)
{
  cudf::test::fixed_width_column_wrapper<int32_t> keys({1, 2, 3});
  cudf::table_view table({keys});

  std::vector<char> buffer;
  auto write_options = cudf::io::experimental::lance_writer_options::builder(
                         cudf::io::sink_info{&buffer}, table)
                         .compression(cudf::io::compression_type::NONE)
                         .build();
  cudf::io::experimental::write_lance(write_options);

  auto read_options = cudf::io::experimental::lance_reader_options::builder(
                        cudf::io::source_info{cudf::host_span<char>{buffer.data(), buffer.size()}})
                        .rows({})
                        .build();
  auto result = cudf::io::experimental::read_lance(read_options);

  cudf::test::fixed_width_column_wrapper<int32_t> expected_keys({});
  cudf::table_view expected({expected_keys});
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
  ASSERT_EQ(result.metadata.num_rows_per_source.size(), 1);
  EXPECT_EQ(result.metadata.num_rows_per_source[0], 0);
}

TEST_F(LanceWriterTest, ReadsSparseRowsAcrossMiniBlockChunks)
{
  std::vector<int32_t> values(5000);
  std::iota(values.begin(), values.end(), 0);
  cudf::test::fixed_width_column_wrapper<int32_t> col(values.begin(), values.end());
  cudf::table_view table({col});

  std::vector<char> buffer;
  auto write_options = cudf::io::experimental::lance_writer_options::builder(
                         cudf::io::sink_info{&buffer}, table)
                         .compression(cudf::io::compression_type::NONE)
                         .max_rows_per_page(5000)
                         .build();
  cudf::io::experimental::write_lance(write_options);

  auto read_options = cudf::io::experimental::lance_reader_options::builder(
                        cudf::io::source_info{cudf::host_span<char>{buffer.data(), buffer.size()}})
                        .rows({0, 4095, 4096, 4999, 4096})
                        .build();
  auto result = cudf::io::experimental::read_lance(read_options);

  cudf::test::fixed_width_column_wrapper<int32_t> expected({0, 4095, 4096, 4999, 4096});
  cudf::table_view expected_table({expected});
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table, result.tbl->view());
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
