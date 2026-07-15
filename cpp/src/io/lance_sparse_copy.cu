/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace cudf::io::experimental::detail {

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

namespace {

__global__ void pack_lance_miniblocks_kernel(lance_pack_chunk const* chunks,
                                             std::size_t num_chunks,
                                             std::uint8_t* output)
{
  auto const chunk_idx = static_cast<std::size_t>(blockIdx.x);
  if (chunk_idx >= num_chunks) { return; }

  auto const chunk = chunks[chunk_idx];
  auto* chunk_output = output + chunk.output_offset;
  if (threadIdx.x == 0) {
    chunk_output[0] = 0;  // no rep/def levels
    chunk_output[1] = 0;
    chunk_output[2] = static_cast<std::uint8_t>(chunk.payload_size & 0xff);
    chunk_output[3] = static_cast<std::uint8_t>((chunk.payload_size >> 8) & 0xff);
    chunk_output[4] = static_cast<std::uint8_t>((chunk.payload_size >> 16) & 0xff);
    chunk_output[5] = static_cast<std::uint8_t>((chunk.payload_size >> 24) & 0xff);
  }

  auto* payload_output = chunk_output + 8;
  for (auto idx = static_cast<std::size_t>(threadIdx.x); idx < chunk.payload_size;
       idx += static_cast<std::size_t>(blockDim.x)) {
    payload_output[idx] = chunk.payload[idx];
  }
}

__global__ void decode_lance_miniblock_headers_kernel(std::uint8_t const* const* headers,
                                                      lance_miniblock_header* decoded,
                                                      std::size_t num_headers)
{
  auto const idx = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= num_headers) { return; }

  auto const* header       = headers[idx];
  decoded[idx].num_levels  = static_cast<std::uint32_t>(header[0]) |
                             (static_cast<std::uint32_t>(header[1]) << 8);
  decoded[idx].payload_size = static_cast<std::uint32_t>(header[2]) |
                              (static_cast<std::uint32_t>(header[3]) << 8) |
                              (static_cast<std::uint32_t>(header[4]) << 16) |
                              (static_cast<std::uint32_t>(header[5]) << 24);
}

template <typename T>
__global__ void sparse_copy_kernel(T const* source,
                                   T* target,
                                   size_type const* source_rows,
                                   size_type const* target_rows,
                                   size_type num_rows)
{
  auto const tid    = static_cast<size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  auto const stride = static_cast<size_type>(blockDim.x * gridDim.x);
  for (size_type idx = tid; idx < num_rows; idx += stride) {
    target[target_rows[idx]] = source[source_rows[idx]];
  }
}

template <typename T>
void launch_sparse_copy(std::uint8_t const* source,
                        std::uint8_t* target,
                        size_type const* source_rows,
                        size_type const* target_rows,
                        size_type num_rows,
                        rmm::cuda_stream_view stream)
{
  constexpr int block_size = 256;
  auto const grid_size     = (num_rows + block_size - 1) / block_size;
  sparse_copy_kernel<<<grid_size, block_size, 0, stream.value()>>>(
    reinterpret_cast<T const*>(source),
    reinterpret_cast<T*>(target),
    source_rows,
    target_rows,
    num_rows);
}

template <typename T>
__device__ void sparse_copy_chunk(lance_sparse_copy_chunk const& chunk,
                                  unsigned int blocks_per_chunk)
{
  auto const* source = reinterpret_cast<T const*>(chunk.source);
  auto* target       = reinterpret_cast<T*>(chunk.target);
  auto const tid =
    static_cast<size_type>(blockIdx.y * blockDim.x + threadIdx.x);
  auto const stride = static_cast<size_type>(blockDim.x * blocks_per_chunk);
  for (size_type idx = tid; idx < chunk.num_rows; idx += stride) {
    target[chunk.target_rows[idx]] = source[chunk.source_rows[idx]];
  }
}

__global__ void sparse_copy_batch_kernel(lance_sparse_copy_chunk const* chunks,
                                         std::size_t num_chunks,
                                         unsigned int blocks_per_chunk)
{
  auto const chunk_idx = static_cast<std::size_t>(blockIdx.x);
  if (chunk_idx >= num_chunks) { return; }

  auto const chunk = chunks[chunk_idx];
  switch (chunk.type_width) {
    case sizeof(std::uint8_t): sparse_copy_chunk<std::uint8_t>(chunk, blocks_per_chunk); break;
    case sizeof(std::uint16_t): sparse_copy_chunk<std::uint16_t>(chunk, blocks_per_chunk); break;
    case sizeof(std::uint32_t): sparse_copy_chunk<std::uint32_t>(chunk, blocks_per_chunk); break;
    case sizeof(std::uint64_t): sparse_copy_chunk<std::uint64_t>(chunk, blocks_per_chunk); break;
  }
}

template <typename T>
__device__ void sparse_copy_single_row(lance_sparse_copy_row const& row)
{
  auto const* source = reinterpret_cast<T const*>(row.source);
  auto* target       = reinterpret_cast<T*>(row.target);
  target[row.target_row] = source[row.source_row];
}

__global__ void sparse_copy_single_row_batch_kernel(lance_sparse_copy_row const* rows,
                                                    std::size_t num_rows)
{
  auto const row_idx = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row_idx >= num_rows) { return; }

  auto const row = rows[row_idx];
  switch (row.type_width) {
    case sizeof(std::uint8_t): sparse_copy_single_row<std::uint8_t>(row); break;
    case sizeof(std::uint16_t): sparse_copy_single_row<std::uint16_t>(row); break;
    case sizeof(std::uint32_t): sparse_copy_single_row<std::uint32_t>(row); break;
    case sizeof(std::uint64_t): sparse_copy_single_row<std::uint64_t>(row); break;
  }
}

}  // namespace

void pack_lance_miniblocks(lance_pack_chunk const* chunks,
                           std::size_t num_chunks,
                           std::uint8_t* output,
                           rmm::cuda_stream_view stream)
{
  if (num_chunks == 0) { return; }

  constexpr int block_size = 256;
  CUDF_EXPECTS(num_chunks <= static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()),
               "Too many Lance MiniBlocks to pack");
  pack_lance_miniblocks_kernel<<<static_cast<unsigned int>(num_chunks),
                                 block_size,
                                 0,
                                 stream.value()>>>(chunks, num_chunks, output);
  CUDF_CUDA_TRY(cudaPeekAtLastError());
}

void decode_lance_miniblock_headers(std::uint8_t const* const* headers,
                                    lance_miniblock_header* decoded,
                                    std::size_t num_headers,
                                    rmm::cuda_stream_view stream)
{
  if (num_headers == 0) { return; }

  constexpr int block_size = 256;
  auto const grid_size     = (num_headers + block_size - 1) / block_size;
  CUDF_EXPECTS(grid_size <= static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()),
               "Too many Lance MiniBlock headers to decode");
  decode_lance_miniblock_headers_kernel<<<static_cast<unsigned int>(grid_size),
                                          block_size,
                                          0,
                                          stream.value()>>>(headers, decoded, num_headers);
  CUDF_CUDA_TRY(cudaPeekAtLastError());
}

void copy_sparse_fixed_width(std::uint8_t const* source,
                             std::uint8_t* target,
                             size_type const* source_rows,
                             size_type const* target_rows,
                             size_type num_rows,
                             std::size_t type_width,
                             rmm::cuda_stream_view stream)
{
  if (num_rows == 0) { return; }

  switch (type_width) {
    case sizeof(std::uint8_t):
      launch_sparse_copy<std::uint8_t>(
        source, target, source_rows, target_rows, num_rows, stream);
      break;
    case sizeof(std::uint16_t):
      launch_sparse_copy<std::uint16_t>(
        source, target, source_rows, target_rows, num_rows, stream);
      break;
    case sizeof(std::uint32_t):
      launch_sparse_copy<std::uint32_t>(
        source, target, source_rows, target_rows, num_rows, stream);
      break;
    case sizeof(std::uint64_t):
      launch_sparse_copy<std::uint64_t>(
        source, target, source_rows, target_rows, num_rows, stream);
      break;
    default: CUDF_FAIL("Unsupported Lance fixed-width sparse copy width");
  }

  CUDF_CUDA_TRY(cudaPeekAtLastError());
}

void copy_sparse_fixed_width_batch(lance_sparse_copy_chunk const* chunks,
                                   std::size_t num_chunks,
                                   unsigned int blocks_per_chunk,
                                   rmm::cuda_stream_view stream)
{
  if (num_chunks == 0) { return; }

  constexpr int block_size = 256;
  CUDF_EXPECTS(num_chunks <= static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()),
               "Too many Lance sparse copy chunks");
  CUDF_EXPECTS(blocks_per_chunk > 0,
               "Lance sparse copy batch must launch at least one block per chunk");

  dim3 grid{static_cast<unsigned int>(num_chunks), blocks_per_chunk, 1};
  sparse_copy_batch_kernel<<<grid, block_size, 0, stream.value()>>>(
    chunks, num_chunks, blocks_per_chunk);
  CUDF_CUDA_TRY(cudaPeekAtLastError());
}

void copy_sparse_fixed_width_single_row_batch(lance_sparse_copy_row const* rows,
                                              std::size_t num_rows,
                                              rmm::cuda_stream_view stream)
{
  if (num_rows == 0) { return; }

  constexpr int block_size = 256;
  auto const grid_size     = (num_rows + block_size - 1) / block_size;
  CUDF_EXPECTS(grid_size <= static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()),
               "Too many Lance sparse single-row copies");
  sparse_copy_single_row_batch_kernel<<<static_cast<unsigned int>(grid_size),
                                        block_size,
                                        0,
                                        stream.value()>>>(rows, num_rows);
  CUDF_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace cudf::io::experimental::detail
