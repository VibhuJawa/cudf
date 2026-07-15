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

namespace cudf::io::experimental::detail {
namespace {

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

}  // namespace

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

}  // namespace cudf::io::experimental::detail
