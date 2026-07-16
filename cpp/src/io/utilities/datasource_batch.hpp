/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/datasource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstddef>
#include <cstdint>
#include <future>
#include <vector>

namespace cudf::io::detail {

struct datasource_device_read_request {
  std::size_t offset{};
  std::size_t size{};
  std::uint8_t* dst{};
};

std::vector<std::future<std::size_t>> device_read_async_batch(
  datasource& source,
  host_span<datasource_device_read_request const> requests,
  rmm::cuda_stream_view stream,
  bool stream_is_ready);

}  // namespace cudf::io::detail
