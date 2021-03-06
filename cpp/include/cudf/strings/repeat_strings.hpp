/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>

namespace cudf {
namespace strings {
/**
 * @addtogroup strings_copy
 * @{
 * @file
 * @brief Strings APIs for copying strings.
 */

/**
 * @brief Repeat the given string scalar by a given number of times.
 *
 * For a given string scalar, an output string scalar is generated by repeating the input string by
 * a number of times given by the @p `repeat_times` parameter. If `repeat_times` is not a positve
 * value, an empty (valid) string scalar will be returned. An invalid input scalar will always
 * result in an invalid output scalar regardless of the value of `repeat_times` parameter.
 *
 * @code{.pseudo}
 * Example:
 * s   = '123XYZ-'
 * out = repeat_strings(s, 3)
 * out is '123XYZ-123XYZ-123XYZ-'
 * @endcode
 *
 * @throw cudf::logic_error if the size of the ouput string scalar exceeds the maximum value that
 *        can be stored by the index type
 *        (i.e., `input.size() * repeat_times > numeric_limits<size_type>::max()`).
 *
 * @param input The scalar containing the string to repeat.
 * @param repeat_times The number of times the `input` string is copied to the output.
 * @param mr Device memory resource used to allocate the returned string scalar.
 * @return New string scalar in which the string is repeated from the input.
 */
std::unique_ptr<string_scalar> repeat_strings(
  string_scalar const& input,
  size_type repeat_times,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Repeat each string in the given strings column by a given number of times.
 *
 * For a given strings column, an output strings column is generated by repeating each string from
 * the input by a number of times given by the @p `repeat_times` parameter. If `repeat_times` is not
 * a positve value, all the rows of the output strings column will be an empty string. Any null row
 * will result in a null row regardless of the value of `repeat_times` parameter.
 *
 * Note that this function cannot handle the cases when the size of the output column exceeds the
 * maximum value that can be indexed by size_type (offset_type). In such situations, an exception
 * may be thrown, or the output result is undefined.
 *
 * @code{.pseudo}
 * Example:
 * strs = ['aa', null, '', 'bbc']
 * out  = repeat_strings(strs, 3)
 * out is ['aaaaaa', null, '', 'bbcbbcbbc']
 * @endcode
 *
 * @param input The column containing strings to repeat.
 * @param repeat_times The number of times each input string is copied to the output.
 * @param mr Device memory resource used to allocate the returned strings column.
 * @return New column with concatenated results.
 */
std::unique_ptr<column> repeat_strings(
  strings_column_view const& input,
  size_type repeat_times,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace cudf
