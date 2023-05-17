/***
 * @Date: 2020-09-27 02:54:00
 * @LastEditTime: 2021-08-21 19:46:42
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath: \sph_seepage_flows\seepage_flows\include\kiri_utils.h
 */

#ifndef _KIRI_UTILS_H_
#define _KIRI_UTILS_H_

#pragma once
// clang-format off
#include <root_directory.h>

#include <kiri_pch.h>
#include <filesystem>
#include <sys/types.h>
#include <sys/stat.h>

// clang-format on
namespace KIRI {
String UInt2Str4Digit(UInt input) {
  char output[5];
  snprintf(output, 5, "%04d", input);
  return String(output);
};

} // namespace KIRI
#endif