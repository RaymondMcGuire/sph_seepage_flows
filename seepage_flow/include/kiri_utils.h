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
#include <partio/Partio.h>
// clang-format on
namespace KIRI {
String UInt2Str4Digit(UInt input) {
  char output[5];
  snprintf(output, 5, "%04d", input);
  return String(output);
};

std::vector<float4> ReadBgeoFileForGPU(String Folder, String Name,
                                       bool flip_yz = false) {
  String root_folder = "bgeo";
  String extension = ".bgeo";
  String file_path =
      String(DB_PBR_PATH) + root_folder + "/" + Folder + "/" + Name + extension;
  Partio::ParticlesDataMutable *data = Partio::read(file_path.c_str());

  Partio::ParticleAttribute pos_attr;
  Partio::ParticleAttribute pscale_attr;
  if (!data->attributeInfo("position", pos_attr) ||
      (pos_attr.type != Partio::FLOAT && pos_attr.type != Partio::VECTOR) ||
      pos_attr.count != 3) {
    KIRI_LOG_ERROR("Failed to Get Proper Position Attribute");
  }

  bool pscaleLoaded = data->attributeInfo("pscale", pscale_attr);

  std::vector<float4> pos_array;
  for (Int i = 0; i < data->numParticles(); i++) {
    const float *pos = data->data<float>(pos_attr, i);
    if (pscaleLoaded) {
      const float *pscale = data->data<float>(pscale_attr, i);
      if (i == 0) {
        KIRI_LOG_INFO("pscale={0}", *pscale);
      }

      if (flip_yz) {
        pos_array.push_back(make_float4(pos[0], pos[2], pos[1], *pscale));
      } else {
        pos_array.push_back(make_float4(pos[0], pos[1], pos[2], *pscale));
      }
    } else {
      if (flip_yz) {
        pos_array.push_back(make_float4(pos[0], pos[2], pos[1], 0.01f));
      } else {
        pos_array.push_back(make_float4(pos[0], pos[1], pos[2], 0.01f));
      }
    }
  }

  data->release();

  return pos_array;
}

} // namespace KIRI
#endif