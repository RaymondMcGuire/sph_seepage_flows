/*** 
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-05-14 20:01:11
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-05-17 18:49:19
 * @FilePath: \sph_seepage_flows\seepage_flow_cuda\src\kiri_pbs_cuda\emitter\cuda_boundary_emitter.cpp
 * @Description: 
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved. 
 */
#include <kiri_pbs_cuda/emitter/cuda_boundary_emitter.cuh>
namespace KIRI {
void CudaBoundaryEmitter::BuildWorldBoundary(BoundaryData &data,
                                             const float3 &lowest,
                                             const float3 &highest,
                                             const float particleRadius) {
  if (!bEnable)
    return;

  size_t epsilon = 0;
  float spacing = particleRadius * 2.f;
  float3 sides = (highest - lowest) / spacing;

  // ZX plane - bottom
  for (size_t i = -epsilon; i <= sides.x + epsilon; ++i) {
    for (size_t j = -epsilon; j <= sides.z + epsilon; ++j) {
      data.pos.emplace_back(make_float3(lowest.x + i * spacing, lowest.y,
                                        lowest.z + j * spacing));
      data.label.emplace_back(0);
    }
  }

  // ZX plane - top
  for (size_t i = -epsilon; i <= sides.x + epsilon; ++i) {
    for (size_t j = -epsilon; j <= sides.z + epsilon; ++j) {
      data.pos.emplace_back(make_float3(lowest.x + i * spacing, highest.y,
                                        lowest.z + j * spacing));
      data.label.emplace_back(0);
    }
  }

  // XY plane - back
  for (size_t i = -epsilon; i <= sides.x + epsilon; ++i) {
    for (size_t j = -epsilon; j <= sides.y + epsilon; ++j) {
      data.pos.emplace_back(make_float3(lowest.x + i * spacing,
                                        lowest.y + j * spacing, lowest.z));
      data.label.emplace_back(0);
    }
  }

  // XY plane - front
  for (size_t i = -epsilon; i <= sides.x + epsilon; ++i) {
    for (size_t j = -epsilon; j <= sides.y - epsilon; ++j) {
      data.pos.emplace_back(make_float3(lowest.x + i * spacing,
                                        lowest.y + j * spacing, highest.z));
      data.label.emplace_back(0);
    }
  }

  // YZ plane - left
  for (size_t i = -epsilon; i <= sides.y + epsilon; ++i) {
    for (size_t j = -epsilon; j <= sides.z + epsilon; ++j) {
      data.pos.emplace_back(make_float3(lowest.x, lowest.y + i * spacing,
                                        lowest.z + j * spacing));
      data.label.emplace_back(0);
    }
  }

  // YZ plane - right
  for (size_t i = -epsilon; i <= sides.y + epsilon; ++i) {
    for (size_t j = -epsilon; j <= sides.z + epsilon; ++j) {
      data.pos.emplace_back(make_float3(highest.x, lowest.y + i * spacing,
                                        lowest.z + j * spacing));
      data.label.emplace_back(0);
    }
  }
}

void CudaBoundaryEmitter::BuildBoundaryShapeVolume(BoundaryData &data,
                                                   const  std::vector<float3>& shape,
                                                   const bool axis_change) {
  if (!bEnable)
    return;

  for (size_t i = 0; i < shape.size(); i++) {
    if(axis_change)
      data.pos.emplace_back(make_float3(shape[i].x, shape[i].z, shape[i].y));
    else
      data.pos.emplace_back(make_float3(shape[i].x, shape[i].y, shape[i].z));

    data.label.emplace_back(1);
  }
}

} // namespace KIRI
