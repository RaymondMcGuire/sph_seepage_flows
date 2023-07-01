/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-05-14 20:01:11
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-05-17 18:49:28
 * @FilePath:
 * \sph_seepage_flows\seepage_flow_cuda\include\kiri_pbs_cuda\emitter\cuda_boundary_emitter.cuh
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */
#ifndef _CUDA_BOUNDARY_EMITTER_CUH_
#define _CUDA_BOUNDARY_EMITTER_CUH_

#pragma once

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>

namespace KIRI {
struct BoundaryData {
  Vec_Float3 pos;
  Vec_SizeT label;
};

class CudaBoundaryEmitter {
public:
  explicit CudaBoundaryEmitter(bool enable = true) : bEnable(enable) {}

  CudaBoundaryEmitter(const CudaBoundaryEmitter &) = delete;
  CudaBoundaryEmitter &operator=(const CudaBoundaryEmitter &) = delete;
  virtual ~CudaBoundaryEmitter() noexcept {}

  void BuildWorldBoundary(BoundaryData &data, const float3 &lowest,
                          const float3 &highest, const float particleRadius);

  void BuildBoundaryShapeVolume(BoundaryData &data,
                                const std::vector<float3> &shape,
                                const bool axis_change = false);

private:
  bool bEnable;
};

typedef SharedPtr<CudaBoundaryEmitter> CudaBoundaryEmitterPtr;
} // namespace KIRI

#endif