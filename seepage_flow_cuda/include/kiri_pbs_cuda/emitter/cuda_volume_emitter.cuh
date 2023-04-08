/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-03-21 12:32:20
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-03-21 18:53:26
 * @FilePath:
 * \sph_seepage_flows\seepage_flow_cuda\include\kiri_pbs_cuda\emitter\cuda_volume_emitter.cuh
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */

#ifndef _CUDA_VOLUME_EMITTER_CUH_
#define _CUDA_VOLUME_EMITTER_CUH_

#pragma once

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>

namespace KIRI {

struct SeepageflowVolumeData {
  float sandMinRadius;
  Vec_Float3 pos;
  Vec_Float3 col;
  Vec_Float mass;
  Vec_Float inertia;
  Vec_Float radius;
  Vec_SizeT label;
};

struct SeepageflowMultiVolumeData {
  float sandMinRadius;
  Vec_Float3 pos;
  Vec_Float3 col;
  Vec_Float mass;
  Vec_Float inertia;
  Vec_Float radius;
  Vec_SizeT label;
  Vec_Float2 amcamcp;
  Vec_Float3 cda0asat;
};

class CudaVolumeEmitter {
public:
  explicit CudaVolumeEmitter(bool enable = true) : bEnable(enable) {}

  CudaVolumeEmitter(const CudaVolumeEmitter &) = delete;
  CudaVolumeEmitter &operator=(const CudaVolumeEmitter &) = delete;
  virtual ~CudaVolumeEmitter() noexcept {}

  void BuildSeepageflowShapeVolume(SeepageflowVolumeData &data,
                                   Vec_Float4 shape, float3 color,
                                   float sandDensity, bool offsetY = false,
                                   float worldLowestY = 0.f,
                                   float2 offsetXZ = make_float2(0.f));
  void BuildSeepageflowShapeMultiVolume(SeepageflowMultiVolumeData &data,
                                        Vec_Float4 shape, float3 color,
                                        float sandDensity, float3 cda0asat,
                                        float2 amcamcp, bool offsetY = false,
                                        float worldLowestY = 0.f,
                                        float2 offsetXZ = make_float2(0.f));

  inline constexpr bool GetEmitterStatus() const { return bEnable; }

private:
  bool bEnable;
};

typedef SharedPtr<CudaVolumeEmitter> CudaVolumeEmitterPtr;
} // namespace KIRI

#endif