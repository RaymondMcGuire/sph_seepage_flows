/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-01-12 14:57:38
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-01-12 14:58:33
 * @FilePath:
 * \sph_seepage_flows\seepage_flows_cuda\include\kiri_pbs_cuda\emitter\cuda_volume_emitter.cuh
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#ifndef _CUDA_VOLUME_EMITTER_CUH_
#define _CUDA_VOLUME_EMITTER_CUH_

#pragma once

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>

namespace KIRI {

struct SphVolumeData {
  Vec_Float3 pos;
  Vec_Float3 col;
};

struct DemVolumeData {
  Vec_Float3 pos;
  Vec_Float3 col;
  Vec_Float mass;
};

struct DemShapeVolumeData {
  float minRadius;
  Vec_Float3 pos;
  Vec_Float3 col;
  Vec_Float mass;
  Vec_Float radius;
};

struct SeepageflowVolumeData {
  float sandMinRadius;
  Vec_Float3 pos;
  Vec_Float3 col;
  Vec_Float mass;
  Vec_Float radius;
  Vec_SizeT label;
};

struct SeepageflowMultiVolumeData {
  float sandMinRadius;
  Vec_Float3 pos;
  Vec_Float3 col;
  Vec_Float mass;
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

  void BuildSphVolume(SphVolumeData &data, float3 lowest, int3 vsize,
                      float particleRadius, float3 color);
  void BuildUniDemVolume(DemVolumeData &data, float3 lowest, int3 vsize,
                         float particleRadius, float3 color, float mass,
                         float jitter = 0.001f);
  void BuildDemShapeVolume(DemShapeVolumeData &data, Vec_Float4 shape,
                           float3 color, float density);

  void BuildSeepageflowBoxVolume(SeepageflowVolumeData &data, float3 lowest,
                                 int3 vsize, float particleRadius, float3 color,
                                 float mass, size_t label,
                                 float jitter = 0.001f);
  void BuildSeepageflowShapeVolume(SeepageflowVolumeData &data,
                                   Vec_Float4 shape, float3 color,
                                   float sandDensity, bool offsetY = false,
                                   float worldLowestY = 0.f);
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