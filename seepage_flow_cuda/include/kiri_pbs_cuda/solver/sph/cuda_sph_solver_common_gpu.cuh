/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-03-15 15:35:48
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-03-18 19:49:09
 * @FilePath:
 * \sph_seepage_flows\seepage_flow_cuda\include\kiri_pbs_cuda\solver\sph\cuda_sph_solver_common_gpu.cuh
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */

#ifndef _CUDA_SPH_SOLVER_COMMON_GPU_CUH_
#define _CUDA_SPH_SOLVER_COMMON_GPU_CUH_

#pragma once

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>

namespace KIRI {

template <typename Func>
__device__ void _ComputeBoundaryDensity(float *density, const float3 posi,
                                        const float3 *bpos, const float *volume,
                                        const float rho0, size_t j,
                                        const size_t cellEnd, Func W) {
  while (j < cellEnd) {
    *density += rho0 * volume[j] * W(length(posi - bpos[j]));
    ++j;
  }
  return;
}

template <typename GradientFunc>
__device__ void
_ComputeBoundaryPressure(float3 *a, const float3 posi, const float densityi,
                         const float pressurei, const float3 *bpos,
                         const float *volume, const float rho0, size_t j,
                         const size_t cellEnd, GradientFunc nablaW) {
  while (j < cellEnd) {
    *a += -rho0 * volume[j] *
          (pressurei / fmaxf(KIRI_EPSILON, densityi * densityi)) *
          nablaW(posi - bpos[j]);
    ++j;
  }
  return;
}

template <typename GradientFunc>
__device__ void _ComputeBoundaryViscosity(
    float3 *a, const float3 posi, const float3 *bpos, const float3 veli,
    const float densityi, const float *volume, const float bnu,
    const float rho0, size_t j, const size_t cellEnd, GradientFunc nablaW) {
  while (j < cellEnd) {

    float3 dpij = posi - bpos[j];

    float dot_dvdp = dot(veli, dpij);
    if (dot_dvdp < 0.f) {
      float pij = -bnu / (2.f * densityi) *
                  (dot_dvdp / (lengthSquared(dpij) + KIRI_EPSILON));
      *a += -volume[j] * rho0 * pij * nablaW(dpij);
    }

    ++j;
  }
  return;
}

} // namespace KIRI

#endif /* _CUDA_SPH_SOLVER_COMMON_GPU_CUH_ */