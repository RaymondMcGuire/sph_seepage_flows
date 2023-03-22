/*** 
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-03-22 14:45:47
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-03-22 15:48:41
 * @FilePath: \sph_seepage_flows\seepage_flow_cuda\include\kiri_pbs_cuda\solver\sph\cuda_dfsph_solver_common_gpu.cuh
 * @Description: 
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved. 
 */
#ifndef _CUDA_DFSPH_SOLVER_COMMON_GPU_CUH_
#define _CUDA_DFSPH_SOLVER_COMMON_GPU_CUH_

#pragma once

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>

namespace KIRI {

template <typename GradientFunc>
__device__ void _ComputeAlpha(float *alpha, float3 *grad_pi, const size_t i,
                              const size_t *label,const float3 *pos, const float *mass, size_t j,
                              const size_t cellEnd, GradientFunc nablaW) {
  while (j < cellEnd) {
    if (i != j && label[i] == label[j]) {
      float3 grad_pj = mass[j] * nablaW(pos[i] - pos[j]);
      *alpha += lengthSquared(grad_pj);
      *grad_pi += grad_pj;
    }

    ++j;
  }
  return;
}

template <typename GradientFunc>
__device__ void
_ComputeBoundaryAlpha(float3 *grad_pi, const float3 posi, const float3 *bpos,
                      const float *volume, const float rho0, size_t j,
                      const size_t cellEnd, GradientFunc nablaW) {
  while (j < cellEnd) {
    *grad_pi += rho0 * volume[j] * nablaW(posi - bpos[j]);
    ++j;
  }
  return;
}

template <typename GradientFunc>
__device__ void
_ComputeDivergenceError(float *error, const size_t i, const size_t *label,const float3 *pos,
                        const float *mass, const float3 *vel, size_t j,
                        const size_t cellEnd, GradientFunc nablaW) {
  while (j < cellEnd) {
    if (i != j && label[i] == label[j])
      *error += mass[j] * dot((vel[i] - vel[j]), nablaW(pos[i] - pos[j]));
    ++j;
  }

  return;
}

template <typename GradientFunc>
__device__ void
_ComputeDivergenceErrorBoundary(float *error, const float3 posi,
                                const float3 veli, const float3 *bpos,
                                const float *volume, const float rho0, size_t j,
                                const size_t cellEnd, GradientFunc nablaW) {
  while (j < cellEnd) {
    *error += rho0 * volume[j] * dot(veli, nablaW(posi - bpos[j]));
    ++j;
  }
  return;
}

template <typename GradientFunc>
__device__ void
_AdaptVelocitiesByDivergence(float3 *v, const size_t i, const size_t *label,const float *stiff,
                             const float3 *pos, const float *mass, size_t j,
                             const size_t cellEnd, GradientFunc nablaW) {

  while (j < cellEnd) {
    if (i != j && label[i] == label[j])
      *v += mass[j] * (stiff[i] + stiff[j]) * nablaW(pos[i] - pos[j]);
    ++j;
  }

  return;
}

template <typename GradientFunc>
__device__ void _AdaptVelocitiesBoundaryByDivergence(
    float3 *v, const float3 posi, const float stiffi, const float3 *bpos,
    const float *volume, const float rho0, size_t j, const size_t cellEnd,
    GradientFunc nablaW) {
  while (j < cellEnd) {
    *v += rho0 * volume[j] * stiffi * nablaW(posi - bpos[j]);
    ++j;
  }
  return;
}

template <typename GradientFunc>
__device__ void
_AdaptVelocitiesByPressure(float3 *v, const size_t i, const size_t *label,const float *stiff,
                           const float3 *pos, const float *mass, const float dt,
                           size_t j, const size_t cellEnd,
                           GradientFunc nablaW) {
  float inv_h = 1.f / dt;
  while (j < cellEnd) {
    if (i != j && label[i] == label[j])
      *v += inv_h * mass[j] * (stiff[i] + stiff[j]) * nablaW(pos[i] - pos[j]);
    ++j;
  }

  return;
}

template <typename GradientFunc>
__device__ void _AdaptVelocitiesBoundaryByPressure(
    float3 *v, const float3 posi, const float stiffi, const float3 *bpos,
    const float *volume, const float rho0, const float dt, size_t j,
    const size_t cellEnd, GradientFunc nablaW) {
  float inv_h = 1.f / dt;
  while (j < cellEnd) {
    *v += inv_h * rho0 * volume[j] * stiffi * nablaW(posi - bpos[j]);
    ++j;
  }
  return;
}

} // namespace KIRI

#endif /* _CUDA_DFSPH_SOLVER_COMMON_GPU_CUH_ */