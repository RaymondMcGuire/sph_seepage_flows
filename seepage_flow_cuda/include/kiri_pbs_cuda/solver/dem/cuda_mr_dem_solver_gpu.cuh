/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-02-28 02:50:46
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-03-15 16:06:50
 * @FilePath:
 * \sph_seepage_flows\seepage_flow_cuda\include\kiri_pbs_cuda\solver\dem\cuda_mr_dem_solver_gpu.cuh
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */
#ifndef _CUDA_MRDEM_SOLVER_GPU_CUH_
#define _CUDA_MRDEM_SOLVER_GPU_CUH_

#pragma once

#include <kiri_pbs_cuda/solver/dem/cuda_dem_solver_common_gpu.cuh>

namespace KIRI {

static __device__ void _ComputeMRDEMForcesTorque(
    float3 *f, float3 *torque, const size_t i, const float3 *pos,
    const float3 *vel, const float3 *angularVel, const float *radius,
    const float young, const float poisson, const float tanFrictionAngle,
    size_t j, const size_t cellEnd) {
  while (j < cellEnd) {

    if (i != j) {
      float3 dij = pos[j] - pos[i];
      float rij = radius[i] + radius[j];

      float dist = length(dij);
      float penetration_depth = rij - dist;

      if (penetration_depth > 0.f) {

        float3 n = dij / dist;
        float alpha = rij / (rij - penetration_depth);
        float3 vij = (vel[j] - vel[i]) * alpha +
                     cross(angularVel[j], -radius[j] * n) -
                     cross(angularVel[i], radius[i] * n);

        float kni = young * radius[i];
        float knj = young * radius[j];
        float ksi = kni * poisson;
        float ksj = knj * poisson;

        float kn = 2.f * kni * knj / (kni + knj);
        float ks = 2.f * ksi * ksj / (ksi + ksj);

        float3 force = _ComputeDEMForces(dij, vij, rij, kn, ks,
                                         tanFrictionAngle, penetration_depth);
        *f += force;
        *torque += (radius[i] - 0.5f * penetration_depth) * cross(n, force);
      }
    }
    ++j;
  }
  return;
}

template <typename AttenuFunc>
static __device__ void
_ComputeMRDEMCapillaryForcesTorque(float3 *f, const size_t i, const float3 *pos,
                                   const float3 *vel, const float *radius,
                                   const float sr, size_t j,
                                   const size_t cellEnd, AttenuFunc G) {
  while (j < cellEnd) {

    if (i != j) {
      float3 dij = pos[j] - pos[i];

      *f += _ComputeMRDEMCapillaryForces(dij, radius[i], radius[j], sr, G);
    }
    ++j;
  }
  return;
}

template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename AttenuFunc>
__global__ void _ComputeMRDEMLinearMomentum_CUDA(
    float3 *acc, float3 *angularAcc, const float3 *pos, const float3 *vel,
    const float3 *angularVel, const float *mass, const float *inertia,
    const float *radius, const float young, const float poisson,
    const float tanFrictionAngle, const float sr, const size_t num,
    const float3 lowestPoint, const float3 highestPoint, size_t *cellStart,
    const int3 gridSize, Pos2GridXYZ p2xyz, GridXYZ2GridHash xyz2hash,
    AttenuFunc G) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;

  float3 f = make_float3(0.f);
  float3 torque = make_float3(0.f);
  int3 grid_xyz = p2xyz(pos[i]);

#pragma unroll
  for (int m = 0; m < 27; ++m) {
    int3 cur_grid_xyz =
        grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
    const size_t hash_idx =
        xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
    if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
      continue;

    _ComputeMRDEMForcesTorque(&f, &torque, i, pos, vel, angularVel, radius,
                              young, poisson, tanFrictionAngle,
                              cellStart[hash_idx], cellStart[hash_idx + 1]);
    _ComputeMRDEMCapillaryForcesTorque(&f, i, pos, vel, radius, sr,
                                       cellStart[hash_idx],
                                       cellStart[hash_idx + 1], G);
  }

  _ComputeDEMWorldBoundaryForcesTorque(
      &f, &torque, pos[i], vel[i], angularVel[i], radius[i], 0.01f, young,
      poisson, tanFrictionAngle, num, lowestPoint, highestPoint);

  acc[i] += 2.f * f / mass[i];
  angularAcc[i] = 2.f * torque / inertia[i];
  return;
}

} // namespace KIRI

#endif /* _CUDA_DEM_SOLVER_GPU_CUH_ */