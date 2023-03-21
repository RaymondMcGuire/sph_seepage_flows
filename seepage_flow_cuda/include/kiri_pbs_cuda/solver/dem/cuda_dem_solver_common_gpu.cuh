/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-03-15 17:21:41
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-03-20 23:11:53
 * @FilePath:
 * \sph_seepage_flows\seepage_flow_cuda\include\kiri_pbs_cuda\solver\dem\cuda_dem_solver_common_gpu.cuh
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */

#ifndef _CUDA_DEM_SOLVER_COMMON_GPU_CUH_
#define _CUDA_DEM_SOLVER_COMMON_GPU_CUH_

#pragma once

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>

namespace KIRI {

static __device__ float3 ComputeDemForces(float3 dij, float3 vij, float rij,
                                          float kn, float ks,
                                          float tanFrictionAngle) {
  float3 f = make_float3(0.f);
  float dist = length(dij);
  float penetration_depth = rij - dist;
  if (penetration_depth > 0.f) {
    float3 n = dij / dist;
    float dot_epslion = dot(vij, n);
    float3 vij_tangential = vij - dot_epslion * n;

    float3 normal_force = kn * penetration_depth * n;
    float3 shear_force = -ks * vij_tangential;

    float max_fs =
        lengthSquared(normal_force) * std::powf(tanFrictionAngle, 2.f);

    if (lengthSquared(shear_force) > max_fs) {
      float ratio = sqrt(max_fs) / length(shear_force);
      shear_force *= ratio;
    }

    f = -normal_force - shear_force;
  }
  return f;
}

static __device__ void _ComputeDEMWorldBoundaryForces(
    float3 *f, float3 posi, float3 veli, const float radiusi,
    const float boundaryRadius, const float young, const float poisson,
    const float tanFrictionAngle, const size_t num, const float3 lowestPoint,
    const float3 highestPoint) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;

  float rij = boundaryRadius + radiusi;

  float kni = young * radiusi;
  float knj = young * boundaryRadius;
  float ksi = kni * poisson;
  float ksj = knj * poisson;

  float kn = 2.f * kni * knj / (kni + knj);
  float ks = 2.f * ksi * ksj / (ksi + ksj);

  float3 N = make_float3(0.f);
  float diff = 0.f;

  if (posi.x > highestPoint.x - rij) {
    N = make_float3(1.f, 0.f, 0.f);
    diff = abs(posi.x - highestPoint.x);
    *f += ComputeDemForces(N * diff, -veli, rij, kn, ks, tanFrictionAngle);
  }

  if (posi.x < lowestPoint.x + rij) {
    N = make_float3(-1.f, 0.f, 0.f);
    diff = abs(posi.x - lowestPoint.x);
    *f += ComputeDemForces(N * diff, -veli, rij, kn, ks, tanFrictionAngle);
  }

  if (posi.y > highestPoint.y - rij) {
    N = make_float3(0.f, 1.f, 0.f);
    diff = abs(posi.y - highestPoint.y);
    *f += ComputeDemForces(N * diff, -veli, rij, kn, ks, tanFrictionAngle);
  }

  if (posi.y < lowestPoint.y + rij) {
    N = make_float3(0.f, -1.f, 0.f);
    diff = abs(posi.y - lowestPoint.y);
    *f += ComputeDemForces(N * diff, -veli, rij, kn, ks, tanFrictionAngle);
  }

  if (posi.z > highestPoint.z - rij) {
    N = make_float3(0.f, 0.f, 1.f);
    diff = abs(posi.z - highestPoint.z);
    *f += ComputeDemForces(N * diff, -veli, rij, kn, ks, tanFrictionAngle);
  }

  if (posi.z < lowestPoint.z + 2 * rij) {
    N = make_float3(0.f, 0.f, -1.f);
    diff = abs(posi.z - lowestPoint.z);
    *f += ComputeDemForces(N * diff, -veli, rij, kn, ks, tanFrictionAngle);
  }

  return;
}

} // namespace KIRI

#endif /* _CUDA_DEM_SOLVER_COMMON_GPU_CUH_ */