/*
 * @Author: Xu.WANG
 * @Date: 2020-07-04 14:48:23
 * @LastEditTime: 2022-03-19 02:59:49
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath:
 * \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\solver\dem\cuda_dem_solver_common_gpu.cuh
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

template <typename AttenuFunc>
static __device__ float3 ComputeDemCapillaryForces(float3 dij, float3 vij,
                                                   const float radiusi,
                                                   const float sr,
                                                   AttenuFunc G) {
  float3 f = make_float3(0.f);
  // rupture distance
  float contact_angle = 30.f / 180.f * KIRI_PI;
  float volume_liquid_bridge =
      4.f / 3.f * KIRI_PI * powf(radiusi, 3.f) * 0.01f * 0.01f;
  float s_rupture = (1.f + 0.5f * contact_angle) *
                    (powf(volume_liquid_bridge, 1.f / 3.f) +
                     0.1f * powf(volume_liquid_bridge, 2.f / 3.f));

  float dist = length(dij);
  float H = dist - (radiusi + radiusi);
  if (H < s_rupture && H > 0.f) {
    float3 N = dij / dist;
    float dot_epslion = dot(vij, N);

    float3 vij_normal = dot_epslion * N;
    float3 vij_tangential = vij - dot_epslion * N;

    // float coeff_c = csat + (1.f - sr) * (c0 - csat);
    float coeff_c = G(sr);

    // printf("cohesive=%.3f \n", coeff_c);

    float d = -H + sqrtf(H * H + volume_liquid_bridge / (KIRI_PI * radiusi));
    float phi = sqrtf(2.f * H / radiusi *
                      (-1.f + sqrtf(1.f + volume_liquid_bridge /
                                              (KIRI_PI * radiusi * H * H))));
    float neck_curvature_pressure = -2.f * KIRI_PI * coeff_c * radiusi *
                                    cosf(contact_angle) / (1.f + H / (2.f * d));
    float surface_tension_force =
        -2.f * KIRI_PI * coeff_c * radiusi * phi * sinf(contact_angle);

    f = -N * (neck_curvature_pressure + surface_tension_force);
  }
  return f;
}

template <typename AttenuFunc>
static __device__ float3 ComputeMRDemCapillaryForces(float3 dij, float3 vij,
                                                     const float radiusi,
                                                     const float radiusj,
                                                     const float sr,
                                                     AttenuFunc G) {
  float3 f = make_float3(0.f);
  // rupture distance
  float contact_angle = 30.f / 180.f * KIRI_PI;
  float volume_liquid_bridge = 4.f / 3.f * KIRI_PI *
                               powf((radiusi + radiusj) / 2.f, 3.f) * 0.01f *
                               0.01f;
  float s_rupture = (1.f + 0.5f * contact_angle) *
                    (powf(volume_liquid_bridge, 1.f / 3.f) +
                     0.1f * powf(volume_liquid_bridge, 2.f / 3.f));

  float avg_radius = (radiusi + radiusj) / 2.f;

  float dist = length(dij);
  float H = dist - (radiusi + radiusj);
  if (H < s_rupture && H > 0.f) {
    float3 N = dij / dist;
    float dot_epslion = dot(vij, N);

    float3 vij_normal = dot_epslion * N;
    float3 vij_tangential = vij - dot_epslion * N;

    // float coeff_c = csat + (1.f - sr) * (c0 - csat);
    float coeff_c = G(sr);

    float d = -H + sqrtf(H * H + volume_liquid_bridge / (KIRI_PI * avg_radius));
    float phi = sqrtf(2.f * H / avg_radius *
                      (-1.f + sqrtf(1.f + volume_liquid_bridge /
                                              (KIRI_PI * avg_radius * H * H))));
    float neck_curvature_pressure = -2.f * KIRI_PI * coeff_c * avg_radius *
                                    cosf(contact_angle) / (1.f + H / (2.f * d));
    float surface_tension_force =
        -2.f * KIRI_PI * coeff_c * avg_radius * phi * sinf(contact_angle);

    f = N * (neck_curvature_pressure + surface_tension_force);
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
  float kn = young * radiusi;
  float ks = kn * poisson;

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