/*** 
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-03-21 12:33:24
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-03-21 15:32:23
 * @FilePath: \sph_seepage_flows\seepage_flow_cuda\include\kiri_pbs_cuda\solver\dem\cuda_mr_dem_solver_gpu.cuh
 * @Description: 
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved. 
 */
#ifndef _CUDA_MRDEM_SOLVER_GPU_CUH_
#define _CUDA_MRDEM_SOLVER_GPU_CUH_

#pragma once

#include <kiri_pbs_cuda/solver/dem/cuda_dem_solver_common_gpu.cuh>

namespace KIRI {

template <typename AttenuFunc>
static __device__ float3 _ComputeMSDEMCapillaryForce(
    const float3 dij, const float radiusi, const float radiusj, const float sr,
    const float maxForceFactor, AttenuFunc G) {
  float3 f = make_float3(0.f);
  float avg_radius = 2.f * radiusi * radiusj / (radiusi + radiusj);
  // rupture distance
  float contact_angle = 30.f / 180.f * KIRI_PI;
  float volume_liquid_bridge =
      4.f / 3.f * KIRI_PI * powf(avg_radius, 3.f) * 0.01f * 0.01f;
  float s_rupture = (1.f + 0.5f * contact_angle) *
                    (powf(volume_liquid_bridge, 1.f / 3.f) +
                     0.1f * powf(volume_liquid_bridge, 2.f / 3.f));

  float particle_density = 2700.f;
  float gravity = 9.81f;
  float massi = 4.f / 3.f * KIRI_PI * powf(radiusi, 3.f) * particle_density;
  float massj = 4.f / 3.f * KIRI_PI * powf(radiusj, 3.f) * particle_density;
  float weighti = massi * gravity;
  float weightj = massj * gravity;
  float max_force_allowed = maxForceFactor * min(weighti, weightj);

  float dist = length(dij);
  float H = dist - (radiusi + radiusj);
  if (H < s_rupture && H > 0.f) {

    float3 n = dij / dist;

    float coeff_c = G(sr);
    float d = -H + sqrtf(H * H + volume_liquid_bridge / (KIRI_PI * avg_radius));
    float phi = sqrtf(2.f * H / avg_radius *
                      (-1.f + sqrtf(1.f + volume_liquid_bridge /
                                              (KIRI_PI * avg_radius * H * H))));
    float neck_curvature_pressure = -2.f * KIRI_PI * coeff_c * avg_radius *
                                    cosf(contact_angle) / (1.f + H / (2.f * d));
    float surface_tension_force =
        -2.f * KIRI_PI * coeff_c * avg_radius * phi * sinf(contact_angle);

    f = -n * (neck_curvature_pressure + surface_tension_force);
  }

  float force_magnitude = length(f);
  if (force_magnitude > max_force_allowed) {
    f = f * (max_force_allowed / force_magnitude);
  }

  return f;
}

// static __device__ void _ComputeMRDEMForcesTorque(
//     float3 *f, float3 *torque, const size_t i, const float3 *pos,
//     const float3 *vel, const float3 *angularVel, const float *radius,
//     const float young, const float poisson, const float tanFrictionAngle,
//     size_t j, const size_t cellEnd) {
//   while (j < cellEnd) {

//     if (i != j) {
//       float3 dij = pos[j] - pos[i];
//       float rij = radius[i] + radius[j];

//       float dist = length(dij);
//       float penetration_depth = rij - dist;

//       if (penetration_depth > 0.f) {

//         float3 n = dij / dist;
//         float alpha = rij / (rij - penetration_depth);
//         float3 vij = (vel[j] - vel[i]) * alpha +
//                      cross(angularVel[j], -radius[j] * n) -
//                      cross(angularVel[i], radius[i] * n);

//         float kni = young * radius[i];
//         float knj = young * radius[j];
//         float ksi = kni * poisson;
//         float ksj = knj * poisson;

//         float kn = 2.f * kni * knj / (kni + knj);
//         float ks = 2.f * ksi * ksj / (ksi + ksj);

//         float3 force = _ComputeDEMForces(dij, vij, rij, kn, ks,
//                                          tanFrictionAngle,
//                                          penetration_depth);
//         *f += force;
//         *torque += (radius[i] - 0.5f * penetration_depth) * cross(n, force);
//       }
//     }
//     ++j;
//   }
//   return;
// }

template <typename AttenuFunc>
static __device__ void
_ComputeMRDEMCapillaryForces(float3 *f, const size_t i, const float3 *pos,
                             const float *radius, const float sr,
                             const float maxForceFactor, size_t j,
                             const size_t cellEnd, AttenuFunc G) {
  while (j < cellEnd) {

    if (i != j) {
      float3 dij = pos[j] - pos[i];

      *f += _ComputeMSDEMCapillaryForce(dij, radius[i], radius[j], sr,
                                        maxForceFactor, G);
    }
    ++j;
  }
  return;
}

// template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename
// AttenuFunc>
// __global__ void _ComputeMRDEMLinearMomentum_CUDA(
//     float3 *acc, float3 *angularAcc, const float3 *pos, const float3 *vel,
//     const float3 *angularVel, const float *mass, const float *inertia,
//     const float *radius, const float young, const float poisson,
//     const float tanFrictionAngle, const float sr, const float maxForceFactor,
//     const size_t num, const float3 lowestPoint, const float3 highestPoint,
//     size_t *cellStart, const int3 gridSize, Pos2GridXYZ p2xyz,
//     GridXYZ2GridHash xyz2hash, AttenuFunc G) {
//   const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
//   if (i >= num)
//     return;

//   float3 f = make_float3(0.f);
//   float3 torque = make_float3(0.f);
//   int3 grid_xyz = p2xyz(pos[i]);

// #pragma unroll
//   for (int m = 0; m < 27; ++m) {
//     int3 cur_grid_xyz =
//         grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
//     const size_t hash_idx =
//         xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
//     if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
//       continue;

//     _ComputeMRDEMForcesTorque(&f, &torque, i, pos, vel, angularVel, radius,
//                               young, poisson, tanFrictionAngle,
//                               cellStart[hash_idx], cellStart[hash_idx + 1]);
//     _ComputeMRDEMCapillaryForces(&f,
//     i,
//      pos,
//      vel,
//      radius,
//      sr,
//      maxForceFactor,
//                                        cellStart[hash_idx],
//                                        cellStart[hash_idx + 1],
//                                        G);
//   }

//   _ComputeDEMWorldBoundaryForcesTorque(
//       &f, &torque, pos[i], vel[i], angularVel[i], radius[i], 0.01f, young,
//       poisson, tanFrictionAngle, num, lowestPoint, highestPoint);

//   acc[i] += 2.f * f / mass[i];
//   angularAcc[i] = 2.f * torque / inertia[i];
//   return;
// }

} // namespace KIRI

#endif /* _CUDA_DEM_SOLVER_GPU_CUH_ */