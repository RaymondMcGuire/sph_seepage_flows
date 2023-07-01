/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-03-21 12:33:24
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-03-22 14:47:19
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

  // float particle_density = 2700.f;
  // float gravity = 9.81f;
  // float massi = 4.f / 3.f * KIRI_PI * powf(radiusi, 3.f) * particle_density;
  // float massj = 4.f / 3.f * KIRI_PI * powf(radiusj, 3.f) * particle_density;
  // float weighti = massi * gravity;
  // float weightj = massj * gravity;
  // float max_force_allowed = maxForceFactor * min(weighti, weightj);

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

  // float force_magnitude = length(f);
  // if (force_magnitude > max_force_allowed) {
  //   f = f * (max_force_allowed / force_magnitude);
  // }

  return f;
}

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

} // namespace KIRI

#endif /* _CUDA_DEM_SOLVER_GPU_CUH_ */