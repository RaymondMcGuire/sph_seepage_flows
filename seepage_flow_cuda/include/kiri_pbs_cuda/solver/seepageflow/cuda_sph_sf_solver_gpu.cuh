/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-01-12 14:57:38
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-02-28 22:56:45
 * @FilePath:
 * \sph_seepage_flows\seepage_flows_cuda\include\kiri_pbs_cuda\solver\seepageflow\cuda_sph_sf_solver_gpu.cuh
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */
#ifndef _CUDA_SPH_SF_SOLVER_GPU_CUH_
#define _CUDA_SPH_SF_SOLVER_GPU_CUH_

#pragma once

#include <kiri_pbs_cuda/solver/dem/cuda_dem_solver_common_gpu.cuh>
#include <kiri_pbs_cuda/solver/seepageflow/cuda_sf_utils.cuh>
#include <kiri_pbs_cuda/solver/sph/cuda_sph_solver_common_gpu.cuh>
namespace KIRI {

static __device__ void
_ComputeSFUniRadiusDEMForces(float3 *f, const size_t i, const size_t *label,
                             const float3 *pos, const float3 *vel,
                             const float radius, const float young,
                             const float poisson, const float tanFrictionAngle,
                             size_t j, const size_t cellEnd) {
  while (j < cellEnd) {

    if (i != j && label[i] == label[j]) {
      float3 dij = pos[j] - pos[i];
      float rij = 2.f * radius;
      float3 vij = vel[j] - vel[i];
      float kn = young * radius;
      float ks = kn * poisson;

      *f += ComputeDemForces(dij, vij, rij, kn, ks, tanFrictionAngle);
    }
    ++j;
  }
  return;
}

static __device__ void _ComputeSFUniRadiusDEMBoundaryForces(
    float3 *f, const float3 posi, const float3 veli, const size_t *bLabel,
    const float3 *bpos, const float radius, const float young,
    const float poisson, const float tanFrictionAngle, size_t j,
    const size_t cellEnd) {
  while (j < cellEnd) {
    // only collide scene boundary particles
    if (bLabel[j] == 1) {
      float3 dij = posi - bpos[j];
      float rij = 2.f * radius;
      float kn = young * radius;
      float ks = kn * poisson;

      float3 vij = make_float3(-veli.x, -veli.y, -veli.z);
      *f += ComputeDemForces(dij, vij, rij, kn, ks, tanFrictionAngle);
    }
    ++j;
  }
  return;
}

template <typename AttenuFunc>
static __device__ void _ComputeSFUniRadiusDEMCapillaryForces(
    float3 *f, const size_t i, const size_t *label, const float3 *pos,
    const float3 *vel, const float *sr, const float radius, size_t j,
    const size_t cellEnd, AttenuFunc G) {
  while (j < cellEnd) {

    if (i != j && label[i] == label[j]) {
      float3 dij = pos[j] - pos[i];
      float3 vij = vel[j] - vel[i];
      float avg_srij = (sr[i] + sr[j]) / 2.f;
      *f += ComputeDemCapillaryForces(dij, vij, radius, avg_srij, G);
    }
    ++j;
  }
  return;
}

template <typename Func>
__device__ void _ComputeSFSandVolume(float *solidV, const size_t i,
                                     const size_t *label, const float3 *pos,
                                     const float *mass, const float *density,
                                     size_t j, const size_t cellEnd, Func W) {
  while (j < cellEnd) {
    if (label[j] == 1 && density[j] != 0.f) {
      float vj = mass[j] / density[j];
      float wij = W(length(pos[i] - pos[j]));
      *solidV += vj * wij;
    }

    ++j;
  }
  return;
}

template <typename Func>
__device__ void
_ComputeSFAvgFlow(float3 *avgVelS, float3 *avgVelW, float *vS, float *vW,
                  const size_t i, const size_t *label, const float3 *pos,
                  const float3 *vel, const float *mass, const float *density,
                  size_t j, const size_t cellEnd, Func W) {
  while (j < cellEnd) {
    float vj = mass[j] / density[j];
    float wij = W(length(pos[i] - pos[j]));

    if (label[i] == 1 && label[j] == 0) {
      // avg flow for soild
      *avgVelS += vel[j] * vj * wij;
      *vS += vj * wij;
    } else if (label[i] == 0 && label[j] == 1) {
      // avg flow for fluid
      *avgVelW += vel[j] * vj * wij;
      *vW += vj * wij;
    }
    ++j;
  }
  return;
}

template <typename AttenuFunc, typename Func, typename AdhesionFunc>
__device__ void _ComputeSFWaterAdhesionForces(
    float3 *f, float *vS, const size_t i, const size_t *label,
    const float3 *pos, const float *mass, const float *density, const float *sr,
    size_t j, const size_t cellEnd, AttenuFunc G, Func W, AdhesionFunc A) {
  while (j < cellEnd) {
    // i: water j:sand
    if (label[i] == 0 && label[j] == 1) {
      float3 dpij = pos[i] - pos[j];
      float ldp = length(dpij);
      float wij = W(ldp);
      float adij = A(ldp);
      *f +=
          G(sr[j]) * mass[i] * mass[j] * adij * dpij / fmaxf(KIRI_EPSILON, ldp);
      float vj = mass[j] / density[j];
      *vS += vj * wij;
    }
    ++j;
  }
  return;
}

template <typename Func>
__device__ void
_ComputeSFSandVoidage(float *voidN, float *voidD, const size_t i,
                      const size_t *label, const float3 *pos, const float *mass,
                      const float *density, const float *voidage, size_t j,
                      const size_t cellEnd, Func W) {
  while (j < cellEnd) {

    if (label[i] == 1 && label[j] == 0) {
      float vj = mass[j] / density[j];
      float wij = W(length(pos[i] - pos[j]));

      *voidN += voidage[j] * vj * wij;
      *voidD += vj * wij;
    }

    ++j;
  }
  return;
}

template <typename Func, typename GradientFunc>
__device__ void _ComputeSFSandBuoyancyForces(
    float3 *f, float *vW, const size_t i, const size_t *label,
    const float3 *pos, const float *mass, const float *density,
    const float *pressure, const float *voidage, size_t j, const size_t cellEnd,
    Func W, GradientFunc nablaW) {
  while (j < cellEnd) {
    if (i != j && label[i] == 1 && label[j] == 0 && voidage[j] > 0.f &&
        voidage[j] <= 1.f) {
      float densityi = density[i];
      float densityj = density[j] / fmaxf(KIRI_EPSILON, voidage[j]);

      float3 dij = pos[i] - pos[j];
      float wij = W(length(dij));
      float3 nabla_wij = nablaW(dij);

      *vW += mass[j] / densityj * wij;
      *f += -mass[i] * mass[j] * pressure[j] /
            fmaxf(KIRI_EPSILON, densityi * densityj) * nabla_wij;
    }
    ++j;
  }
  return;
}

template <typename Func, typename GradientFunc>
__device__ void _ComputeSFWaterSeepageTerm(
    float3 *a, const size_t i, const size_t *label, const float3 *pos,
    const float *mass, const float *density, const float *pressure,
    const float *voidage, const float3 *avgDragForce, size_t j,
    const size_t cellEnd, Func W, GradientFunc nablaW) {
  while (j < cellEnd) {
    if (i != j && label[i] == 0 && label[j] == 1) {
      float densityi = density[i] / fmaxf(KIRI_EPSILON, voidage[i]);
      float densityj = density[j];

      float3 dij = pos[i] - pos[j];
      float wij = W(length(dij));
      float3 nabla_wij = nablaW(dij);

      // buoyancy term
      float3 bouyancy = -mass[j] * pressure[i] /
                        fmaxf(KIRI_EPSILON, densityi * densityj) * nabla_wij;
      // drag term
      float3 drag = -wij / densityi * avgDragForce[j];

      *a += bouyancy + drag;
    }
    ++j;
  }
  return;
}

template <typename Func>
__device__ void
_ComputeSFSandAdhesionTerm(float3 *f, const size_t i, const size_t *label,
                           const float3 *pos, const float *mass,
                           const float *density, const float3 *avgAdhesionForce,
                           size_t j, const size_t cellEnd, Func W) {
  while (j < cellEnd) {
    if (label[i] == 1 && label[j] == 0) {
      float3 dij = pos[i] - pos[j];
      float wij = W(length(dij));

      float v = mass[i] / fmaxf(KIRI_EPSILON, density[i]);
      float3 adhesion = wij * v * avgAdhesionForce[j];
      *f += adhesion;
    }
    ++j;
  }
  return;
}

template <typename GradientFunc>
__device__ void _ComputeSFWaterPressureTerm(
    float3 *a, const size_t i, const size_t *label, const float3 *pos,
    const float *mass, const float *density, const float *pressure,
    const float *voidage, size_t j, const size_t cellEnd, GradientFunc nablaW) {
  while (j < cellEnd) {
    if (i != j && label[i] == label[j]) {
      float densityi = density[i] / fmaxf(KIRI_EPSILON, voidage[i]);
      float densityj = density[j] / fmaxf(KIRI_EPSILON, voidage[j]);
      *a += -mass[j] *
            (pressure[i] / fmaxf(KIRI_EPSILON, densityi * densityi) +
             pressure[j] / fmaxf(KIRI_EPSILON, densityj * densityj)) *
            nablaW(pos[i] - pos[j]);
    }
    ++j;
  }

  return;
}

template <typename GradientFunc>
__device__ void _ComputeSFWaterArtificialViscosity(
    float3 *a, const size_t i, const size_t *label, const float3 *pos,
    const float3 *vel, const float *mass, const float *density,
    const float *voidage, const float nu, size_t j, const size_t cellEnd,
    GradientFunc nablaW) {
  while (j < cellEnd) {
    if (i != j && label[i] == label[j]) {
      float3 dpij = pos[i] - pos[j];
      float3 dv = vel[i] - vel[j];

      float dot_dvdp = dot(dv, dpij);
      if (dot_dvdp < 0.f) {
        float densityi = density[i] / fmaxf(KIRI_EPSILON, voidage[i]);
        float densityj = density[j] / fmaxf(KIRI_EPSILON, voidage[j]);
        float pij = -nu / (densityi + densityj) *
                    (dot_dvdp / (lengthSquared(dpij) + KIRI_EPSILON));
        *a += -mass[j] * pij * nablaW(dpij);
      }
    }

    ++j;
  }
  return;
}

template <typename Func>
__device__ void _ComputeSFFluidDensity(float *density, const size_t i,
                                       const size_t *label, const float3 *pos,
                                       const float *mass, size_t j,
                                       const size_t cellEnd, Func W) {
  while (j < cellEnd) {
    if (label[i] == label[j]) {
      *density += mass[j] * W(length(pos[i] - pos[j]));
    }

    ++j;
  }

  return;
}

/**
 * @description:
 * @param {rho0: water; rho1: sand; label: 0=water 1=sand}
 * @return {*}
 */
template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename Func>
__global__ void _ComputeSFDensity_CUDA(
    float *density, const size_t *label, const float3 *pos, const float *mass,
    const float rho0, const float rho1, const size_t num, size_t *cellStart,
    const float3 *bPos, const float *bVolume, size_t *bCellStart,
    const int3 gridSize, Pos2GridXYZ p2xyz, GridXYZ2GridHash xyz2hash, Func W) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;

  if (label[i] == 1) {
    density[i] = rho1;
    return;
  }

  int3 grid_xyz = p2xyz(pos[i]);
  density[i] = mass[i] * W(0.f);
#pragma unroll
  for (size_t m = 0; m < 27; ++m) {
    int3 cur_grid_xyz =
        grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
    const size_t hash_idx =
        xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
    if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
      continue;

    _ComputeSFFluidDensity(&density[i], i, label, pos, mass,
                           cellStart[hash_idx], cellStart[hash_idx + 1], W);
    _ComputeBoundaryDensity(&density[i], pos[i], bPos, bVolume, rho0,
                            bCellStart[hash_idx], bCellStart[hash_idx + 1], W);
  }

  return;
}

static __global__ void
_ComputeSFPressure_CUDA(float *pressure, const size_t *label,
                        const float *density, const size_t num,
                        const float rho0, const float stiff,
                        const float negativeScale) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;

  if (label[i] == 0) {
    pressure[i] = stiff * (density[i] - rho0);

    if (pressure[i] < 0.f)
      pressure[i] *= negativeScale;
  } else if (label[i] == 1) {
    pressure[i] = 0.f;
  }

  return;
}

template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename Func>
__global__ void _ComputeSFAvgFlow_CUDA(

    float3 *avgFlowVel, float *voidage, float *saturation, const size_t *label,
    const float3 *pos, const float3 *vel, const float *mass,
    const float *density, const size_t num, size_t *cellStart,
    const int3 gridSize, Pos2GridXYZ p2xyz, GridXYZ2GridHash xyz2hash, Func W) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;

  int3 grid_xyz = p2xyz(pos[i]);

  float v_s = 0.f;
  float v_w = 0.f;
  float sand_volume = 0.f;

  float3 avg_vel_s = make_float3(0.f);
  float3 avg_vel_w = make_float3(0.f);

#pragma unroll
  for (size_t m = 0; m < 27; ++m) {
    int3 cur_grid_xyz =
        grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
    const size_t hash_idx =
        xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
    if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
      continue;

    _ComputeSFSandVolume(&sand_volume, i, label, pos, mass, density,
                         cellStart[hash_idx], cellStart[hash_idx + 1], W);
    _ComputeSFAvgFlow(&avg_vel_s, &avg_vel_w, &v_s, &v_w, i, label, pos, vel,
                      mass, density, cellStart[hash_idx],
                      cellStart[hash_idx + 1], W);
  }

  // average flow
  if (label[i] == 0 && v_w != 0.f) {
    // water average flow
    avgFlowVel[i] = avg_vel_w / v_w;
  } else if (label[i] == 1 && v_s != 0.f) {
    //  sand average flow
    avgFlowVel[i] = avg_vel_s / v_s;
  } else
    avgFlowVel[i] = make_float3(0.f);

  // virtual voidage for fluid
  voidage[i] = 1.f - sand_volume;

  // saturation for sand
  saturation[i] = v_s;
  return;
}

template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename AttenuFunc,
          typename Func, typename AdhesionFunc>
__global__ void _ComputeSFWaterAdhesionForces_CUDA(
    float3 *adhesionForce, float3 *avgAdhesionForce, const size_t *label,
    const float3 *pos, const float *mass, const float *density,
    const float *saturation, const size_t num, size_t *cellStart,
    const int3 gridSize, Pos2GridXYZ p2xyz, GridXYZ2GridHash xyz2hash,
    AttenuFunc G, Func W, AdhesionFunc A) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num || label[i] != 0)
    return;

  int3 grid_xyz = p2xyz(pos[i]);

  float v_s = 0.f;
  float3 ad_forces = make_float3(0.f);

#pragma unroll
  for (size_t m = 0; m < 27; ++m) {
    int3 cur_grid_xyz =
        grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
    const size_t hash_idx =
        xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
    if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
      continue;

    _ComputeSFWaterAdhesionForces(&ad_forces, &v_s, i, label, pos, mass,
                                  density, saturation, cellStart[hash_idx],
                                  cellStart[hash_idx + 1], G, W, A);
  }

  adhesionForce[i] = ad_forces;

  // average adhesion force
  if (v_s != 0.f)
    avgAdhesionForce[i] = ad_forces / v_s;

  return;
}

template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename Func,
          typename AdhesionFunc>
__global__ void _ComputeMultiSFWaterAdhesionForces_CUDA(
    float3 *adhesionForce, float3 *avgAdhesionForce, const size_t *label,
    const float3 *pos, const float *mass, const float *density,
    const float *saturation, const float3 *cda0asat, const float2 *amcamcp,
    const size_t num, size_t *cellStart, const int3 gridSize, Pos2GridXYZ p2xyz,
    GridXYZ2GridHash xyz2hash, Func W, AdhesionFunc A) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num || label[i] != 0)
    return;

  int3 grid_xyz = p2xyz(pos[i]);

  float v_s = 0.f;
  float3 ad_forces = make_float3(0.f);
#pragma unroll
  for (size_t m = 0; m < 27; ++m) {
    int3 cur_grid_xyz =
        grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
    const size_t hash_idx =
        xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
    if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
      continue;

    _ComputeSFWaterAdhesionForces(
        &ad_forces, &v_s, i, label, pos, mass, density, saturation,
        cellStart[hash_idx], cellStart[hash_idx + 1],
        QuadraticBezierCoeff(0.f, 1.5f, 0.5f, 0.8f), W, A);
  }

  adhesionForce[i] = ad_forces;

  // average adhesion force
  if (v_s != 0.f)
    avgAdhesionForce[i] = ad_forces / v_s;

  return;
}

template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename Func>
__global__ void _ComputeSFSandVoidage_CUDA(
    float *voidage, const size_t *label, const float3 *pos, const float *mass,
    const float *density, const size_t num, size_t *cellStart,
    const int3 gridSize, Pos2GridXYZ p2xyz, GridXYZ2GridHash xyz2hash, Func W) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num || label[i] != 1)
    return;

  int3 grid_xyz = p2xyz(pos[i]);

  float void_numerator = 0.f;
  float void_denominator = 0.f;

#pragma unroll
  for (size_t m = 0; m < 27; ++m) {
    int3 cur_grid_xyz =
        grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
    const size_t hash_idx =
        xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
    if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
      continue;

    _ComputeSFSandVoidage(&void_numerator, &void_denominator, i, label, pos,
                          mass, density, voidage, cellStart[hash_idx],
                          cellStart[hash_idx + 1], W);
  }

  // voidage for sand
  if (void_denominator != 0.f)
    voidage[i] = void_numerator / void_denominator;
  else
    voidage[i] = 0.f;

  return;
}

template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename AttenuFunc,
          typename Func, typename GradientFunc>
__global__ void _ComputeSFSandLinearMomentum_CUDA(
    float3 *avgDragForce, float3 *acc, const size_t *label, const float3 *pos,
    const float3 *vel, const float *mass, const float *density,
    const float *pressure, const float *voidage, const float *saturation,
    const float3 *avgFlowVel, const float3 *avgAdhesionForce,
    const float sandRadius, const float waterRadius, const float young,
    const float poisson, const float tanFrictionAngle, const float cd,
    const float gravity, const float rho0, const size_t num,
    const float3 lowestPoint, const float3 highestPoint, size_t *cellStart,
    const float3 *bPos, const size_t *bLabel, size_t *bCellStart,
    const int3 gridSize, Pos2GridXYZ p2xyz, GridXYZ2GridHash xyz2hash,
    AttenuFunc G, Func W, GradientFunc nablaW) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num || label[i] != 1)
    return;

  float v_w = 0.f;
  float3 f = make_float3(0.f);
  float3 df = make_float3(0.f);
  int3 grid_xyz = p2xyz(pos[i]);

#pragma unroll
  for (size_t m = 0; m < 27; ++m) {
    int3 cur_grid_xyz =
        grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
    const size_t hash_idx =
        xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
    if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
      continue;

    // sand particles
    _ComputeSFUniRadiusDEMForces(&f, i, label, pos, vel, sandRadius, young,
                                 poisson, tanFrictionAngle, cellStart[hash_idx],
                                 cellStart[hash_idx + 1]);
    _ComputeSFUniRadiusDEMCapillaryForces(&f, i, label, pos, vel, saturation,
                                          sandRadius, cellStart[hash_idx],
                                          cellStart[hash_idx + 1], G);

    // sand-water interactive(buoyancy term)
    _ComputeSFSandBuoyancyForces(&f, &v_w, i, label, pos, mass, density,
                                 pressure, voidage, cellStart[hash_idx],
                                 cellStart[hash_idx + 1], W, nablaW);

    // adhesion
    _ComputeSFSandAdhesionTerm(&f, i, label, pos, mass, density,
                               avgAdhesionForce, cellStart[hash_idx],
                               cellStart[hash_idx + 1], W);

    // scene boundary particles interactive
    _ComputeSFUniRadiusDEMBoundaryForces(
        &f, pos[i], vel[i], bLabel, bPos, sandRadius, young, poisson,
        tanFrictionAngle, bCellStart[hash_idx], bCellStart[hash_idx + 1]);
  }

  _ComputeDEMWorldBoundaryForces(&f, pos[i], vel[i], sandRadius, waterRadius,
                                 young, poisson, tanFrictionAngle, num,
                                 lowestPoint, highestPoint);

  // sand-water interactive(drag term)
  if (voidage[i] < 1.f && voidage[i] > 0.f) {
    float dragCoeff = cd * powf(voidage[i], 3.f) / powf(1.f - voidage[i], 2.f);

    if (dragCoeff != 0.f)
      df = gravity * rho0 * powf(voidage[i], 2.f) / dragCoeff *
           (avgFlowVel[i] - vel[i]) * mass[i] / density[i];
  }

  if (v_w != 0.f)
    avgDragForce[i] = df / v_w;
  else
    avgDragForce[i] = make_float3(0.f);

  // sand linear momentum
  acc[i] += 2.f * (f + df) / mass[i];
  return;
}

template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename AttenuFunc,
          typename Func, typename GradientFunc>
__global__ void _ComputeMultiSFSandLinearMomentum_CUDA(
    float3 *avgDragForce, float3 *acc, const size_t *label, const float3 *pos,
    const float3 *vel, const float *mass, const float *density,
    const float *pressure, const float *voidage, const float *saturation,
    const float3 *avgFlowVel, const float3 *avgAdhesionForce,
    const float sandRadius, const float waterRadius, const float young,
    const float poisson, const float tanFrictionAngle, const float3 *cda0asat,
    const float gravity, const float rho0, const size_t num,
    const float3 lowestPoint, const float3 highestPoint, size_t *cellStart,
    const float3 *bPos, const size_t *bLabel, size_t *bCellStart,
    const int3 gridSize, Pos2GridXYZ p2xyz, GridXYZ2GridHash xyz2hash,
    AttenuFunc G, Func W, GradientFunc nablaW) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num || label[i] != 1)
    return;

  float v_w = 0.f;
  float3 f = make_float3(0.f);
  float3 df = make_float3(0.f);
  int3 grid_xyz = p2xyz(pos[i]);

#pragma unroll
  for (size_t m = 0; m < 27; ++m) {
    int3 cur_grid_xyz =
        grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
    const size_t hash_idx =
        xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
    if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
      continue;

    // sand particles
    _ComputeSFUniRadiusDEMForces(&f, i, label, pos, vel, sandRadius, young,
                                 poisson, tanFrictionAngle, cellStart[hash_idx],
                                 cellStart[hash_idx + 1]);
    _ComputeSFUniRadiusDEMCapillaryForces(&f, i, label, pos, vel, saturation,
                                          sandRadius, cellStart[hash_idx],
                                          cellStart[hash_idx + 1], G);

    // sand-water interactive(buoyancy term)
    _ComputeSFSandBuoyancyForces(&f, &v_w, i, label, pos, mass, density,
                                 pressure, voidage, cellStart[hash_idx],
                                 cellStart[hash_idx + 1], W, nablaW);

    // adhesion
    _ComputeSFSandAdhesionTerm(&f, i, label, pos, mass, density,
                               avgAdhesionForce, cellStart[hash_idx],
                               cellStart[hash_idx + 1], W);

    // scene boundary particles interactive
    _ComputeSFUniRadiusDEMBoundaryForces(
        &f, pos[i], vel[i], bLabel, bPos, sandRadius, young, poisson,
        tanFrictionAngle, bCellStart[hash_idx], bCellStart[hash_idx + 1]);
  }

  _ComputeDEMWorldBoundaryForces(&f, pos[i], vel[i], sandRadius, waterRadius,
                                 young, poisson, tanFrictionAngle, num,
                                 lowestPoint, highestPoint);

  // sand-water interactive(drag term)
  if (voidage[i] < 1.f && voidage[i] > 0.f) {
    float dragCoeff =
        cda0asat[i].x * powf(voidage[i], 3.f) / powf(1.f - voidage[i], 2.f);

    if (dragCoeff != 0.f)
      df = gravity * rho0 * powf(voidage[i], 2.f) / dragCoeff *
           (avgFlowVel[i] - vel[i]) * mass[i] / density[i];
  }

  if (v_w != 0.f)
    avgDragForce[i] = df / v_w;
  else
    avgDragForce[i] = make_float3(0.f);

  // sand linear momentum
  acc[i] += 2.f * (f + df) / mass[i];
  return;
}

template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename Func,
          typename GradientFunc>
__global__ void _ComputeSFWaterLinearMomentum_CUDA(
    float3 *acc, const size_t *label, const float3 *pos, const float3 *vel,
    const float *mass, const float *density, const float *pressure,
    const float *voidage, const float3 *avgDragForce,
    const float3 *adhesionForce, const float rho0, const float nu,
    const float bnu, const size_t num, size_t *cellStart, const float3 *bPos,
    const float *bVolume, size_t *bCellStart, const int3 gridSize,
    Pos2GridXYZ p2xyz, GridXYZ2GridHash xyz2hash, Func W, GradientFunc nablaW) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num || label[i] != 0)
    return;

  float3 a = make_float3(0.f);
  int3 grid_xyz = p2xyz(pos[i]);

#pragma unroll
  for (size_t m = 0; m < 27; ++m) {
    int3 cur_grid_xyz =
        grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
    const size_t hash_idx =
        xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
    if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
      continue;

    _ComputeSFWaterPressureTerm(&a, i, label, pos, mass, density, pressure,
                                voidage, cellStart[hash_idx],
                                cellStart[hash_idx + 1], nablaW);
    _ComputeSFWaterArtificialViscosity(&a, i, label, pos, vel, mass, density,
                                       voidage, nu, cellStart[hash_idx],
                                       cellStart[hash_idx + 1], nablaW);
    _ComputeSFWaterSeepageTerm(&a, i, label, pos, mass, density, pressure,
                               voidage, avgDragForce, cellStart[hash_idx],
                               cellStart[hash_idx + 1], W, nablaW);

    _ComputeBoundaryPressure(&a, pos[i], rho0, pressure[i], bPos, bVolume, rho0,
                             bCellStart[hash_idx], bCellStart[hash_idx + 1],
                             nablaW);
    _ComputeBoundaryViscosity(&a, pos[i], bPos, vel[i], rho0, bVolume, bnu,
                              rho0, bCellStart[hash_idx],
                              bCellStart[hash_idx + 1], nablaW);
  }

  acc[i] += a - adhesionForce[i] / mass[i];
  return;
}

static __global__ void
_ComputeSFWetSandColor_CUDA(float *maxSaturation, float3 *col,
                            const size_t *label, const float *saturation,
                            const size_t num, const float3 dryCol,
                            const float3 wetCol) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;

  if (label[i] == 1) {
    float3 blend_color = make_float3(col[i]);
    if (maxSaturation[i] < saturation[i])
      blend_color = dryCol + (wetCol - dryCol) * saturation[i];
    col[i] = blend_color;

    if (maxSaturation[i] < saturation[i])
      maxSaturation[i] = saturation[i];
  }

  return;
}

template <typename Pos2GridXYZ, typename GridXYZ2GridHash>
__global__ void _SFWaterBoundaryConstrain_CUDA(
    float3 *pos, float3 *vel, const size_t *label, const size_t num,
    const float3 lowestPoint, const float3 highestPoint, const float radius,
    const float3 *bPos, const size_t *bLabel, const size_t *bCellStart,
    const int3 gridSize, Pos2GridXYZ p2xyz, GridXYZ2GridHash xyz2hash) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num || label[i] != 0)
    return;

  float3 tmp_pos = pos[i];
  float3 tmp_vel = vel[i];

  // world boundary
  if (tmp_pos.x > highestPoint.x - 2 * radius) {
    tmp_pos.x = highestPoint.x - 2 * radius;
    tmp_vel.x = fminf(tmp_vel.x, 0.f);
  }

  if (tmp_pos.x < lowestPoint.x + 2 * radius) {
    tmp_pos.x = lowestPoint.x + 2 * radius;
    tmp_vel.x = fmaxf(tmp_vel.x, 0.f);
  }

  if (tmp_pos.y > highestPoint.y - 2 * radius) {
    tmp_pos.y = highestPoint.y - 2 * radius;
    tmp_vel.y = fminf(tmp_vel.y, 0.f);
  }

  if (tmp_pos.y < lowestPoint.y + 2 * radius) {
    tmp_pos.y = lowestPoint.y + 2 * radius;
    tmp_vel.y = fmaxf(tmp_vel.y, 0.f);
  }

  if (tmp_pos.z > highestPoint.z - 2 * radius) {
    tmp_pos.z = highestPoint.z - 2 * radius;
    tmp_vel.z = fminf(tmp_vel.z, 0.f);
  }

  if (tmp_pos.z < lowestPoint.z + 2 * radius) {
    tmp_pos.z = lowestPoint.z + 2 * radius;
    tmp_vel.z = fmaxf(tmp_vel.z, 0.f);
  }

  // boundary particles
  int3 grid_xyz = p2xyz(pos[i]);
#pragma unroll
  for (size_t m = 0; m < 27; ++m) {
    int3 cur_grid_xyz =
        grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
    const size_t hash_idx =
        xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
    if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
      continue;

    size_t j = bCellStart[hash_idx];
    const size_t cellEnd = bCellStart[hash_idx + 1];
    while (j < cellEnd) {

      if (bLabel[j] == 1) {
        float dpij = length(pos[i] - bPos[j]);
        float overlap = dpij - 2.f * radius;
        if (overlap < 0.f) {
          float3 n = (pos[i] - bPos[j]) / dpij;
          tmp_vel *= 0.f;
          tmp_pos += n * -overlap;
        }
      }

      ++j;
    }
  }

  pos[i] = tmp_pos;
  vel[i] = tmp_vel;

  return;
}

} // namespace KIRI

#endif /* _CUDA_SPH_SF_SOLVER_GPU_CUH_ */