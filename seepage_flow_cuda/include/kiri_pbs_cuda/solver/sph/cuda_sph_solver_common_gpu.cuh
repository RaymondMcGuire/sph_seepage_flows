/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-02-28 22:45:47
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-02-28 22:59:14
 * @FilePath:
 * \sph_seepage_flows\seepage_flows_cuda\include\kiri_pbs_cuda\solver\sph\cuda_sph_solver_common_gpu.cuh
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */

#ifndef _CUDA_SPH_SOLVER_COMMON_GPU_CUH_
#define _CUDA_SPH_SOLVER_COMMON_GPU_CUH_

#pragma once

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>

namespace KIRI {
static __global__ void ComputePressure_CUDA(float *density, float *pressure,
                                            const size_t num, const float rho0,
                                            const float stiff,
                                            const float negativeScale) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;

  pressure[i] = stiff * (density[i] - rho0);

  if (pressure[i] < 0.f)
    pressure[i] *= negativeScale;

  // if (i == 1000)
  //     printf("pressure = %.3f \n", pressure[i]);

  return;
}

template <typename Func>
__device__ void ComputeFluidDensity(float *density, const size_t i,
                                    const float3 *pos, const float *mass,
                                    size_t j, const size_t cellEnd, Func W) {
  while (j < cellEnd) {
    *density += mass[j] * W(length(pos[i] - pos[j]));
    ++j;
  }

  return;
}

template <typename Func>
__device__ void ComputeBoundaryDensity(float *density, const float3 posi,
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
ComputeBoundaryPressure(float3 *a, const float3 posi, const float densityi,
                        const float pressurei, const float3 *bpos,
                        float *volume, const float rho0, size_t j,
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
__device__ void
ComputeBoundaryViscosity(float3 *a, const float3 posi, const float3 *bpos,
                         const float3 veli, const float densityi,
                         const float *volume, const float bnu, const float rho0,
                         size_t j, const size_t cellEnd, GradientFunc nablaW) {
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

template <typename GradientFunc>
__device__ void
ComputeFluidPressure(float3 *a, const size_t i, float3 *pos, float *mass,
                     float *density, float *pressure, size_t j,
                     const size_t cellEnd, GradientFunc nablaW) {
  while (j < cellEnd) {
    if (i != j)
      *a -= mass[j] *
            (pressure[i] / fmaxf(KIRI_EPSILON, density[i] * density[i]) +
             pressure[j] / fmaxf(KIRI_EPSILON, density[j] * density[j])) *
            nablaW(pos[i] - pos[j]);
    ++j;
  }

  return;
}

template <typename LaplacianFunc>
__device__ void ViscosityMuller2003(float3 *a, const size_t i, float3 *pos,
                                    float3 *vel, float *mass, float *density,
                                    size_t j, const size_t cellEnd,
                                    LaplacianFunc nablaW2) {
  while (j < cellEnd) {
    if (i != j)
      *a += mass[j] * ((vel[j] - vel[i]) / fmaxf(KIRI_EPSILON, density[j])) *
            nablaW2(length(pos[i] - pos[j]));
    ++j;
  }
  return;
}

template <typename GradientFunc>
__device__ void ArtificialViscosity(float3 *a, const size_t i, float3 *pos,
                                    float3 *vel, float *mass, float *density,
                                    const float nu, size_t j,
                                    const size_t cellEnd, GradientFunc nablaW) {
  while (j < cellEnd) {
    if (i != j) {
      float3 dpij = pos[i] - pos[j];
      float3 dv = vel[i] - vel[j];

      float dot_dvdp = dot(dv, dpij);
      if (dot_dvdp < 0.f) {
        float pij = -nu / fmaxf(KIRI_EPSILON, (density[i] + density[j])) *
                    (dot_dvdp / (lengthSquared(dpij) + KIRI_EPSILON));
        *a += -mass[j] * pij * nablaW(dpij);
      }
    }

    ++j;
  }
  return;
}

template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename Func>
__global__ void
ComputeDensity_CUDA(float3 *pos, float *mass, float *density, const float rho0,
                    const size_t num, size_t *cellStart, float3 *bPos,
                    float *bVolume, size_t *bCellStart, const int3 gridSize,
                    Pos2GridXYZ p2xyz, GridXYZ2GridHash xyz2hash, Func W) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;

  int3 grid_xyz = p2xyz(pos[i]);
  density[i] = mass[i] * W(0.f);
#pragma unroll
  for (int m = 0; m < 27; ++m) {
    int3 cur_grid_xyz =
        grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
    const size_t hash_idx =
        xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
    if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
      continue;

    ComputeFluidDensity(&density[i], i, pos, mass, cellStart[hash_idx],
                        cellStart[hash_idx + 1], W);
    ComputeBoundaryDensity(&density[i], pos[i], bPos, bVolume, rho0,
                           bCellStart[hash_idx], bCellStart[hash_idx + 1], W);
  }

  if (density[i] != density[i])
    printf("sph density nan!! density[i]=%.3f \n", density[i]);

  return;
}

template <typename Pos2GridXYZ, typename GridXYZ2GridHash,
          typename GradientFunc, typename LaplacianFunc>
__global__ void ComputeViscosityTerm_CUDA(
    float3 *pos, float3 *vel, float3 *acc, float *mass, float *density,
    const float rho0, const float visc, const float bnu, const size_t num,
    size_t *cellStart, float3 *bPos, float *bVolume, size_t *bCellStart,
    const int3 gridSize, Pos2GridXYZ p2xyz, GridXYZ2GridHash xyz2hash,
    GradientFunc nablaW, LaplacianFunc nablaW2) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;

  float3 a = make_float3(0.f);
  int3 grid_xyz = p2xyz(pos[i]);

#pragma unroll
  for (int m = 0; m < 27; ++m) {

    int3 cur_grid_xyz =
        grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
    const size_t hash_idx =
        xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
    if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
      continue;

    ViscosityMuller2003(&a, i, pos, vel, mass, density, cellStart[hash_idx],
                        cellStart[hash_idx + 1], nablaW2);
    ComputeBoundaryViscosity(&a, pos[i], bPos, vel[i], density[i], bVolume, bnu,
                             rho0, bCellStart[hash_idx],
                             bCellStart[hash_idx + 1], nablaW);
  }

  if (a.x != a.x || a.y != a.y || a.z != a.z) {
    printf("ViscosityMuller2003 acc nan!! a=%.3f,%.3f,%.3f \n",
           KIRI_EXPANDF3(a));
  }

  acc[i] += visc * a;
  return;
}

template <typename Pos2GridXYZ, typename GridXYZ2GridHash,
          typename GradientFunc>
__global__ void ComputeArtificialViscosityTerm_CUDA(
    float3 *pos, float3 *vel, float3 *acc, float *mass, float *density,
    const float rho0, const float nu, const float bnu, const size_t num,
    size_t *cellStart, float3 *bPos, float *bVolume, size_t *bCellStart,
    const int3 gridSize, Pos2GridXYZ p2xyz, GridXYZ2GridHash xyz2hash,
    GradientFunc nablaW) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;

  float3 a = make_float3(0.f);
  int3 grid_xyz = p2xyz(pos[i]);

#pragma unroll
  for (int m = 0; m < 27; ++m) {

    int3 cur_grid_xyz =
        grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
    const size_t hash_idx =
        xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
    if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
      continue;

    ArtificialViscosity(&a, i, pos, vel, mass, density, nu, cellStart[hash_idx],
                        cellStart[hash_idx + 1], nablaW);
    ComputeBoundaryViscosity(&a, pos[i], bPos, vel[i], density[i], bVolume, bnu,
                             rho0, bCellStart[hash_idx],
                             bCellStart[hash_idx + 1], nablaW);
  }

  if (a.x != a.x || a.y != a.y || a.z != a.z) {
    printf("ArtificialViscosity acc nan!! a=%.3f,%.3f,%.3f \n",
           KIRI_EXPANDF3(a));
  }

  acc[i] += a;
  return;
}

} // namespace KIRI

#endif /* _CUDA_SPH_SOLVER_COMMON_GPU_CUH_ */