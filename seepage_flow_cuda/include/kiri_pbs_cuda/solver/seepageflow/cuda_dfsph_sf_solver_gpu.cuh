/*** 
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-03-22 14:42:08
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-03-22 15:49:38
 * @FilePath: \sph_seepage_flows\seepage_flow_cuda\include\kiri_pbs_cuda\solver\seepageflow\cuda_dfsph_sf_solver_gpu.cuh
 * @Description: 
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved. 
 */
#ifndef _CUDA_DFSPH_SF_SOLVER_GPU_CUH_
#define _CUDA_DFSPH_SF_SOLVER_GPU_CUH_

#pragma once

#include <kiri_pbs_cuda/solver/sph/cuda_dfsph_solver_common_gpu.cuh>
namespace KIRI {


template <typename Pos2GridXYZ, typename GridXYZ2GridHash,
          typename GradientFunc>
__global__ void
_ComputeAlpha_CUDA(float *alpha, const size_t *label,const float3 *pos, const float *mass,
                   const float *density, const float rho0, const size_t num,
                   size_t *cellStart, const float3 *bPos, const float *bVolume,
                   size_t *bCellStart, const int3 gridSize, Pos2GridXYZ p2xyz,
                   GridXYZ2GridHash xyz2hash, GradientFunc nablaW) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;
  
  alpha[i] = 0.f;

  if (label[i] == 1) 
    return;
  

  int3 grid_xyz = p2xyz(pos[i]);
  float3 grad_pi = make_float3(0.f);
  __syncthreads();

#pragma unroll
  for (size_t m = 0; m < 27; __syncthreads(), ++m) {
    int3 cur_grid_xyz =
        grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
    const size_t hash_idx =
        xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
    if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
      continue;

    _ComputeAlpha(&alpha[i], &grad_pi, i, label, pos, mass, cellStart[hash_idx],
                  cellStart[hash_idx + 1], nablaW);
    _ComputeBoundaryAlpha(&grad_pi, pos[i], bPos, bVolume, rho0,
                          bCellStart[hash_idx], bCellStart[hash_idx + 1],
                          nablaW);
  }

  alpha[i] = -1.f / fmaxf(KIRI_EPSILON, alpha[i] + lengthSquared(grad_pi));

  return;
}

template <typename Pos2GridXYZ, typename GridXYZ2GridHash,
          typename GradientFunc>
__global__ void _ComputeDivgenceError_CUDA(
    float *stiff, float *densityError,const size_t *label, const float *alpha, const float3 *pos,
    const float3 *vel, const float *mass, const float *density,
    const float rho0, const float dt, const size_t num, size_t *cellStart,
    float3 *bPos, float *bVolume, size_t *bCellStart, const int3 gridSize,
    Pos2GridXYZ p2xyz, GridXYZ2GridHash xyz2hash, GradientFunc nablaW) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num || label[i] == 1)
    return;

  int3 grid_xyz = p2xyz(pos[i]);
  auto error = 0.f;
  __syncthreads();
#pragma unroll
  for (size_t m = 0; m < 27; __syncthreads(), ++m) {
    int3 cur_grid_xyz =
        grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
    const size_t hash_idx =
        xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
    if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
      continue;

    _ComputeDivergenceError(&error, i,label, pos, mass, vel, cellStart[hash_idx],
                            cellStart[hash_idx + 1], nablaW);
    _ComputeDivergenceErrorBoundary(&error, pos[i], vel[i], bPos, bVolume, rho0,
                                    bCellStart[hash_idx],
                                    bCellStart[hash_idx + 1], nablaW);
  }

  densityError[i] = fmaxf(error, 0.f);

  if (density[i] + dt * densityError[i] < rho0 && density[i] <= rho0)
    densityError[i] = 0.f;

  stiff[i] = densityError[i] * alpha[i];

  return;
}

template <typename Pos2GridXYZ, typename GridXYZ2GridHash,
          typename GradientFunc>
__global__ void _CorrectDivergenceByJacobi_CUDA(
    float3 *vel,const size_t *label,  const float *stiff, const float3 *pos, const float *mass,
    const float rho0, const size_t num, size_t *cellStart, const float3 *bPos,
    const float *bVolume, size_t *bCellStart, const int3 gridSize,
    Pos2GridXYZ p2xyz, GridXYZ2GridHash xyz2hash, GradientFunc nablaW) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num || label[i] == 1)
    return;

  int3 grid_xyz = p2xyz(pos[i]);

  __syncthreads();
#pragma unroll
  for (size_t m = 0; m < 27; __syncthreads(), ++m) {
    int3 cur_grid_xyz =
        grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
    const size_t hash_idx =
        xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
    if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
      continue;

    _AdaptVelocitiesByDivergence(&vel[i], i, label, stiff, pos, mass,
                                 cellStart[hash_idx], cellStart[hash_idx + 1],
                                 nablaW);

    _AdaptVelocitiesBoundaryByDivergence(&vel[i], pos[i], stiff[i], bPos,
                                         bVolume, rho0, bCellStart[hash_idx],
                                         bCellStart[hash_idx + 1], nablaW);
  }

  return;
}

template <typename Pos2GridXYZ, typename GridXYZ2GridHash,
          typename GradientFunc>
__global__ void _ComputeDensityError_CUDA(
    float *densityError, float *stiff,const size_t *label, const float *alpha, const float3 *pos,
    const float3 *vel, const float *mass, const float *density,
    const float rho0, const float dt, const size_t num, size_t *cellStart,
    float3 *bPos, float *bVolume, size_t *bCellStart, const int3 gridSize,
    Pos2GridXYZ p2xyz, GridXYZ2GridHash xyz2hash, GradientFunc nablaW) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num || label[i] == 1)
    return;

  int3 grid_xyz = p2xyz(pos[i]);
  auto error = 0.f;
  __syncthreads();
#pragma unroll
  for (size_t m = 0; m < 27; __syncthreads(), ++m) {
    int3 cur_grid_xyz =
        grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
    const size_t hash_idx =
        xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
    if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
      continue;

    _ComputeDivergenceError(&error, i, label, pos, mass, vel, cellStart[hash_idx],
                            cellStart[hash_idx + 1], nablaW);
    _ComputeDivergenceErrorBoundary(&error, pos[i], vel[i], bPos, bVolume, rho0,
                                    bCellStart[hash_idx],
                                    bCellStart[hash_idx + 1], nablaW);
  }

  densityError[i] = fmaxf(dt * error + density[i] - rho0, 0.f);
  stiff[i] = densityError[i] * alpha[i];

  return;
}

template <typename Pos2GridXYZ, typename GridXYZ2GridHash,
          typename GradientFunc>
__global__ void _CorrectPressureByJacobi_CUDA(
    float3 *vel, const size_t *label, const float *stiff, const float3 *pos, const float *mass,
    const float rho0, const float dt, const size_t num, size_t *cellStart,
    const float3 *bPos, const float *bVolume, size_t *bCellStart,
    const int3 gridSize, Pos2GridXYZ p2xyz, GridXYZ2GridHash xyz2hash,
    GradientFunc nablaW) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num || label[i] == 1)
    return;

  int3 grid_xyz = p2xyz(pos[i]);
  auto a = make_float3(0.0f);
  __syncthreads();
#pragma unroll
  for (size_t m = 0; m < 27; __syncthreads(), ++m) {
    int3 cur_grid_xyz =
        grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
    const size_t hash_idx =
        xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
    if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
      continue;

    _AdaptVelocitiesByPressure(&vel[i], i, label, stiff, pos, mass, dt,
                               cellStart[hash_idx], cellStart[hash_idx + 1],
                               nablaW);

    _AdaptVelocitiesBoundaryByPressure(&vel[i], pos[i], stiff[i], bPos, bVolume,
                                       rho0, dt, bCellStart[hash_idx],
                                       bCellStart[hash_idx + 1], nablaW);
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

template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename Func,
          typename GradientFunc>
__global__ void _ComputeDFSFWaterLinearMomentum_CUDA(
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

} // namespace KIRI

#endif /* _CUDA_DFSPH_SF_SOLVER_GPU_CUH_ */