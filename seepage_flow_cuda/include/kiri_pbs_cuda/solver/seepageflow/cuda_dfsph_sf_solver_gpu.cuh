/*** 
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-04-08 12:28:00
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-04-08 12:33:07
 * @FilePath: \sph_seepage_flows\seepage_flow_cuda\include\kiri_pbs_cuda\solver\seepageflow\cuda_dfsph_sf_solver_gpu.cuh
 * @Description: 
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved. 
 */
#ifndef _CUDA_DFSPH_SF_SOLVER_GPU_CUH_
#define _CUDA_DFSPH_SF_SOLVER_GPU_CUH_

#pragma once

#include <kiri_pbs_cuda/solver/sph/cuda_dfsph_solver_common_gpu.cuh>
#include <kiri_pbs_cuda/solver/seepageflow/cuda_sph_sf_solver_gpu.cuh>
namespace KIRI {

static __global__ void _ComputeVelMag_CUDA(float *velMag, const size_t *label,
                                           const float3 *vel, const float3 *acc,
                                           const float dt, const size_t num) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;
  velMag[i] = 0.f;

  if (label[i] == 1)
    return;

  velMag[i] = lengthSquared(vel[i] + acc[i] * dt);

  return;
}

static __global__ void
_ComputeDFSFPressure_CUDA(float *pressure, const size_t *label,
                          const float *density, const float *stiff,
                          const size_t num, const float rho0,
                          const float negativeScale) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;

  if (label[i] == 0) {
    pressure[i] = stiff[i] * (density[i] - rho0);

    if (pressure[i] < 0.f)
      pressure[i] *= negativeScale;
  } else if (label[i] == 1) {
    pressure[i] = 0.f;
  }

  return;
}

template <typename Pos2GridXYZ, typename GridXYZ2GridHash,
          typename GradientFunc>
__global__ void
_ComputeAlpha_CUDA(float *alpha, const size_t *label, const float3 *pos,
                   const float *mass, const float *density, const float rho0,
                   const size_t num, size_t *cellStart, const float3 *bPos,
                   const float *bVolume, size_t *bCellStart,
                   const int3 gridSize, Pos2GridXYZ p2xyz,
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

  if (alpha[i] != alpha[i])
    printf("alpha NaN! \n");

  return;
}

template <typename Pos2GridXYZ, typename GridXYZ2GridHash,
          typename GradientFunc>
__global__ void _ComputeDivgenceError_CUDA(
    float *stiff, float *densityError, const size_t *label, const float *alpha,
    const float3 *pos, const float3 *vel, const float *mass,
    const float *density, const float rho0, const float dt, const size_t num,
    size_t *cellStart, float3 *bPos, float *bVolume, size_t *bCellStart,
    const int3 gridSize, Pos2GridXYZ p2xyz, GridXYZ2GridHash xyz2hash,
    GradientFunc nablaW) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;

  if (label[i] == 1) {
    stiff[i] = 0.f;
    return;
  }

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

    _ComputeDivergenceError(&error, i, label, pos, mass, vel,
                            cellStart[hash_idx], cellStart[hash_idx + 1],
                            nablaW);
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
    float3 *vel, const size_t *label, const float *stiff, const float3 *pos,
    const float *mass, const float rho0, const size_t num, size_t *cellStart,
    const float3 *bPos, const float *bVolume, size_t *bCellStart,
    const int3 gridSize, Pos2GridXYZ p2xyz, GridXYZ2GridHash xyz2hash,
    GradientFunc nablaW) {
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
    float *densityError, float *stiff, const size_t *label, const float *alpha,
    const float3 *pos, const float3 *vel, const float *mass,
    const float *density, const float rho0, const float dt, const size_t num,
    size_t *cellStart, float3 *bPos, float *bVolume, size_t *bCellStart,
    const int3 gridSize, Pos2GridXYZ p2xyz, GridXYZ2GridHash xyz2hash,
    GradientFunc nablaW) {
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

    _ComputeDivergenceError(&error, i, label, pos, mass, vel,
                            cellStart[hash_idx], cellStart[hash_idx + 1],
                            nablaW);
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
    float3 *vel, const size_t *label, const float *stiff, const float3 *pos,
    const float *mass, const float rho0, const float dt, const size_t num,
    size_t *cellStart, const float3 *bPos, const float *bVolume,
    size_t *bCellStart, const int3 gridSize, Pos2GridXYZ p2xyz,
    GridXYZ2GridHash xyz2hash, GradientFunc nablaW) {
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
__device__ void _ComputeDFSFSandBuoyancyForces(
    float3 *f, float *vW, const size_t i, const size_t *label,
    const float3 *pos, const float *mass, const float *density,
    const float *stiff, const float *voidage, size_t j, const size_t cellEnd,
    Func W, GradientFunc nablaW) {
  while (j < cellEnd) {
    if (i != j && label[i] == 1 && label[j] == 0 && voidage[j] > 0.f &&
        voidage[j] <= 1.f) {
      float densityj = density[j] / fmaxf(KIRI_EPSILON, voidage[j]);

      float3 dij = pos[i] - pos[j];
      float wij = W(length(dij));
      float3 nabla_wij = nablaW(dij);

      *vW += mass[j] / densityj * wij;
      *f += -mass[i] * mass[j] * stiff[j] * nabla_wij;
    }
    ++j;
  }
  return;
}

template <typename Func, typename GradientFunc>
__device__ void _ComputeDFSFWaterSeepageTerm(
    float3 *a, const size_t i, const size_t *label, const float3 *pos,
    const float *mass, const float *density, const float *stiff,
    const float *voidage, const float3 *avgDragForce, size_t j,
    const size_t cellEnd, Func W, GradientFunc nablaW) {
  while (j < cellEnd) {
    if (i != j && label[i] == 0 && label[j] == 1) {
      float densityi = density[i] / fmaxf(KIRI_EPSILON, voidage[i]);

      float3 dij = pos[i] - pos[j];
      float wij = W(length(dij));
      float3 nabla_wij = nablaW(dij);

      // buoyancy term
      float3 bouyancy = -mass[j] * stiff[i] * nabla_wij;
      // drag term
      float3 drag = -wij / densityi * avgDragForce[j];

      *a += bouyancy + drag;
    }
    ++j;
  }
  return;
}

template <typename GradientFunc>
__device__ void _ComputeDFSFWaterPressureTerm(
    float3 *a, const size_t i, const size_t *label, const float3 *pos,
    const float *mass, const float *density, const float *stiff,
    const float *voidage, size_t j, const size_t cellEnd, GradientFunc nablaW) {
  while (j < cellEnd) {
    if (i != j && label[i] == label[j]) {
      float densityi = density[i] / fmaxf(KIRI_EPSILON, voidage[i]);
      float densityj = density[j] / fmaxf(KIRI_EPSILON, voidage[j]);
      *a += -mass[j] * (stiff[i] + stiff[j]) * nablaW(pos[i] - pos[j]);
    }
    ++j;
  }

  return;
}

template <typename GradientFunc>
__device__ void
_ComputeDFSFBoundaryPressure(float3 *a, const float3 posi, const float densityi,
                             const float stiffi, const float3 *bpos,
                             const float *volume, const float rho0, size_t j,
                             const size_t cellEnd, GradientFunc nablaW) {
  while (j < cellEnd) {
    *a += -rho0 * volume[j] * stiffi * nablaW(posi - bpos[j]);
    ++j;
  }
  return;
}

template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename Func,
          typename GradientFunc>
__global__ void _ComputeDFSFWaterLinearMomentum_CUDA(
    float3 *acc, const size_t *label, const float3 *pos, const float3 *vel,
    const float *mass, const float *density, const float *stiff,
    const float *voidage, const float3 *avgDragForce,
    const float3 *adhesionForce, const float rho0, const float nu,
    const float bnu, const size_t num, size_t *cellStart, const float3 *bPos,
    const float *bVolume, size_t *bCellStart, const int3 gridSize,
    Pos2GridXYZ p2xyz, GridXYZ2GridHash xyz2hash, Func W, GradientFunc nablaW) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num || label[i] == 1)
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

    // _ComputeDFSFWaterPressureTerm(&a, i, label, pos, mass, density, stiff,
    //                             voidage, cellStart[hash_idx],
    //                             cellStart[hash_idx + 1], nablaW);
    _ComputeSFWaterArtificialViscosity(&a, i, label, pos, vel, mass, density,
                                       voidage, nu, cellStart[hash_idx],
                                       cellStart[hash_idx + 1], nablaW);
    _ComputeDFSFWaterSeepageTerm(&a, i, label, pos, mass, density, stiff,
                                 voidage, avgDragForce, cellStart[hash_idx],
                                 cellStart[hash_idx + 1], W, nablaW);

    // _ComputeDFSFBoundaryPressure(&a, pos[i], rho0, stiff[i], bPos, bVolume,
    // rho0,
    //                          bCellStart[hash_idx], bCellStart[hash_idx + 1],
    //                          nablaW);
    _ComputeBoundaryViscosity(&a, pos[i], bPos, vel[i], rho0, bVolume, bnu,
                              rho0, bCellStart[hash_idx],
                              bCellStart[hash_idx + 1], nablaW);
  }

  acc[i] += a - adhesionForce[i] / mass[i];
  return;
}

template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename AttenuFunc,
          typename Func, typename GradientFunc>
__global__ void _ComputeDFSFSandLinearMomentum_CUDA(
    float3 *avgDragForce, float3 *acc, float3 *angularAcc, const size_t *label,
    const float3 *pos, const float3 *vel, const float3 *angularVel,
    const float *mass, const float *inertia, const float *density,
    const float *stiff, const float *voidage, const float *saturation,
    const float3 *avgFlowVel, const float3 *avgAdhesionForce,
    const float *radius, const float boundaryRadius, const float maxForceFactor,
    const float young, const float poisson, const float tanFrictionAngle,
    const float cd, const float gravity, const float rho0, const size_t num,
    const float3 lowestPoint, const float3 highestPoint, size_t *cellStart,
    const float3 *bPos, const size_t *bLabel, size_t *bCellStart,
    const int3 gridSize, Pos2GridXYZ p2xyz, GridXYZ2GridHash xyz2hash,
    AttenuFunc G, Func W, GradientFunc nablaW) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num || label[i] == 0)
    return;

  // printf("_ComputeDFSFSandLinearMomentum_CUDA \n");

  float v_w = 0.f;
  float3 f = make_float3(0.f);
  float3 torque = make_float3(0.f);
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
    _ComputeSFMSDEMForcesTorque(&f, &torque, i, label, pos, vel, radius,
                                angularVel, young, poisson, tanFrictionAngle,
                                cellStart[hash_idx], cellStart[hash_idx + 1]);

    _ComputeSFMSDEMCapillaryForces(&f, i, label, pos, saturation, radius,
                                   maxForceFactor, cellStart[hash_idx],
                                   cellStart[hash_idx + 1], G);

    // sand-water interactive(buoyancy term)
    _ComputeDFSFSandBuoyancyForces(&f, &v_w, i, label, pos, mass, density,
                                   stiff, voidage, cellStart[hash_idx],
                                   cellStart[hash_idx + 1], W, nablaW);

    // adhesion
    _ComputeSFSandAdhesionTerm(&f, i, label, pos, mass, density,
                               avgAdhesionForce, cellStart[hash_idx],
                               cellStart[hash_idx + 1], W);

    // scene boundary particles interactive
    _ComputeSFMSDEMBoundaryForcesTorque(
        &f, &torque, pos[i], vel[i], angularVel[i], radius[i], boundaryRadius,
        bLabel, bPos, young, poisson, tanFrictionAngle, bCellStart[hash_idx],
        bCellStart[hash_idx + 1]);
  }

  _ComputeDEMWorldBoundaryForcesTorque(
      &f, &torque, pos[i], vel[i], angularVel[i], radius[i], boundaryRadius,
      young, poisson, tanFrictionAngle, num, lowestPoint, highestPoint);

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
  angularAcc[i] = 2.f * torque / inertia[i];
  return;
}

template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename AttenuFunc,
          typename Func, typename GradientFunc>
__global__ void _ComputeMultiDFSFSandLinearMomentum_CUDA(
    float3 *avgDragForce, float3 *acc, float3 *angularAcc, const size_t *label,
    const float3 *pos, const float3 *vel, const float3 *angularVel,
    const float *mass, const float *inertia, const float *density,
    const float *stiff, const float *voidage, const float *saturation,
    const float3 *avgFlowVel, const float3 *avgAdhesionForce,
    const float *radius, const float boundaryRadius, const float maxForceFactor,
    const float young, const float poisson, const float tanFrictionAngle,
    const float3 *cda0asat, const float gravity, const float rho0,
    const size_t num, const float3 lowestPoint, const float3 highestPoint,
    size_t *cellStart, const float3 *bPos, const size_t *bLabel,
    size_t *bCellStart, const int3 gridSize, Pos2GridXYZ p2xyz,
    GridXYZ2GridHash xyz2hash, AttenuFunc G, Func W, GradientFunc nablaW) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num || label[i] == 0)
    return;

  // printf("_ComputeMultiDFSFSandLinearMomentum_CUDA \n");

  float v_w = 0.f;
  float3 f = make_float3(0.f);
  float3 torque = make_float3(0.f);
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
    _ComputeSFMSDEMForcesTorque(&f, &torque, i, label, pos, vel, radius,
                                angularVel, young, poisson, tanFrictionAngle,
                                cellStart[hash_idx], cellStart[hash_idx + 1]);
    _ComputeSFMSDEMCapillaryForces(&f, i, label, pos, saturation, radius,
                                   maxForceFactor, cellStart[hash_idx],
                                   cellStart[hash_idx + 1], G);

    // sand-water interactive(buoyancy term)
    _ComputeDFSFSandBuoyancyForces(&f, &v_w, i, label, pos, mass, density,
                                   stiff, voidage, cellStart[hash_idx],
                                   cellStart[hash_idx + 1], W, nablaW);

    // adhesion
    _ComputeSFSandAdhesionTerm(&f, i, label, pos, mass, density,
                               avgAdhesionForce, cellStart[hash_idx],
                               cellStart[hash_idx + 1], W);

    // scene boundary particles interactive
    _ComputeSFMSDEMBoundaryForcesTorque(
        &f, &torque, pos[i], vel[i], angularVel[i], radius[i], boundaryRadius,
        bLabel, bPos, young, poisson, tanFrictionAngle, bCellStart[hash_idx],
        bCellStart[hash_idx + 1]);
  }

  _ComputeDEMWorldBoundaryForcesTorque(
      &f, &torque, pos[i], vel[i], angularVel[i], radius[i], boundaryRadius,
      young, poisson, tanFrictionAngle, num, lowestPoint, highestPoint);

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
  angularAcc[i] = 2.f * torque / inertia[i];
  return;
}

} // namespace KIRI

#endif /* _CUDA_DFSPH_SF_SOLVER_GPU_CUH_ */