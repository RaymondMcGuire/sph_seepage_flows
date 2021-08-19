/*
 * @Author: Xu.WANG
 * @Date: 2020-07-04 14:48:23
 * @LastEditTime: 2021-08-16 00:29:11
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriPBSCuda\include\kiri_pbs_cuda\solver\seepageflow\cuda_iisph_sf_solver_gpu.cuh
 */

#ifndef _CUDA_IISPH_SF_SOLVER_GPU_CUH_
#define _CUDA_IISPH_SF_SOLVER_GPU_CUH_

#pragma once

#include <kiri_pbs_cuda/solver/seepageflow/cuda_sph_sf_solver_gpu.cuh>
#include <kiri_pbs_cuda/solver/sph/cuda_iisph_solver_common_gpu.cuh>

namespace KIRI
{
    template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename GradientFunc>
    __global__ void ComputeIISFDiiTerm_CUDA(
        float3 *dii,
        const size_t *label,
        const float3 *pos,
        const float *mass,
        const float *density,
        const float rho0,
        const size_t num,
        size_t *cellStart,
        const float3 *bPos,
        const float *bVolume,
        size_t *bCellStart,
        const int3 gridSize,
        Pos2GridXYZ p2xyz,
        GridXYZ2GridHash xyz2hash,
        GradientFunc nablaW)
    {
        const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (i >= num || label[i] != 0)
            return;

        int3 grid_xyz = p2xyz(pos[i]);
        dii[i] = make_float3(0.f);

#pragma unroll
        for (size_t m = 0; m < 27; ++m)
        {
            int3 cur_grid_xyz = grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
            const size_t hash_idx = xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
            if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
                continue;

            ComputeDii(&dii[i], i, pos, mass, density[i], cellStart[hash_idx], cellStart[hash_idx + 1], nablaW);
            ComputeBoundaryDii(&dii[i], pos[i], density[i], bPos, bVolume, rho0, bCellStart[hash_idx], bCellStart[hash_idx + 1], nablaW);
        }

        return;
    }

    template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename GradientFunc>
    __global__ void ComputeIISFAiiTerm_CUDA(
        float *aii,
        float *densityAdv,
        float *pressure,
        const size_t *label,
        const float3 *dii,
        const float3 *pos,
        const float3 *vel,
        const float3 *acc,
        const float *mass,
        const float *density,
        const float *lastPressure,
        const float rho0,
        const float dt,
        const size_t num,
        size_t *cellStart,
        float3 *bPos,
        float *bVolume,
        size_t *bCellStart,
        const int3 gridSize,
        Pos2GridXYZ p2xyz,
        GridXYZ2GridHash xyz2hash,
        GradientFunc nablaW)
    {
        const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (i >= num || label[i] != 0)
            return;

        int3 grid_xyz = p2xyz(pos[i]);
        aii[i] = 0.f;
        densityAdv[i] = density[i];
        pressure[i] = 0.5f * lastPressure[i];

        float dpi = mass[i] / fmaxf(KIRI_EPSILON, density[i] * density[i]);

#pragma unroll
        for (size_t m = 0; m < 27; ++m)
        {
            int3 cur_grid_xyz = grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
            const size_t hash_idx = xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
            if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
                continue;

            ComputeDensityAdv(&densityAdv[i], i, pos, mass, vel, dt, cellStart[hash_idx], cellStart[hash_idx + 1], nablaW);
            ComputeBoundaryDensityAdv(&densityAdv[i], pos[i], vel[i], dt, bPos, bVolume, rho0, bCellStart[hash_idx], bCellStart[hash_idx + 1], nablaW);

            ComputeAii(&aii[i], i, pos, mass, dpi, dii[i], cellStart[hash_idx], cellStart[hash_idx + 1], nablaW);
            ComputeBoundaryAii(&aii[i], pos[i], dpi, dii[i], bPos, bVolume, rho0, bCellStart[hash_idx], bCellStart[hash_idx + 1], nablaW);
        }

        return;
    }

    template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename GradientFunc>
    __global__ void ComputeIISFDijPjTerm_CUDA(
        float3 *dijpj,
        const size_t *label,
        const float3 *pos,
        const float *mass,
        const float *density,
        const float *lastPressure,
        const size_t num,
        size_t *cellStart,
        const int3 gridSize,
        Pos2GridXYZ p2xyz,
        GridXYZ2GridHash xyz2hash,
        GradientFunc nablaW)
    {
        const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (i >= num || label[i] != 0)
            return;

        int3 grid_xyz = p2xyz(pos[i]);
        dijpj[i] = make_float3(0.f);

#pragma unroll
        for (size_t m = 0; m < 27; ++m)
        {
            int3 cur_grid_xyz = grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
            const size_t hash_idx = xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
            if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
                continue;

            ComputeDijPj(&dijpj[i], i, pos, mass, density, lastPressure, cellStart[hash_idx], cellStart[hash_idx + 1], nablaW);
        }

        return;
    }

    template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename GradientFunc>
    __global__ void ComputeIISFPressureByJacobi_CUDA(
        float *pressure,
        float *lastPressure,
        float *densityError,
        const size_t *label,
        const float *aii,
        const float3 *dijpj,
        const float3 *dii,
        const float *densityAdv,
        const float3 *pos,
        const float *mass,
        const float *density,
        const float rho0,
        const float dt,
        const size_t num,
        size_t *cellStart,
        const float3 *bPos,
        const float *bVolume,
        size_t *bCellStart,
        const int3 gridSize,
        Pos2GridXYZ p2xyz,
        GridXYZ2GridHash xyz2hash,
        GradientFunc nablaW)
    {
        const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (i >= num || label[i] != 0)
            return;

        int3 grid_xyz = p2xyz(pos[i]);
        pressure[i] = 0.f;
        float dpi = mass[i] / fmaxf(KIRI_EPSILON, density[i] * density[i]);

        float sum = 0.f;
#pragma unroll
        for (size_t m = 0; m < 27; ++m)
        {
            int3 cur_grid_xyz = grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
            const size_t hash_idx = xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
            if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
                continue;

            ComputePressureSumParts(&sum, i, dijpj, dii, lastPressure, pos, mass, dpi, cellStart[hash_idx], cellStart[hash_idx + 1], nablaW);
            ComputeBoundaryPressureSumParts(&sum, pos[i], dijpj[i], bPos, bVolume, rho0, bCellStart[hash_idx], bCellStart[hash_idx + 1], nablaW);
        }

        __syncthreads();

        const float omega = 0.5f;
        const float h2 = dt * dt;
        const float b = rho0 - densityAdv[i];
        const float lpi = lastPressure[i];
        const float denom = aii[i] * h2;
        if (abs(denom) > KIRI_EPSILON)
            pressure[i] = max((1.f - omega) * lpi + omega / denom * (b - h2 * sum), 0.f);
        else
            pressure[i] = 0.f;

        if (pressure[i] != 0.f)
        {
            const float newDensity = rho0 * ((aii[i] * pressure[i] + sum) * h2 - b) + rho0;
            densityError[i] = newDensity - rho0;
        }

        lastPressure[i] = pressure[i];
        return;
    }

    template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename GradientFunc>
    __global__ void ComputeIISFPressureAcceleration_CUDA(
        float3 *pacc,
        const size_t *label,
        const float3 *pos,
        const float *mass,
        const float *density,
        const float *pressure,
        const float rho0,
        const size_t num,
        size_t *cellStart,
        const float3 *bPos,
        const float *bVolume,
        size_t *bCellStart,
        const int3 gridSize,
        Pos2GridXYZ p2xyz,
        GridXYZ2GridHash xyz2hash,
        GradientFunc nablaW)
    {
        const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (i >= num || label[i] != 0)
            return;

        int3 grid_xyz = p2xyz(pos[i]);
        pacc[i] = make_float3(0.f);
        float dpi = pressure[i] / fmaxf(KIRI_EPSILON, density[i] * density[i]);

#pragma unroll
        for (size_t m = 0; m < 27; ++m)
        {
            int3 cur_grid_xyz = grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
            const size_t hash_idx = xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
            if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
                continue;

            ComputePressureAcc(&pacc[i], i, pressure, pos, mass, density, dpi, cellStart[hash_idx], cellStart[hash_idx + 1], nablaW);
            ComputeBoundaryPressureAcc(&pacc[i], pos[i], dpi, bPos, bVolume, rho0, bCellStart[hash_idx], bCellStart[hash_idx + 1], nablaW);
        }

        return;
    }

    template <typename Func>
    __device__ void ComputeIISFNeighborWaterVolume(
        float *v_w,
        const size_t i,
        const size_t *label,
        const float3 *pos,
        const float *mass,
        const float *density,
        const float *voidage,
        size_t j,
        const size_t cellEnd,
        Func W)
    {
        while (j < cellEnd)
        {
            if (i != j && label[i] == 1 && label[j] == 0 && voidage[j] > 0.f && voidage[j] <= 1.f)
            {
                float densityj = density[j] / fmaxf(KIRI_EPSILON, voidage[j]);

                float3 dij = pos[i] - pos[j];
                float wij = W(length(dij));

                *v_w += mass[j] / densityj * wij;
            }
            ++j;
        }
        return;
    }

    // no buoyancy
    template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename AttenuFunc, typename Func>
    __global__ void ComputeIISFSandLMNB_CUDA(
        size_t *label,
        float3 *pos,
        float3 *vel,
        float3 *acc,
        float *mass,
        float *density,
        float *voidage,
        float *saturation,
        float3 *avgFlowVel,
        float3 *avgDragForce,
        float3 *avgAdhesionForce,
        const float sandRadius,
        const float waterRadius,
        const float young,
        const float poisson,
        const float tanFrictionAngle,
        const float cd,
        const float gravity,
        const float rho0,
        const size_t num,
        const float3 lowestPoint,
        const float3 highestPoint,
        size_t *cellStart,
        float3 *bPos,
        size_t *bLabel,
        size_t *bCellStart,
        const int3 gridSize,
        Pos2GridXYZ p2xyz,
        GridXYZ2GridHash xyz2hash,
        AttenuFunc G,
        Func W)
    {
        const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (i >= num || label[i] != 1)
            return;

        float v_w = 0.f;
        float3 f = make_float3(0.f);
        float3 df = make_float3(0.f);
        int3 grid_xyz = p2xyz(pos[i]);

#pragma unroll
        for (size_t m = 0; m < 27; ++m)
        {
            int3 cur_grid_xyz = grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
            const size_t hash_idx = xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
            if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
                continue;

            // sand particles
            ComputeSFUniRadiusDemForces(&f, i, label, pos, vel, sandRadius, young, poisson, tanFrictionAngle, cellStart[hash_idx], cellStart[hash_idx + 1]);
            ComputeSFUniRadiusDemCapillaryForces(&f, i, label, pos, vel, saturation, sandRadius, cellStart[hash_idx], cellStart[hash_idx + 1], G);

            // water volume
            ComputeIISFNeighborWaterVolume(&v_w, i, label, pos, mass, density, voidage, cellStart[hash_idx], cellStart[hash_idx + 1], W);

            // adhesion
            ComputeSFSandAdhesionTerm(&f, i, label, pos, mass, density, avgAdhesionForce, cellStart[hash_idx], cellStart[hash_idx + 1], W);

            // scene boundary particles interactive
            ComputeSFUniRadiusDemBoundaryForces(&f, pos[i], -vel[i], bLabel, bPos, sandRadius, young, poisson, tanFrictionAngle, bCellStart[hash_idx], bCellStart[hash_idx + 1]);
        }

        ComputeDemWorldBoundaryForces(&f, pos[i], -vel[i], sandRadius, waterRadius, young, poisson, tanFrictionAngle, num, lowestPoint, highestPoint);

        // sand-water interactive(drag term)
        if (voidage[i] < 1.f && voidage[i] > 0.f)
        {
            float dragCoeff = cd * powf(voidage[i], 3.f) / powf(1.f - voidage[i], 2.f);

            if (dragCoeff != 0.f)
                df = gravity * rho0 * powf(voidage[i], 2.f) / dragCoeff * (avgFlowVel[i] - vel[i]) * mass[i] / density[i];
        }

        if (v_w != 0.f)
            avgDragForce[i] = df / v_w;
        else
            avgDragForce[i] = make_float3(0.f);

        acc[i] += 2.f * (f + df) / mass[i];
        return;
    }

    template <typename GradientFunc>
    __device__ void ComputeIISFSandBuoyancyForces(
        float3 *f,
        const size_t i,
        const size_t *label,
        const float3 *pos,
        const float *mass,
        const float *density,
        const float *pressure,
        const float *voidage,
        size_t j,
        const size_t cellEnd,
        GradientFunc nablaW)
    {
        while (j < cellEnd)
        {
            if (i != j && label[i] == 1 && label[j] == 0 && voidage[j] > 0.f && voidage[j] <= 1.f)
            {
                float densityi = density[i];
                float densityj = density[j] / fmaxf(KIRI_EPSILON, voidage[j]);

                float3 dij = pos[i] - pos[j];
                float3 nabla_wij = nablaW(dij);

                *f += -mass[i] * mass[j] * pressure[j] / fmaxf(KIRI_EPSILON, densityi * densityj) * nabla_wij;
            }
            ++j;
        }
        return;
    }

    // sand buoyancy
    template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename GradientFunc>
    __global__ void ComputeIISFSandLMWB_CUDA(
        float3 *acc,
        const size_t *label,
        const float3 *pos,
        const float *mass,
        const float *density,
        const float *pressure,
        const float *voidage,
        const size_t num,
        size_t *cellStart,
        const int3 gridSize,
        Pos2GridXYZ p2xyz,
        GridXYZ2GridHash xyz2hash,
        GradientFunc nablaW)
    {
        const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (i >= num || label[i] != 1)
            return;

        float3 f = make_float3(0.f);
        int3 grid_xyz = p2xyz(pos[i]);

#pragma unroll
        for (size_t m = 0; m < 27; ++m)
        {
            int3 cur_grid_xyz = grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
            const size_t hash_idx = xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
            if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
                continue;

            ComputeIISFSandBuoyancyForces(&f, i, label, pos, mass, density, pressure, voidage, cellStart[hash_idx], cellStart[hash_idx + 1], nablaW);
        }

        acc[i] += 2.f * f / mass[i];
        return;
    }

    template <typename Func>
    __device__ void ComputeIISFWaterDragTerm(
        float3 *a,
        const size_t i,
        const size_t *label,
        const float3 *pos,
        const float *density,
        const float *voidage,
        const float3 *avgDragForce,
        size_t j,
        const size_t cellEnd,
        Func W)
    {
        while (j < cellEnd)
        {
            if (i != j && label[i] == 0 && label[j] == 1)
            {
                float densityi = density[i] / fmaxf(KIRI_EPSILON, voidage[i]);

                float3 dij = pos[i] - pos[j];
                float wij = W(length(dij));

                // drag term
                float3 drag = -wij / densityi * avgDragForce[j];

                *a += drag;
            }
            ++j;
        }
        return;
    }

    template <typename GradientFunc>
    __device__ void ComputeIISFWaterBuoyancyTerm(
        float3 *a,
        const size_t i,
        const size_t *label,
        const float3 *pos,
        const float *mass,
        const float *density,
        const float *pressure,
        const float *voidage,
        size_t j,
        const size_t cellEnd,
        GradientFunc nablaW)
    {
        while (j < cellEnd)
        {
            if (i != j && label[i] == 0 && label[j] == 1)
            {
                float densityi = density[i] / fmaxf(KIRI_EPSILON, voidage[i]);
                float densityj = density[j];

                float3 dij = pos[i] - pos[j];
                float3 nabla_wij = nablaW(dij);

                // buoyancy term
                float3 bouyancy = -mass[j] * pressure[i] / fmaxf(KIRI_EPSILON, densityi * densityj) * nabla_wij;

                *a += bouyancy;
            }
            ++j;
        }
        return;
    }

    template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename Func, typename GradientFunc>
    __global__ void ComputeIISFWaterLMNB_CUDA(
        float3 *acc,
        const size_t *label,
        const float3 *pos,
        const float3 *vel,
        const float *mass,
        const float *density,
        const float *voidage,
        const float3 *avgDragForce,
        const float3 *adhesionForce,
        const float rho0,
        const float nu,
        const float bnu,
        const size_t num,
        size_t *cellStart,
        const float3 *bPos,
        const float *bVolume,
        size_t *bCellStart,
        const int3 gridSize,
        Pos2GridXYZ p2xyz,
        GridXYZ2GridHash xyz2hash,
        Func W,
        GradientFunc nablaW)
    {
        const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (i >= num || label[i] != 0)
            return;

        float3 a = make_float3(0.f);
        int3 grid_xyz = p2xyz(pos[i]);

#pragma unroll
        for (size_t m = 0; m < 27; ++m)
        {
            int3 cur_grid_xyz = grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
            const size_t hash_idx = xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
            if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
                continue;

            ComputeSFWaterArtificialViscosity(&a, i, label, pos, vel, mass, density, voidage, nu, cellStart[hash_idx], cellStart[hash_idx + 1], nablaW);

            ComputeIISFWaterDragTerm(&a, i, label, pos, density, voidage, avgDragForce, cellStart[hash_idx], cellStart[hash_idx + 1], W);

            ComputeBoundaryViscosity(&a, pos[i], bPos, vel[i], rho0, bVolume, bnu, rho0, bCellStart[hash_idx], bCellStart[hash_idx + 1], nablaW);
        }

        acc[i] += a - adhesionForce[i] / mass[i];
        return;
    }

    template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename GradientFunc>
    __global__ void ComputeIISFWaterLMWB_CUDA(
        float3 *pacc,
        const size_t *label,
        const float3 *pos,
        const float *mass,
        const float *density,
        const float *pressure,
        const float *voidage,
        const size_t num,
        size_t *cellStart,
        const int3 gridSize,
        Pos2GridXYZ p2xyz,
        GridXYZ2GridHash xyz2hash,
        GradientFunc nablaW)
    {
        const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (i >= num || label[i] != 0)
            return;

        float3 a = make_float3(0.f);
        int3 grid_xyz = p2xyz(pos[i]);

#pragma unroll
        for (size_t m = 0; m < 27; ++m)
        {
            int3 cur_grid_xyz = grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
            const size_t hash_idx = xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
            if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
                continue;

            ComputeIISFWaterBuoyancyTerm(&a, i, label, pos, mass, density, pressure, voidage, cellStart[hash_idx], cellStart[hash_idx + 1], nablaW);
        }

        pacc[i] += a;
        return;
    }

} // namespace KIRI

#endif /* _CUDA_IISPH_SF_SOLVER_GPU_CUH_ */