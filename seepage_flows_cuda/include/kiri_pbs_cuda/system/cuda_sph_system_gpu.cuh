/*
 * @Author: Xu.WANG
 * @Date: 2020-07-04 14:48:23
 * @LastEditTime: 2021-04-04 14:30:03
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\system\cuda_sph_system_gpu.cuh
 */

#ifndef _CUDA_SPH_SYSTEM_GPU_CUH_
#define _CUDA_SPH_SYSTEM_GPU_CUH_

#pragma once

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>

namespace KIRI
{
    template <typename Func>
    __device__ void ComputeBoundaryVolume(
        float *delta,
        const size_t i,
        float3 *pos,
        size_t j,
        const size_t cellEnd,
        Func W)
    {
        while (j < cellEnd)
        {
            if (i != j)
                *delta += W(length(pos[i] - pos[j]));
            ++j;
        }
        return;
    }

    template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename Func>
    __global__ void ComputeBoundaryVolume_CUDA(
        float3 *pos,
        float *volume,
        const size_t num,
        size_t *cellStart,
        const int3 gridSize,
        Pos2GridXYZ p2xyz,
        GridXYZ2GridHash xyz2hash,
        Func W)
    {
        const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (i >= num)
            return;

        int3 grid_xyz = p2xyz(pos[i]);

#pragma unroll
        for (int m = 0; m < 27; ++m)
        {
            int3 cur_grid_xyz = grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
            const size_t hash_idx = xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
            if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
                continue;

            ComputeBoundaryVolume(&volume[i], i, pos, cellStart[hash_idx], cellStart[hash_idx + 1], W);
        }
        //printf("volume=%.3f \n", volume[i]);
        volume[i] = 1.f / fmaxf(volume[i], KIRI_EPSILON);
        return;
    }

} // namespace KIRI

#endif /* _CUDA_SPH_SYSTEM_GPU_CUH_ */