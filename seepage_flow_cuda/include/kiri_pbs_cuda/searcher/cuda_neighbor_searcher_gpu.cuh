/*
 * @Author: Xu.WANG
 * @Date: 2020-10-18 02:13:36
 * @LastEditTime: 2021-03-19 23:57:05
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath:
 * \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\searcher\cuda_neighbor_searcher_gpu.cuh
 */
#ifndef _CUDA_NEIGHBOR_SEARCHER_GPU_CUH_
#define _CUDA_NEIGHBOR_SEARCHER_GPU_CUH_

#pragma once

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>
namespace KIRI {

__global__ void CountingInCell_CUDA(size_t *cellStart, size_t *particle2cell,
                                    const size_t num) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;
  atomicAdd(&cellStart[particle2cell[i]], 1);
  return;
}
} // namespace KIRI
#endif /* _CUDA_NEIGHBOR_SEARCHER_GPU_CUH_ */