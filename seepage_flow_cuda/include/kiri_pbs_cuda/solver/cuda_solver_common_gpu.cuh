/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-05-14 20:01:11
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-05-17 20:37:52
 * @FilePath:
 * \sph_seepage_flows\seepage_flow_cuda\include\kiri_pbs_cuda\solver\cuda_solver_common_gpu.cuh
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */
#ifndef _CUDA_SOLVER_COMMON_GPU_CUH_
#define _CUDA_SOLVER_COMMON_GPU_CUH_

#pragma once

#include <curand.h>
#include <curand_kernel.h>
#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>
namespace KIRI {
// generates a random float between 0 and 1
static __device__ float RndFloat(curandState *globalState, int ind) {
  curandState local_state = globalState[ind];
  float val = curand_uniform(&local_state);
  globalState[ind] = local_state;
  return val;
}

static __global__ void SetUpRndGen_CUDA(curandState *state,
                                        unsigned long seed) {
  int id = threadIdx.x;
  curand_init(seed, id, 0, &state[id]);
}

} // namespace KIRI

#endif /* _CUDA_SOLVER_COMMON_GPU_CUH_ */