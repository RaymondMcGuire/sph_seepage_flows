/*
 * @Author: Xu.WANG
 * @Date: 2020-07-04 14:48:23
 * @LastEditTime: 2021-08-16 00:34:17
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath:
 * \KiriPBSCuda\include\kiri_pbs_cuda\solver\seepageflow\cuda_wcsph_sf_solver_gpu.cuh
 */

#ifndef _CUDA_WCSPH_SF_SOLVER_GPU_CUH_
#define _CUDA_WCSPH_SF_SOLVER_GPU_CUH_

#pragma once

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>

namespace KIRI {
static __global__ void
ComputeSFPressureByTait_CUDA(size_t *label, float *density, float *pressure,
                             const size_t num, const float rho0,
                             const float stiff, const float negativeScale) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;

  if (label[i] == 0) {
    pressure[i] = stiff * (powf((density[i] / rho0), 7.f) - 1.f);
    if (pressure[i] < 0.f)
      pressure[i] *= negativeScale;
  } else if (label[i] == 1) {
    pressure[i] = 0.f;
  }

  return;
}
} // namespace KIRI

#endif /* _CUDA_WCSPH_SF_SOLVER_GPU_CUH_ */