/*
 * @Author: Xu.WANG
 * @Date: 2021-02-11 01:26:00
 * @LastEditTime: 2021-08-18 16:36:52
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath:
 * \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\data\cuda_boundary_params.cuh
 */

#ifndef _CUDA_BOUNDARY_PARAMS_CUH_
#define _CUDA_BOUNDARY_PARAMS_CUH_

#pragma once

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>

namespace KIRI {
struct CudaBoundaryParams {
  float kernel_radius;
  float3 lowest_point;
  float3 highest_point;
  float3 world_size;
  float3 world_center;
  int3 grid_size;
};

extern CudaBoundaryParams CUDA_BOUNDARY_PARAMS;

} // namespace KIRI

#endif