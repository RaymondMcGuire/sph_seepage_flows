/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-03-25 22:02:18
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-04-08 11:51:39
 * @FilePath:
 * \sph_seepage_flows\seepage_flow_cuda\include\kiri_pbs_cuda\solver\cuda_base_solver.cuh
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */
#ifndef _CUDA_BASE_SOLVER_CUH_
#define _CUDA_BASE_SOLVER_CUH_

#pragma once

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>

namespace KIRI {
class CudaBaseSolver {
public:
  explicit CudaBaseSolver(const size_t num)
      : mCudaGridSize(CuCeilDiv(num, KIRI_CUBLOCKSIZE)) {}

  virtual ~CudaBaseSolver() noexcept {}
  inline float GetCurrentTimeSteps() const { return mDt; }

protected:
  size_t mCudaGridSize;
  float mDt;
};

typedef SharedPtr<CudaBaseSolver> CudaBaseSolverPtr;
} // namespace KIRI

#endif