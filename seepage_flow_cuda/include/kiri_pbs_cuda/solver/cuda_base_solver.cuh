/*
 * @Author: Xu.WANG
 * @Date: 2021-02-01 14:31:30
 * @LastEditTime: 2021-03-19 23:53:26
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\cuda_base_solver.cuh
 */

#ifndef _CUDA_BASE_SOLVER_CUH_
#define _CUDA_BASE_SOLVER_CUH_

#pragma once

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>

namespace KIRI {
class CudaBaseSolver {
public:
  explicit CudaBaseSolver(const size_t num)
      : mCudaGridSize(CuCeilDiv(num, KIRI_CUBLOCKSIZE)), mNumOfSubTimeSteps(1) {
  }

  virtual ~CudaBaseSolver() noexcept {}
  inline size_t GetNumOfSubTimeSteps() const { return mNumOfSubTimeSteps; }

protected:
  size_t mCudaGridSize;
  size_t mNumOfSubTimeSteps;
};

typedef SharedPtr<CudaBaseSolver> CudaBaseSolverPtr;
} // namespace KIRI

#endif