/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-03-25 22:02:18
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-04-08 11:59:15
 * @FilePath:
 * \sph_seepage_flows\seepage_flow_cuda\include\kiri_pbs_cuda\system\cuda_base_system.cuh
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */
#ifndef _CUDA_BASE_SYSTEM_CUH_
#define _CUDA_BASE_SYSTEM_CUH_

#pragma once
#include <kiri_pbs_cuda/particle/cuda_boundary_particles.cuh>
#include <kiri_pbs_cuda/searcher/cuda_neighbor_searcher.cuh>
#include <kiri_pbs_cuda/solver/cuda_base_solver.cuh>

namespace KIRI {
class CudaBaseSystem {
public:
  explicit CudaBaseSystem(CudaBoundaryParticlesPtr &boundaryParticles,
                          CudaBaseSolverPtr &solver,
                          CudaGNBoundarySearcherPtr &boundarySearcher,
                          const size_t maxNumOfParticles,
                          const bool adaptiveSubTimeStep);

  CudaBaseSystem(const CudaBaseSystem &) = delete;
  CudaBaseSystem &operator=(const CudaBaseSystem &) = delete;
  virtual ~CudaBaseSystem() noexcept {}

  float UpdateSystem(float timeIntervalInSeconds);

  inline bool GetAdaptiveSubTimeStep() const { return bAdaptiveSubTimeStep; }
  inline float GetCurrentTimeStep() const {
    return mSolver->GetCurrentTimeSteps();
  }

protected:
  CudaBaseSolverPtr mSolver;
  CudaBoundaryParticlesPtr mBoundaries;
  CudaGNBoundarySearcherPtr mBoundarySearcher;

  virtual void OnUpdateSolver(float timeIntervalInSeconds) = 0;

private:
  bool bAdaptiveSubTimeStep;
};

typedef SharedPtr<CudaBaseSystem> CudaBaseSystemPtr;
} // namespace KIRI

#endif