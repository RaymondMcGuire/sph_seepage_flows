/*
 * @Author: Xu.WANG
 * @Date: 2021-02-03 22:52:09
 * @LastEditTime: 2021-08-21 17:24:38
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath:
 * \sph_seepage_flows\seepage_flows_cuda\include\kiri_pbs_cuda\system\cuda_base_system.cuh
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
  inline size_t GetNumOfSubTimeSteps() const {
    return mSolver->GetNumOfSubTimeSteps();
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