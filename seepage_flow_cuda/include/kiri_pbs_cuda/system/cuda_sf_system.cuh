/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-03-25 22:02:18
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-03-25 22:59:23
 * @FilePath:
 * \sph_seepage_flows\seepage_flow_cuda\include\kiri_pbs_cuda\system\cuda_sf_system.cuh
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */
#ifndef _CUDA_SF_SYSTEM_CUH_
#define _CUDA_SF_SYSTEM_CUH_

#pragma once

#include <kiri_pbs_cuda/system/cuda_base_system.cuh>

#include <kiri_pbs_cuda/kernel/cuda_sph_kernel.cuh>

#include <kiri_pbs_cuda/particle/cuda_boundary_particles.cuh>
#include <kiri_pbs_cuda/particle/cuda_sf_particles.cuh>
#include <kiri_pbs_cuda/solver/seepageflow/cuda_dfsph_sf_solver.cuh>
#include <kiri_pbs_cuda/solver/seepageflow/cuda_wcsph_sf_solver.cuh>

#include <kiri_pbs_cuda/searcher/cuda_neighbor_searcher.cuh>

#include <kiri_pbs_cuda/emitter/cuda_emitter.cuh>
namespace KIRI {
class CudaSFSystem : public CudaBaseSystem {
public:
  explicit CudaSFSystem(CudaSFParticlesPtr &particles,
                        CudaBoundaryParticlesPtr &boundaryParticles,
                        CudaSphSFSolverPtr &solver, CudaGNSearcherPtr &searcher,
                        CudaGNBoundarySearcherPtr &boundarySearcher,
                        CudaEmitterPtr &emitter,
                        bool adaptiveSubTimeStep = false);

  CudaSFSystem(const CudaSFSystem &) = delete;
  CudaSFSystem &operator=(const CudaSFSystem &) = delete;
  virtual ~CudaSFSystem() noexcept {}

  inline size_t NumOfParticles() const { return (*mParticles).Size(); }
  inline size_t MaxNumOfParticles() const { return (*mParticles).MaxSize(); }

  inline CudaSFParticlesPtr GetSFParticles() const { return mParticles; }
  inline void UpdateEmitterVelocity(float3 velocity) {
    mEmitter->UpdateEmitterVelocity(velocity);
  }

protected:
  virtual void OnUpdateSolver(float timeIntervalInSeconds) override;

private:
  const size_t mCudaGridSize;

  CudaSFParticlesPtr mParticles;
  CudaGNSearcherPtr mSearcher;

  CudaEmitterPtr mEmitter;
  float mNextEmitTime;
  float mEmitterElapsedTime;

  void ComputeBoundaryVolume();
};

typedef SharedPtr<CudaSFSystem> CudaSFSystemPtr;
} // namespace KIRI

#endif