/*** 
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-03-22 14:40:51
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-03-22 16:18:26
 * @FilePath: \sph_seepage_flows\seepage_flow_cuda\include\kiri_pbs_cuda\solver\seepageflow\cuda_dfsph_sf_solver.cuh
 * @Description: 
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved. 
 */
#ifndef _CUDA_DFSPH_SF_SOLVER_CUH_
#define _CUDA_DFSPH_SF_SOLVER_CUH_

#pragma once

#include <kiri_pbs_cuda/solver/seepageflow/cuda_sph_sf_solver.cuh>
#include <kiri_pbs_cuda/particle/cuda_dfsph_sf_particles.cuh>

namespace KIRI {
class CudaDFSphSFSolver final : public CudaSphSFSolver {
public:
  explicit CudaDFSphSFSolver(const size_t num)
      : CudaSphSFSolver(num) {}

  virtual ~CudaDFSphSFSolver() noexcept {}

  virtual void UpdateSolver(CudaSFParticlesPtr &particles,
                            CudaBoundaryParticlesPtr &boundaries,
                            const CudaArray<size_t> &cellStart,
                            const CudaArray<size_t> &boundaryCellStart,
                            float renderInterval, CudaSeepageflowParams params,
                            CudaBoundaryParams bparams) override;
protected:
    virtual void ComputeDensity(CudaSFParticlesPtr &particles,
                              CudaBoundaryParticlesPtr &boundaries,
                              const float rho0, const float rho1,
                              const CudaArray<size_t> &cellStart,
                              const CudaArray<size_t> &boundaryCellStart,
                              const float3 lowestPoint,
                              const float kernelRadius, const int3 gridSize)override;

};

typedef SharedPtr<CudaDFSphSFSolver> CudaDFSphSFSolverPtr;
} // namespace KIRI

#endif /* _CUDA_DFSPH_SF_SOLVER_CUH_ */