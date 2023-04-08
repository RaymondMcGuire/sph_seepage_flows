/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-03-25 22:02:18
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-04-08 11:49:55
 * @FilePath:
 * \sph_seepage_flows\seepage_flow_cuda\include\kiri_pbs_cuda\solver\seepageflow\cuda_sph_sf_solver.cuh
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */
#ifndef _CUDA_SPH_SF_SOLVER_CUH_
#define _CUDA_SPH_SF_SOLVER_CUH_

#pragma once

#include <kiri_pbs_cuda/solver/cuda_base_solver.cuh>

#include <kiri_pbs_cuda/data/cuda_boundary_params.h>
#include <kiri_pbs_cuda/data/cuda_seepageflow_params.h>
#include <kiri_pbs_cuda/data/cuda_sph_params.h>

#include <kiri_pbs_cuda/kernel/cuda_sph_kernel.cuh>

#include <kiri_pbs_cuda/particle/cuda_boundary_particles.cuh>
#include <kiri_pbs_cuda/particle/cuda_sf_particles.cuh>

namespace KIRI {
class CudaSphSFSolver : public CudaBaseSolver {
public:
  explicit CudaSphSFSolver(const size_t num) : CudaBaseSolver(num) {}

  virtual ~CudaSphSFSolver() noexcept {}

  virtual void UpdateSolver(CudaSFParticlesPtr &particles,
                            CudaBoundaryParticlesPtr &boundaries,
                            const CudaArray<size_t> &cellStart,
                            const CudaArray<size_t> &boundaryCellStart,
                            float timeIntervalInSeconds,
                            CudaSeepageflowParams params,
                            CudaBoundaryParams bparams);

protected:
  virtual void ExtraForces(CudaSFParticlesPtr &particles, const float3 gravity);

  virtual void Advect(CudaSFParticlesPtr &particles,
                      CudaBoundaryParticlesPtr &boundaries,
                      const CudaArray<size_t> &boundaryCellStart,
                      const float waterRadius, const float dt,
                      const float damping, const float3 lowestPoint,
                      const float3 highestPoint, const float kernelRadius,
                      const int3 gridSize);

  virtual void ComputeDensity(CudaSFParticlesPtr &particles,
                              CudaBoundaryParticlesPtr &boundaries,
                              const float rho0, const float rho1,
                              const CudaArray<size_t> &cellStart,
                              const CudaArray<size_t> &boundaryCellStart,
                              const float3 lowestPoint,
                              const float kernelRadius, const int3 gridSize);

  virtual void ComputePressure(CudaSFParticlesPtr &particles, const float rho0,
                               const float stiff);

  virtual void ComputeAvgFlowVelocity(CudaSFParticlesPtr &particles,
                                      const CudaArray<size_t> &cellStart,
                                      const float3 lowestPoint,
                                      const float kernelRadius,
                                      const int3 gridSize);

  virtual void ComputeSFSandVoidage(CudaSFParticlesPtr &particles,
                                    const CudaArray<size_t> &cellStart,
                                    const float3 lowestPoint,
                                    const float kernelRadius,
                                    const int3 gridSize);

  virtual void ComputeSFWaterAdhesion(
      CudaSFParticlesPtr &particles, const CudaArray<size_t> &cellStart,
      const float a0, const float asat, const float amc, const float amcp,
      const float3 lowestPoint, const float kernelRadius, const int3 gridSize);

  virtual void ComputeMultiSFWaterAdhesion(CudaSFParticlesPtr &particles,
                                           const CudaArray<size_t> &cellStart,
                                           const float3 lowestPoint,
                                           const float kernelRadius,
                                           const int3 gridSize);

  virtual void ComputeSFWetSandColor(CudaSFParticlesPtr &particles,
                                     const float3 dryColor,
                                     const float3 wetColor);

  virtual void ComputeSFSandLinearMomentum(
      CudaSFParticlesPtr &particles, CudaBoundaryParticlesPtr &boundaries,
      const CudaArray<size_t> &cellStart,
      const CudaArray<size_t> &boundaryCellStart, const float boundaryRadius,
      const float maxForceFactor, const float young, const float poisson,
      const float tanFrictionAngle, const float c0, const float csat,
      const float cmc, const float cmcp, const float cd, const float gravity,
      const float rho0, const float3 lowestPoint, const float3 highestPoint,
      const float kernelRadius, const int3 gridSize);

  virtual void ComputeMultiSFSandLinearMomentum(
      CudaSFParticlesPtr &particles, CudaBoundaryParticlesPtr &boundaries,
      const CudaArray<size_t> &cellStart,
      const CudaArray<size_t> &boundaryCellStart, const float boundaryRadius,
      const float maxForceFactor, const float young, const float poisson,
      const float tanFrictionAngle, const float c0, const float csat,
      const float cmc, const float cmcp, const float gravity, const float rho0,
      const float3 lowestPoint, const float3 highestPoint,
      const float kernelRadius, const int3 gridSize);

  virtual void ComputeSFWaterLinearMomentum(
      CudaSFParticlesPtr &particles, CudaBoundaryParticlesPtr &boundaries,
      const CudaArray<size_t> &cellStart,
      const CudaArray<size_t> &boundaryCellStart, const float rho0,
      const float nu, const float bnu, const float3 lowestPoint,
      const float kernelRadius, const int3 gridSize);
};

typedef SharedPtr<CudaSphSFSolver> CudaSphSFSolverPtr;
} // namespace KIRI

#endif /* _CUDA_SPH_SF_SOLVER_CUH_ */