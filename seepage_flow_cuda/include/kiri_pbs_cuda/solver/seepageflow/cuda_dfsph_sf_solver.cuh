/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-03-22 14:40:51
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-04-08 11:49:34
 * @FilePath:
 * \sph_seepage_flows\seepage_flow_cuda\include\kiri_pbs_cuda\solver\seepageflow\cuda_dfsph_sf_solver.cuh
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */
#ifndef _CUDA_DFSPH_SF_SOLVER_CUH_
#define _CUDA_DFSPH_SF_SOLVER_CUH_

#pragma once

#include <kiri_pbs_cuda/particle/cuda_dfsf_particles.cuh>
#include <kiri_pbs_cuda/solver/seepageflow/cuda_sph_sf_solver.cuh>

namespace KIRI {
class CudaDFSphSFSolver final : public CudaSphSFSolver {
public:
  explicit CudaDFSphSFSolver(const size_t num, const float dt = 0.001f,
                             const size_t pressureMinIter = 2,
                             const size_t pressureMaxIter = 100,
                             const size_t divergenceMinIter = 1,
                             const size_t divergenceMaxIter = 100,
                             const float pressureMaxError = 1e-3f,
                             const float divergenceMaxError = 1e-3f)
      : CudaSphSFSolver(num), mPressureErrorThreshold(pressureMaxError),
        mDivergenceErrorThreshold(divergenceMaxError),
        mPressureMinIter(pressureMinIter), mPressureMaxIter(pressureMaxIter),
        mDivergenceMinIter(divergenceMinIter),
        mDivergenceMaxIter(divergenceMaxIter) {}

  virtual ~CudaDFSphSFSolver() noexcept {}

  virtual void UpdateSolver(CudaSFParticlesPtr &particles,
                            CudaBoundaryParticlesPtr &boundaries,
                            const CudaArray<size_t> &cellStart,
                            const CudaArray<size_t> &boundaryCellStart,
                            float renderInterval, CudaSeepageflowParams params,
                            CudaBoundaryParams bparams) override;

protected:
  void ComputeDFPressure(CudaDFSFParticlesPtr &particles, const float rho0);

  virtual void Advect(CudaSFParticlesPtr &particles,
                      CudaBoundaryParticlesPtr &boundaries,
                      const CudaArray<size_t> &boundaryCellStart,
                      const float waterRadius, const float dt,
                      const float damping, const float3 lowestPoint,
                      const float3 highestPoint, const float kernelRadius,
                      const int3 gridSize) override;

  virtual void ComputeDensity(
      CudaSFParticlesPtr &particles, CudaBoundaryParticlesPtr &boundaries,
      const float rho0, const float rho1, const CudaArray<size_t> &cellStart,
      const CudaArray<size_t> &boundaryCellStart, const float3 lowestPoint,
      const float kernelRadius, const int3 gridSize) override;

  void AdvectDFSPHVelocity(CudaDFSFParticlesPtr &fluids);

  void ComputeTimeStepsByCFL(CudaDFSFParticlesPtr &fluids,
                             const float particleRadius, const float dt,
                             const float timeIntervalInSeconds);

  void ComputeDFSPHAlpha(CudaDFSFParticlesPtr &fluids,
                         CudaBoundaryParticlesPtr &boundaries, const float rho0,
                         const CudaArray<size_t> &cellStart,
                         const CudaArray<size_t> &boundaryCellStart,
                         const float3 lowestPoint, const float kernelRadius,
                         const int3 gridSize);

  size_t ApplyDivergenceSolver(CudaDFSFParticlesPtr &fluids,
                               CudaBoundaryParticlesPtr &boundaries,
                               const float rho0,
                               const CudaArray<size_t> &cellStart,
                               const CudaArray<size_t> &boundaryCellStart,
                               const float3 lowestPoint,
                               const float kernelRadius, const int3 gridSize);

  size_t ApplyPressureSolver(CudaDFSFParticlesPtr &fluids,
                             CudaBoundaryParticlesPtr &boundaries,
                             const float rho0,
                             const CudaArray<size_t> &cellStart,
                             const CudaArray<size_t> &boundaryCellStart,
                             const float3 lowestPoint, const float kernelRadius,
                             const int3 gridSize);

  virtual void ComputeSFSandLinearMomentum(
      CudaSFParticlesPtr &particles, CudaBoundaryParticlesPtr &boundaries,
      const CudaArray<size_t> &cellStart,
      const CudaArray<size_t> &boundaryCellStart, const float boundaryRadius,
      const float maxForceFactor, const float young, const float poisson,
      const float tanFrictionAngle, const float c0, const float csat,
      const float cmc, const float cmcp, const float cd, const float gravity,
      const float rho0, const float3 lowestPoint, const float3 highestPoint,
      const float kernelRadius, const int3 gridSize) override;

  virtual void ComputeMultiSFSandLinearMomentum(
      CudaSFParticlesPtr &particles, CudaBoundaryParticlesPtr &boundaries,
      const CudaArray<size_t> &cellStart,
      const CudaArray<size_t> &boundaryCellStart, const float boundaryRadius,
      const float maxForceFactor, const float young, const float poisson,
      const float tanFrictionAngle, const float c0, const float csat,
      const float cmc, const float cmcp, const float gravity, const float rho0,
      const float3 lowestPoint, const float3 highestPoint,
      const float kernelRadius, const int3 gridSize) override;

  virtual void ComputeSFWaterLinearMomentum(
      CudaSFParticlesPtr &particles, CudaBoundaryParticlesPtr &boundaries,
      const CudaArray<size_t> &cellStart,
      const CudaArray<size_t> &boundaryCellStart, const float rho0,
      const float nu, const float bnu, const float3 lowestPoint,
      const float kernelRadius, const int3 gridSize) override;

private:
  float mDivergenceErrorThreshold, mPressureErrorThreshold;
  size_t mPressureMinIter, mPressureMaxIter, mDivergenceMinIter,
      mDivergenceMaxIter;

  const float CFL_FACTOR = 0.5f;
  const float CFL_MIN_TIMESTEP_SIZE = 0.0001f;
  const float CFL_MAX_TIMESTEP_SIZE = 0.005f;
};

typedef SharedPtr<CudaDFSphSFSolver> CudaDFSphSFSolverPtr;
} // namespace KIRI

#endif /* _CUDA_DFSPH_SF_SOLVER_CUH_ */