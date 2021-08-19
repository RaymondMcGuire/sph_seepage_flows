/*
 * @Author: Xu.WANG
 * @Date: 2021-02-01 14:31:30
 * @LastEditTime: 2021-08-16 00:08:49
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\solver\seepageflow\cuda_iisph_sf_solver.cuh
 */

#ifndef _CUDA_IISPH_SF_SOLVER_CUH_
#define _CUDA_IISPH_SF_SOLVER_CUH_

#pragma once

#include <kiri_pbs_cuda/solver/seepageflow/cuda_sph_sf_solver.cuh>

namespace KIRI
{
    class CudaIISphSFSolver final : public CudaSphSFSolver
    {
    public:
        explicit CudaIISphSFSolver(
            const size_t num,
            const size_t minIter = 2,
            const size_t maxIter = 100,
            const float maxError = 0.01f)
            : CudaSphSFSolver(num),
              mMaxError(maxError),
              mMinIter(minIter),
              mMaxIter(maxIter)
        {
        }

        virtual ~CudaIISphSFSolver() noexcept {}

        virtual void UpdateSolver(
            CudaSFParticlesPtr &particles,
            CudaBoundaryParticlesPtr &boundaries,
            const CudaArray<size_t> &cellStart,
            const CudaArray<size_t> &boundaryCellStart,
            float renderInterval,
            CudaSeepageflowParams params,
            CudaBoundaryParams bparams) override;

    protected:
        virtual void Advect(
            CudaSFParticlesPtr &particles,
            CudaBoundaryParticlesPtr &boundaries,
            const CudaArray<size_t> &boundaryCellStart,
            const float waterRadius,
            const float dt,
            const float damping,
            const float3 lowestPoint,
            const float3 highestPoint,
            const float kernelRadius,
            const int3 gridSize) override;

        void PredictVelAdvect(
            CudaSFParticlesPtr &particles,
            const float dt);

        void ComputeDiiTerm(
            CudaSFParticlesPtr &particles,
            CudaBoundaryParticlesPtr &boundaries,
            const CudaArray<size_t> &cellStart,
            const CudaArray<size_t> &boundaryCellStart,
            const float rho0,
            const float3 lowestPoint,
            const float kernelRadius,
            const int3 gridSize);

        void ComputeAiiTerm(
            CudaSFParticlesPtr &particles,
            CudaBoundaryParticlesPtr &boundaries,
            const CudaArray<size_t> &cellStart,
            const CudaArray<size_t> &boundaryCellStart,
            const float rho0,
            const float dt,
            const float3 lowestPoint,
            const float kernelRadius,
            const int3 gridSize);

        size_t PressureSolver(
            CudaSFParticlesPtr &particles,
            CudaBoundaryParticlesPtr &boundaries,
            const float rho0,
            const float dt,
            const CudaArray<size_t> &cellStart,
            const CudaArray<size_t> &boundaryCellStart,
            const float3 lowestPoint,
            const float kernelRadius,
            const int3 gridSize);

        void ComputePressureAcceleration(
            CudaSFParticlesPtr &particles,
            CudaBoundaryParticlesPtr &boundaries,
            const CudaArray<size_t> &cellStart,
            const CudaArray<size_t> &boundaryCellStart,
            const float rho0,
            const float3 lowestPoint,
            const float kernelRadius,
            const int3 gridSize);

    private:
        float mMaxError;
        size_t mMinIter, mMaxIter;

        void ComputeIISFSandLMNB(
            CudaSFParticlesPtr &particles,
            CudaBoundaryParticlesPtr &boundaries,
            const CudaArray<size_t> &cellStart,
            const CudaArray<size_t> &boundaryCellStart,
            const float sandRadius,
            const float waterRadius,
            const float young,
            const float poisson,
            const float tanFrictionAngle,
            const float c0,
            const float csat,
            const float cmc,
            const float cmcp,
            const float cd,
            const float gravity,
            const float rho0,
            const float3 lowestPoint,
            const float3 highestPoint,
            const float kernelRadius,
            const int3 gridSize);

        void ComputeIISFSandLMWB(
            CudaSFParticlesPtr &particles,
            const CudaArray<size_t> &cellStart,
            const float3 lowestPoint,
            const float kernelRadius,
            const int3 gridSize);

        void ComputeIISFWaterLMNB(
            CudaSFParticlesPtr &particles,
            CudaBoundaryParticlesPtr &boundaries,
            const CudaArray<size_t> &cellStart,
            const CudaArray<size_t> &boundaryCellStart,
            const float rho0,
            const float nu,
            const float bnu,
            const float3 lowestPoint,
            const float kernelRadius,
            const int3 gridSize);

        void ComputeIISFWaterLMWB(
            CudaSFParticlesPtr &particles,
            const CudaArray<size_t> &cellStart,
            const float3 lowestPoint,
            const float kernelRadius,
            const int3 gridSize);
    };

    typedef SharedPtr<CudaIISphSFSolver> CudaIISphSFSolverPtr;
} // namespace KIRI

#endif /* _CUDA_IISPH_SF_SOLVER_CUH_ */