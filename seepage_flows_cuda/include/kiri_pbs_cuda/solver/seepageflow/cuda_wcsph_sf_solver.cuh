/*
 * @Author: Xu.WANG
 * @Date: 2021-02-01 14:31:30
 * @LastEditTime: 2021-04-08 00:03:29
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\solver\seepageflow\cuda_wcsph_sf_solver.cuh
 */

#ifndef _CUDA_WCSPH_SF_SOLVER_CUH_
#define _CUDA_WCSPH_SF_SOLVER_CUH_

#pragma once

#include <kiri_pbs_cuda/solver/seepageflow/cuda_sph_sf_solver.cuh>

namespace KIRI
{
    class CudaWCSphSFSolver final : public CudaSphSFSolver
    {
    public:
        explicit CudaWCSphSFSolver(
            const size_t num,
            const float negativeScale = 0.f,
            const float timeStepLimitScale = 3.f,
            const float speedOfSound = 100.f)
            : CudaSphSFSolver(num),
              mNegativeScale(negativeScale),
              mTimeStepLimitScale(timeStepLimitScale),
              mSpeedOfSound(speedOfSound)
        {
        }

        virtual ~CudaWCSphSFSolver() noexcept {}

        virtual void UpdateSolver(
            CudaSFParticlesPtr &particles,
            CudaBoundaryParticlesPtr &boundaries,
            const CudaArray<size_t> &cellStart,
            const CudaArray<size_t> &boundaryCellStart,
            float renderInterval,
            CudaSeepageflowParams params,
            CudaBoundaryParams bparams) override;

        float GetSpeedOfSound() const;
        float GetTimeStepLimitScale() const;

        void SetTimeStepLimitScale(float newScale);
        void SetSpeedOfSound(float newSpeedOfSound);

    protected:
        virtual void ComputePressure(
            CudaSFParticlesPtr &particles,
            const float rho0,
            const float stiff) override;

    private:
        float mNegativeScale, mTimeStepLimitScale, mSpeedOfSound;
        const float mTimeStepLimitBySpeedFactor = 0.4f;
        const float mTimeStepLimitByForceFactor = 0.25f;

        void ComputeSubTimeStepsByCFL(
            CudaSFParticlesPtr &particles,
            const float sphMass,
            const float dt,
            const float kernelRadius,
            float renderInterval);
    };

    typedef SharedPtr<CudaWCSphSFSolver> CudaWCSphSFSolverPtr;
} // namespace KIRI

#endif /* _CUDA_WCSPH_SF_SOLVER_CUH_ */