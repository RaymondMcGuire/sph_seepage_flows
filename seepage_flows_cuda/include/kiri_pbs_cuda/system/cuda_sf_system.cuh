/*
 * @Author: Xu.WANG
 * @Date: 2021-02-03 22:52:09
 * @LastEditTime: 2021-07-19 04:25:34
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\system\cuda_sf_system.cuh
 */
#ifndef _CUDA_SF_SYSTEM_CUH_
#define _CUDA_SF_SYSTEM_CUH_

#pragma once

#include <kiri_pbs_cuda/system/cuda_base_system.cuh>

#include <kiri_pbs_cuda/kernel/cuda_sph_kernel.cuh>

#include <kiri_pbs_cuda/particle/cuda_sf_particles.cuh>
#include <kiri_pbs_cuda/particle/cuda_boundary_particles.cuh>
#include <kiri_pbs_cuda/solver/seepageflow/cuda_wcsph_sf_solver.cuh>

#include <kiri_pbs_cuda/searcher/cuda_neighbor_searcher.cuh>

#include <kiri_pbs_cuda/emitter/cuda_emitter.cuh>
namespace KIRI
{
    class CudaSFSystem : public CudaBaseSystem
    {
    public:
        explicit CudaSFSystem(
            CudaSFParticlesPtr &particles,
            CudaBoundaryParticlesPtr &boundaryParticles,
            CudaSphSFSolverPtr &solver,
            CudaGNSearcherPtr &searcher,
            CudaGNBoundarySearcherPtr &boundarySearcher,
            CudaEmitterPtr &emitter,
            bool adaptiveSubTimeStep = false);

        CudaSFSystem(const CudaSFSystem &) = delete;
        CudaSFSystem &operator=(const CudaSFSystem &) = delete;
        virtual ~CudaSFSystem() noexcept {}

        inline size_t NumOfParticles() const { return (*mParticles).Size(); }
        inline size_t MaxNumOfParticles() const { return (*mParticles).MaxSize(); }

        inline CudaSFParticlesPtr GetSFParticles() const { return mParticles; }

    protected:
        virtual void OnUpdateSolver(float timeIntervalInSeconds) override;

    private:
        const size_t mCudaGridSize;
        size_t mEmitterCounter;
        float mEmitterElapsedTime;
        CudaSFParticlesPtr mParticles;
        CudaGNSearcherPtr mSearcher;
        CudaEmitterPtr mEmitter;

        void ComputeBoundaryVolume();
    };

    typedef SharedPtr<CudaSFSystem> CudaSFSystemPtr;
} // namespace KIRI

#endif