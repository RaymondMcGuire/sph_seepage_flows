/*
 * @Author: Xu.WANG
 * @Date: 2021-02-03 22:59:48
 * @LastEditTime: 2021-08-21 17:23:58
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \sph_seepage_flows\seepage_flows_cuda\src\kiri_pbs_cuda\system\cuda_base_system.cu
 */

#include <kiri_pbs_cuda/system/cuda_base_system.cuh>

namespace KIRI
{
    CudaBaseSystem::CudaBaseSystem(
        CudaBoundaryParticlesPtr &boundaryParticles,
        CudaBaseSolverPtr &solver,
        CudaGNBoundarySearcherPtr &boundarySearcher,
        const size_t maxNumOfParticles,
        const bool adaptiveSubTimeStep)
        : mBoundaries(std::move(boundaryParticles)),
          mSolver(std::move(solver)),
          mBoundarySearcher(std::move(boundarySearcher)),
          bAdaptiveSubTimeStep(adaptiveSubTimeStep)
    {
        // build boundary searcher
        mBoundarySearcher->BuildGNSearcher(mBoundaries);
    }

    float CudaBaseSystem::UpdateSystem(float renderInterval)
    {
        cudaEvent_t start, stop;
        KIRI_CUCALL(cudaEventCreate(&start));
        KIRI_CUCALL(cudaEventCreate(&stop));
        KIRI_CUCALL(cudaEventRecord(start, 0));

        try
        {
            OnUpdateSolver(renderInterval);
        }
        catch (const char *s)
        {
            std::cout << s << "\n";
        }
        catch (...)
        {
            std::cout << "Unknown Exception at " << __FILE__ << ": line " << __LINE__ << "\n";
        }

        float milliseconds;
        KIRI_CUCALL(cudaEventRecord(stop, 0));
        KIRI_CUCALL(cudaEventSynchronize(stop));
        KIRI_CUCALL(cudaEventElapsedTime(&milliseconds, start, stop));
        KIRI_CUCALL(cudaEventDestroy(start));
        KIRI_CUCALL(cudaEventDestroy(stop));
        return milliseconds;
    }
} // namespace KIRI
