/*
 * @Author: Xu.WANG
 * @Date: 2021-02-03 22:59:48
 * @LastEditTime: 2021-08-16 00:46:39
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \KiriPBSCuda\src\kiri_pbs_cuda\system\cuda_base_system.cu
 */

#include <kiri_pbs_cuda/system/cuda_base_system.cuh>

#include <glad/glad.h>
#include <cuda_gl_interop.h>
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
        KIRI_CUCALL(cudaMalloc((void **)&mCudaPositionVBO, sizeof(float4) * maxNumOfParticles));
        KIRI_CUCALL(cudaMalloc((void **)&mCudaColorVBO, sizeof(float4) * maxNumOfParticles));

        // init position vbo
        size_t buf_size = maxNumOfParticles * sizeof(float4);
        glGenBuffers(1, &mPositionsVBO);
        glBindBuffer(GL_ARRAY_BUFFER, mPositionsVBO);
        glBufferData(GL_ARRAY_BUFFER, buf_size, nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // init color vbo
        size_t color_buf_size = maxNumOfParticles * sizeof(float4);
        glGenBuffers(1, &mColorsVBO);
        glBindBuffer(GL_ARRAY_BUFFER, mColorsVBO);
        glBufferData(GL_ARRAY_BUFFER, color_buf_size, nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // build boundary searcher
        mBoundarySearcher->BuildGNSearcher(mBoundaries);
    }

    void CudaBaseSystem::UpdateSystemForVBO(float renderInterval)
    {
        KIRI_CUCALL(cudaGraphicsGLRegisterBuffer(&mCudaGraphPosVBORes, mPositionsVBO,
                                                 cudaGraphicsMapFlagsNone));
        KIRI_CUCALL(cudaGraphicsGLRegisterBuffer(&mCudaGraphColorVBORes, mColorsVBO,
                                                 cudaGraphicsMapFlagsNone));

        size_t num_bytes = 0;
        KIRI_CUCALL(cudaGraphicsMapResources(1, &mCudaGraphPosVBORes, 0));
        KIRI_CUCALL(cudaGraphicsResourceGetMappedPointer(
            (void **)&mCudaPositionVBO, &num_bytes, mCudaGraphPosVBORes));

        size_t color_num_bytes = 0;
        KIRI_CUCALL(cudaGraphicsMapResources(1, &mCudaGraphColorVBORes, 0));
        KIRI_CUCALL(cudaGraphicsResourceGetMappedPointer(
            (void **)&mCudaColorVBO, &color_num_bytes, mCudaGraphColorVBORes));

        if (bAdaptiveSubTimeStep)
        {
            float remaining_time = renderInterval;
            while (remaining_time > KIRI_EPSILON)
            {
                UpdateSystem(remaining_time);
                remaining_time -= remaining_time / static_cast<float>(mSolver->GetNumOfSubTimeSteps());
            }
        }
        else
        {
            size_t sub_timesteps_num = mSolver->GetNumOfSubTimeSteps();
            for (size_t i = 0; i < sub_timesteps_num; i++)
                UpdateSystem(renderInterval);
        }

        OnTransferGPUData2VBO(mCudaPositionVBO, mCudaColorVBO);

        KIRI_CUCALL(cudaGraphicsUnmapResources(1, &mCudaGraphPosVBORes, 0));
        KIRI_CUCALL(cudaGraphicsUnregisterResource(mCudaGraphPosVBORes));

        KIRI_CUCALL(cudaGraphicsUnmapResources(1, &mCudaGraphColorVBORes, 0));
        KIRI_CUCALL(cudaGraphicsUnregisterResource(mCudaGraphColorVBORes));
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
