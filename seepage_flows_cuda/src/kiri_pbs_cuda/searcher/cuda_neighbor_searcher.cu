/*
 * @Author: Xu.WANG
 * @Date: 2021-02-05 12:33:37
 * @LastEditTime: 2021-08-27 23:44:32
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \sph_seepage_flows\seepage_flows_cuda\src\kiri_pbs_cuda\searcher\cuda_neighbor_searcher.cu
 */

#include <kiri_pbs_cuda/thrust_helper/helper_thrust.cuh>
#include <kiri_pbs_cuda/searcher/cuda_neighbor_searcher.cuh>
#include <kiri_pbs_cuda/searcher/cuda_neighbor_searcher_gpu.cuh>

#include <kiri_pbs_cuda/particle/cuda_sf_particles.cuh>

namespace KIRI
{

    CudaGNBaseSearcher::CudaGNBaseSearcher(
        const float3 lowestPoint,
        const float3 highestPoint,
        const size_t maxNumOfParticles,
        const float cellSize)
        : mLowestPoint(lowestPoint),
          mHighestPoint(highestPoint),
          mCellSize(cellSize),
          mGridSize(make_int3((highestPoint - lowestPoint) / cellSize)),
          mNumOfGridCells(mGridSize.x * mGridSize.y * mGridSize.z + 1),
          mCellStart(mNumOfGridCells),
          mMaxNumOfParticles(maxNumOfParticles),
          mGridIdxArray(max(mNumOfGridCells, maxNumOfParticles)),
          mCudaGridSize(CuCeilDiv(maxNumOfParticles, KIRI_CUBLOCKSIZE))
    {
    }

    void CudaGNBaseSearcher::BuildGNSearcher(const CudaParticlesPtr &particles)
    {
        thrust::transform(thrust::device,
                          particles->GetPosPtr(), particles->GetPosPtr() + particles->Size(),
                          mGridIdxArray.Data(),
                          ThrustHelper::Pos2GridHash<float3>(mLowestPoint, mCellSize, mGridSize));

        this->SortData(particles);

        thrust::fill(thrust::device, mCellStart.Data(), mCellStart.Data() + mNumOfGridCells, 0);
        CountingInCell_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(mCellStart.Data(), mGridIdxArray.Data(), particles->Size());
        thrust::exclusive_scan(thrust::device, mCellStart.Data(), mCellStart.Data() + mNumOfGridCells, mCellStart.Data());

        cudaDeviceSynchronize();
        KIRI_CUKERNAL();
    }

    CudaGNSearcher::CudaGNSearcher(
        const float3 lp,
        const float3 hp,
        const size_t num,
        const float cellSize,
        const SearcherParticleType type)
        : CudaGNBaseSearcher(lp, hp, num, cellSize),
          mSearcherParticleType(type)
    {
    }

    void CudaGNSearcher::SortData(const CudaParticlesPtr &particles)
    {

       if (mSearcherParticleType == SearcherParticleType::SEEPAGE)
        {
            auto seepage_flow = std::dynamic_pointer_cast<CudaSFParticles>(particles);
            thrust::sort_by_key(thrust::device,
                                mGridIdxArray.Data(),
                                mGridIdxArray.Data() + particles->Size(),
                                thrust::make_zip_iterator(
                                    thrust::make_tuple(
                                        seepage_flow->GetLabelPtr(),
                                        seepage_flow->GetPosPtr(),
                                        seepage_flow->GetVelPtr(),
                                        seepage_flow->GetColPtr(),
                                        seepage_flow->GetRadiusPtr(),
                                        seepage_flow->GetMassPtr(),
                                        seepage_flow->GetMaxSaturationPtr())));
        }

        cudaDeviceSynchronize();
        KIRI_CUKERNAL();
    }

    CudaGNBoundarySearcher::CudaGNBoundarySearcher(
        const float3 lp,
        const float3 hp,
        const size_t num,
        const float cellSize)
        : CudaGNBaseSearcher(lp, hp, num, cellSize) {}

    void CudaGNBoundarySearcher::SortData(const CudaParticlesPtr &particles)
    {
        auto boundaries = std::dynamic_pointer_cast<CudaBoundaryParticles>(particles);
        thrust::sort_by_key(thrust::device,
                            mGridIdxArray.Data(),
                            mGridIdxArray.Data() + particles->Size(),
                            thrust::make_zip_iterator(
                                thrust::make_tuple(
                                    boundaries->GetPosPtr(),
                                    boundaries->GetLabelPtr())));

        cudaDeviceSynchronize();
        KIRI_CUKERNAL();
    }

} // namespace KIRI
