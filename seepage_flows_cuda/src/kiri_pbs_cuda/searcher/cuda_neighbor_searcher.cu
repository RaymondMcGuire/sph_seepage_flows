/*
 * @Author: Xu.WANG
 * @Date: 2021-02-05 12:33:37
 * @LastEditTime: 2021-08-18 09:22:59
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\searcher\cuda_neighbor_searcher.cu
 */

#include <kiri_pbs_cuda/thrust_helper/helper_thrust.cuh>
#include <kiri_pbs_cuda/searcher/cuda_neighbor_searcher.cuh>
#include <kiri_pbs_cuda/searcher/cuda_neighbor_searcher_gpu.cuh>

#include <kiri_pbs_cuda/particle/cuda_sph_particles.cuh>
#include <kiri_pbs_cuda/particle/cuda_iisph_particles.cuh>
#include <kiri_pbs_cuda/particle/cuda_sph_bn_particles.cuh>
#include <kiri_pbs_cuda/particle/cuda_dem_particles.cuh>
#include <kiri_pbs_cuda/particle/cuda_mr_dem_particles.cuh>
#include <kiri_pbs_cuda/particle/cuda_sf_particles.cuh>
#include <kiri_pbs_cuda/particle/cuda_iisf_particles.cuh>

#include <kiri_pbs_cuda/particle/cuda_multisph_ren14_particles.cuh>
#include <kiri_pbs_cuda/particle/cuda_multisph_yan16_particles.cuh>

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

        if (mSearcherParticleType == SearcherParticleType::SPH)
        {
            auto fluids = std::dynamic_pointer_cast<CudaSphParticles>(particles);
            thrust::sort_by_key(thrust::device,
                                mGridIdxArray.Data(),
                                mGridIdxArray.Data() + particles->Size(),
                                thrust::make_zip_iterator(
                                    thrust::make_tuple(
                                        fluids->GetPosPtr(),
                                        fluids->GetVelPtr(),
                                        fluids->GetColPtr())));
        }
        else if (mSearcherParticleType == SearcherParticleType::IISPH)
        {
            auto fluids = std::dynamic_pointer_cast<CudaIISphParticles>(particles);
            thrust::sort_by_key(thrust::device,
                                mGridIdxArray.Data(),
                                mGridIdxArray.Data() + particles->Size(),
                                thrust::make_zip_iterator(
                                    thrust::make_tuple(
                                        fluids->GetPosPtr(),
                                        fluids->GetVelPtr(),
                                        fluids->GetColPtr(),
                                        fluids->GetLastPressurePtr())));
        }
        else if (mSearcherParticleType == SearcherParticleType::DEM)
        {
            auto sands = std::dynamic_pointer_cast<CudaDemParticles>(particles);
            thrust::sort_by_key(thrust::device,
                                mGridIdxArray.Data(),
                                mGridIdxArray.Data() + particles->Size(),
                                thrust::make_zip_iterator(
                                    thrust::make_tuple(
                                        sands->GetPosPtr(),
                                        sands->GetVelPtr(),
                                        sands->GetColPtr())));
        }
        else if (mSearcherParticleType == SearcherParticleType::MRDEM)
        {
            auto sands = std::dynamic_pointer_cast<CudaMRDemParticles>(particles);
            thrust::sort_by_key(thrust::device,
                                mGridIdxArray.Data(),
                                mGridIdxArray.Data() + particles->Size(),
                                thrust::make_zip_iterator(
                                    thrust::make_tuple(
                                        sands->GetPosPtr(),
                                        sands->GetVelPtr(),
                                        sands->GetColPtr(),
                                        sands->GetMassPtr(),
                                        sands->GetRadiusPtr())));
        }
        else if (mSearcherParticleType == SearcherParticleType::BNSPH)
        {
            auto fluids = std::dynamic_pointer_cast<CudaSphBNParticles>(particles);
            thrust::sort_by_key(thrust::device,
                                mGridIdxArray.Data(),
                                mGridIdxArray.Data() + particles->Size(),
                                thrust::make_zip_iterator(
                                    thrust::make_tuple(
                                        fluids->GetPosPtr(),
                                        fluids->GetVelPtr(),
                                        fluids->GetColPtr(),
                                        fluids->GetMassPtr(),
                                        fluids->GetRadiusPtr())));
        }
        else if (mSearcherParticleType == SearcherParticleType::MULTISPH_REN14)
        {
            auto ren14 = std::dynamic_pointer_cast<CudaMultiSphRen14Particles>(particles);
            thrust::sort_by_key(thrust::device,
                                mGridIdxArray.Data(),
                                mGridIdxArray.Data() + particles->Size(),
                                thrust::make_zip_iterator(
                                    thrust::make_tuple(
                                        ren14->GetPhaseLabelPtr(),
                                        ren14->GetPosPtr(),
                                        ren14->GetVelPtr(),
                                        ren14->GetAccPtr(),
                                        ren14->GetColPtr(),
                                        ren14->GetMixMassPtr(),
                                        ren14->GetRen14PhasePtr())));
        }
        else if (mSearcherParticleType == SearcherParticleType::MULTISPH_YAN16)
        {
            auto yan16 = std::dynamic_pointer_cast<CudaMultiSphYan16Particles>(particles);
            thrust::sort_by_key(thrust::device,
                                mGridIdxArray.Data(),
                                mGridIdxArray.Data() + particles->Size(),
                                thrust::make_zip_iterator(
                                    thrust::make_tuple(
                                        yan16->GetPhaseLabelPtr(),
                                        yan16->GetPhaseTypePtr(),
                                        yan16->GetPosPtr(),
                                        yan16->GetVelPtr(),
                                        yan16->GetAccPtr(),
                                        yan16->GetColPtr(),
                                        yan16->GetMixMassPtr(),
                                        yan16->GetYan16PhasePtr())));
        }
        else if (mSearcherParticleType == SearcherParticleType::SEEPAGE)
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
        else if (mSearcherParticleType == SearcherParticleType::IISEEPAGE)
        {
            auto seepage_flow = std::dynamic_pointer_cast<CudaIISFParticles>(particles);
            thrust::sort_by_key(thrust::device,
                                mGridIdxArray.Data(),
                                mGridIdxArray.Data() + particles->Size(),
                                thrust::make_zip_iterator(
                                    thrust::make_tuple(
                                        seepage_flow->GetLastPressurePtr(),
                                        seepage_flow->GetLabelPtr(),
                                        seepage_flow->GetPosPtr(),
                                        seepage_flow->GetVelPtr(),
                                        seepage_flow->GetColPtr(),
                                        seepage_flow->GetRadiusPtr(),
                                        seepage_flow->GetMassPtr(),
                                        seepage_flow->GetMaxSaturationPtr())));
        }
        else if (mSearcherParticleType == SearcherParticleType::SEEPAGE_MULTI)
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
                                        seepage_flow->GetMaxSaturationPtr(),
                                        seepage_flow->GetCdA0AsatPtr(),
                                        seepage_flow->GetAmcAmcpPtr())));
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
