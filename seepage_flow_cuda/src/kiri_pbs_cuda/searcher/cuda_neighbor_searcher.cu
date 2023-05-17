/*** 
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-05-14 20:01:11
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-05-17 15:11:51
 * @FilePath: \sph_seepage_flows\seepage_flow_cuda\src\kiri_pbs_cuda\searcher\cuda_neighbor_searcher.cu
 * @Description: 
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved. 
 */

#include <kiri_pbs_cuda/searcher/cuda_neighbor_searcher.cuh>
#include <kiri_pbs_cuda/searcher/cuda_neighbor_searcher_gpu.cuh>
#include <kiri_pbs_cuda/thrust_helper/helper_thrust.cuh>

#include <kiri_pbs_cuda/particle/cuda_dfsf_particles.cuh>

namespace KIRI {

CudaGNBaseSearcher::CudaGNBaseSearcher(const float3 lowestPoint,
                                       const float3 highestPoint,
                                       const size_t maxNumOfParticles,
                                       const float cellSize)
    : mLowestPoint(lowestPoint), mHighestPoint(highestPoint),
      mCellSize(cellSize),
      mGridSize(make_int3((highestPoint - lowestPoint) / cellSize)),
      mNumOfGridCells(mGridSize.x * mGridSize.y * mGridSize.z + 1),
      mCellStart(mNumOfGridCells), mMaxNumOfParticles(maxNumOfParticles),
      mGridIdxArray(max(mNumOfGridCells, maxNumOfParticles)),
      mCudaGridSize(CuCeilDiv(maxNumOfParticles, KIRI_CUBLOCKSIZE)) {}

void CudaGNBaseSearcher::BuildGNSearcher(const CudaParticlesPtr &particles) {
  thrust::transform(
      thrust::device, particles->GetPosPtr(),
      particles->GetPosPtr() + particles->Size(),
      particles->GetParticle2CellPtr(),
      ThrustHelper::Pos2GridHash<float3>(mLowestPoint, mCellSize, mGridSize));

  this->SortData(particles);

  thrust::fill(thrust::device, mCellStart.Data(),
               mCellStart.Data() + mNumOfGridCells, 0);
  CountingInCell_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
      mCellStart.Data(), particles->GetParticle2CellPtr(), particles->Size());
  thrust::exclusive_scan(thrust::device, mCellStart.Data(),
                         mCellStart.Data() + mNumOfGridCells,
                         mCellStart.Data());

  cudaDeviceSynchronize();
  KIRI_CUKERNAL();
}

CudaGNSearcher::CudaGNSearcher(const float3 lp, const float3 hp,
                               const size_t num, const float cellSize,
                               const SearcherParticleType type)
    : CudaGNBaseSearcher(lp, hp, num, cellSize), mSearcherParticleType(type) {}

void CudaGNSearcher::SortData(const CudaParticlesPtr &particles) {

  // auto seepage_flow =
  // std::dynamic_pointer_cast<CudaDFSFParticles>(particles);

  if (mSearcherParticleType == SearcherParticleType::SEEPAGE) {
    auto seepage_flow = std::dynamic_pointer_cast<CudaSFParticles>(particles);
    KIRI_CUCALL(cudaMemcpy(
        mGridIdxArray.Data(), seepage_flow->GetParticle2CellPtr(),
        sizeof(size_t) * particles->Size(), cudaMemcpyDeviceToDevice));
    thrust::sort_by_key(
        thrust::device, mGridIdxArray.Data(),
        mGridIdxArray.Data() + particles->Size(),

        thrust::make_zip_iterator(thrust::make_tuple(
            seepage_flow->GetIdPtr(),
            seepage_flow->GetLabelPtr(), seepage_flow->GetPosPtr(),
            seepage_flow->GetVelPtr(), seepage_flow->GetColPtr(),
            seepage_flow->GetRadiusPtr(), seepage_flow->GetMassPtr(),
            seepage_flow->GetAngularVelPtr(), seepage_flow->GetInertiaPtr(),
            seepage_flow->GetMaxSaturationPtr())));
  } else if (mSearcherParticleType == SearcherParticleType::SEEPAGE_MULTI) {
    auto seepage_flow = std::dynamic_pointer_cast<CudaSFParticles>(particles);
    KIRI_CUCALL(cudaMemcpy(
        mGridIdxArray.Data(), seepage_flow->GetParticle2CellPtr(),
        sizeof(size_t) * particles->Size(), cudaMemcpyDeviceToDevice));
    thrust::sort_by_key(
        thrust::device, mGridIdxArray.Data(),
        mGridIdxArray.Data() + particles->Size(),

        thrust::make_zip_iterator(thrust::make_tuple(
            seepage_flow->GetIdPtr(),
            seepage_flow->GetLabelPtr(), seepage_flow->GetPosPtr(),
            seepage_flow->GetVelPtr(), seepage_flow->GetColPtr(),
            seepage_flow->GetRadiusPtr(), seepage_flow->GetMassPtr())));

    KIRI_CUCALL(cudaMemcpy(
        mGridIdxArray.Data(), seepage_flow->GetParticle2CellPtr(),
        sizeof(size_t) * particles->Size(), cudaMemcpyDeviceToDevice));
    thrust::sort_by_key(
        thrust::device, mGridIdxArray.Data(),
        mGridIdxArray.Data() + particles->Size(),
        thrust::make_zip_iterator(thrust::make_tuple(
            seepage_flow->GetAngularVelPtr(), seepage_flow->GetInertiaPtr(),
            seepage_flow->GetMaxSaturationPtr(), seepage_flow->GetCdA0AsatPtr(),
            seepage_flow->GetAmcAmcpPtr())));
  } else if (mSearcherParticleType == SearcherParticleType::DFSF_MULTI) {
    auto seepage_flow = std::dynamic_pointer_cast<CudaDFSFParticles>(particles);
    KIRI_CUCALL(cudaMemcpy(
        mGridIdxArray.Data(), seepage_flow->GetParticle2CellPtr(),
        sizeof(size_t) * particles->Size(), cudaMemcpyDeviceToDevice));
    thrust::sort_by_key(
        thrust::device, mGridIdxArray.Data(),
        mGridIdxArray.Data() + particles->Size(),

        thrust::make_zip_iterator(thrust::make_tuple(
            seepage_flow->GetIdPtr(),
            seepage_flow->GetLabelPtr(), seepage_flow->GetPosPtr(),
            seepage_flow->GetVelPtr(), seepage_flow->GetColPtr(),
            seepage_flow->GetRadiusPtr(), seepage_flow->GetMassPtr(),
            seepage_flow->GetWarmStiffPtr())));

    KIRI_CUCALL(cudaMemcpy(
        mGridIdxArray.Data(), seepage_flow->GetParticle2CellPtr(),
        sizeof(size_t) * particles->Size(), cudaMemcpyDeviceToDevice));
    thrust::sort_by_key(
        thrust::device, mGridIdxArray.Data(),
        mGridIdxArray.Data() + particles->Size(),
        thrust::make_zip_iterator(thrust::make_tuple(
            seepage_flow->GetAngularVelPtr(), seepage_flow->GetInertiaPtr(),
            seepage_flow->GetMaxSaturationPtr(), seepage_flow->GetCdA0AsatPtr(),
            seepage_flow->GetAmcAmcpPtr())));
  }

  cudaDeviceSynchronize();
  KIRI_CUKERNAL();
}

CudaGNBoundarySearcher::CudaGNBoundarySearcher(const float3 lp, const float3 hp,
                                               const size_t num,
                                               const float cellSize)
    : CudaGNBaseSearcher(lp, hp, num, cellSize) {}

void CudaGNBoundarySearcher::SortData(const CudaParticlesPtr &particles) {
  auto boundaries = std::dynamic_pointer_cast<CudaBoundaryParticles>(particles);
  KIRI_CUCALL(
      cudaMemcpy(mGridIdxArray.Data(), boundaries->GetParticle2CellPtr(),
                 sizeof(size_t) * particles->Size(), cudaMemcpyDeviceToDevice));
  thrust::sort_by_key(thrust::device, mGridIdxArray.Data(),
                      mGridIdxArray.Data() + particles->Size(),
                      thrust::make_zip_iterator(thrust::make_tuple(
                          boundaries->GetPosPtr(), boundaries->GetLabelPtr())));

  cudaDeviceSynchronize();
  KIRI_CUKERNAL();
}

} // namespace KIRI
