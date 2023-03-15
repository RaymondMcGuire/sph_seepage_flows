/*
 * @Author: Xu.WANG
 * @Date: 2020-07-26 17:30:04
 * @LastEditTime: 2022-03-20 11:26:59
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath:
 * \sph_seepage_flows\seepage_flows_cuda\include\kiri_pbs_cuda\searcher\cuda_neighbor_searcher.cuh
 */

#ifndef _CUDA_NEIGHBOR_SEARCHER_CUH_
#define _CUDA_NEIGHBOR_SEARCHER_CUH_

#pragma once

#include <kiri_pbs_cuda/particle/cuda_boundary_particles.cuh>
namespace KIRI {

enum SearcherParticleType { SEEPAGE = 0, SEEPAGE_MULTI = 1 };

class CudaGNBaseSearcher {
public:
  explicit CudaGNBaseSearcher(const float3 lowestPoint,
                              const float3 highestPoint,
                              const size_t numOfParticles,
                              const float cellSize);

  CudaGNBaseSearcher(const CudaGNBaseSearcher &) = delete;
  CudaGNBaseSearcher &operator=(const CudaGNBaseSearcher &) = delete;

  virtual ~CudaGNBaseSearcher() noexcept {}

  float3 GetLowestPoint() const { return mLowestPoint; }
  float3 GetHighestPoint() const { return mHighestPoint; }
  float GetCellSize() const { return mCellSize; }
  int3 GetGridSize() const { return mGridSize; }

  size_t *GetCellStartPtr() const { return mCellStart.Data(); }
  const CudaArray<size_t> &GetCellStart() const { return mCellStart; }

  size_t *GetGridIdxArrayPtr() const { return mGridIdxArray.Data(); }
  const CudaArray<size_t> &GetGridIdxArray() const { return mGridIdxArray; }

  void BuildGNSearcher(const CudaParticlesPtr &particles);

protected:
  const size_t mCudaGridSize;
  const int3 mGridSize;
  const float mCellSize;
  const float3 mLowestPoint;
  const float3 mHighestPoint;
  const size_t mNumOfGridCells;
  const size_t mMaxNumOfParticles;

  CudaArray<size_t> mGridIdxArray;
  CudaArray<size_t> mCellStart;

  virtual void SortData(const CudaParticlesPtr &particles) = 0;
};

class CudaGNSearcher final : public CudaGNBaseSearcher {
public:
  explicit CudaGNSearcher(const float3 lp, const float3 hp, const size_t num,
                          const float cellSize,
                          const SearcherParticleType type);

  CudaGNSearcher(const CudaGNSearcher &) = delete;
  CudaGNSearcher &operator=(const CudaGNSearcher &) = delete;

  virtual ~CudaGNSearcher() noexcept {}

  inline constexpr SearcherParticleType GetSearcherType() const {
    return mSearcherParticleType;
  }

protected:
  virtual void SortData(const CudaParticlesPtr &particles) override final;

private:
  SearcherParticleType mSearcherParticleType;
};

class CudaGNBoundarySearcher final : public CudaGNBaseSearcher {
public:
  explicit CudaGNBoundarySearcher(const float3 lp, const float3 hp,
                                  const size_t num, const float cellSize);

  CudaGNBoundarySearcher(const CudaGNBoundarySearcher &) = delete;
  CudaGNBoundarySearcher &operator=(const CudaGNBoundarySearcher &) = delete;

  virtual ~CudaGNBoundarySearcher() noexcept {}

protected:
  virtual void SortData(const CudaParticlesPtr &particles) override final;
};

typedef SharedPtr<CudaGNBaseSearcher> CudaGNBaseSearcherPtr;
typedef SharedPtr<CudaGNSearcher> CudaGNSearcherPtr;
typedef SharedPtr<CudaGNBoundarySearcher> CudaGNBoundarySearcherPtr;
} // namespace KIRI

#endif /* CudaNeighborSearcher */