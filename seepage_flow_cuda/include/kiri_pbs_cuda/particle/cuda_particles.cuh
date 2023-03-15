/*
 * @Author: Xu.WANG
 * @Date: 2021-02-04 12:36:10
 * @LastEditTime: 2022-03-20 15:38:27
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath:
 * \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\particle\cuda_particles.cuh
 */

#ifndef _CUDA_PARTICLES_CUH_
#define _CUDA_PARTICLES_CUH_

#pragma once

#include <kiri_pbs_cuda/data/cuda_array.cuh>

namespace KIRI {
class CudaParticles {
public:
  explicit CudaParticles(const size_t numOfMaxParticles)
      : mPos(numOfMaxParticles), mParticle2Cell(numOfMaxParticles),
        mNumOfParticles(numOfMaxParticles),
        mNumOfMaxParticles(numOfMaxParticles) {}

  explicit CudaParticles(const Vec_Float3 &p)
      : mPos(p.size()), mParticle2Cell(p.size()), mNumOfParticles(p.size()),
        mNumOfMaxParticles(p.size()) {
    KIRI_CUCALL(cudaMemcpy(mPos.Data(), &p[0], sizeof(float3) * p.size(),
                           cudaMemcpyHostToDevice));
  }

  explicit CudaParticles(const size_t numOfMaxParticles, const Vec_Float3 &p)
      : mPos(numOfMaxParticles), mParticle2Cell(numOfMaxParticles),
        mNumOfParticles(p.size()), mNumOfMaxParticles(numOfMaxParticles) {
    KIRI_CUCALL(cudaMemcpy(mPos.Data(), &p[0], sizeof(float3) * p.size(),
                           cudaMemcpyHostToDevice));
  }

  CudaParticles(const CudaParticles &) = delete;
  CudaParticles &operator=(const CudaParticles &) = delete;

  virtual ~CudaParticles() noexcept {}

  inline size_t Size() const { return mNumOfParticles; }
  inline size_t MaxSize() const { return mNumOfMaxParticles; }
  inline float3 *GetPosPtr() const { return mPos.Data(); }
  inline size_t *GetParticle2CellPtr() const { return mParticle2Cell.Data(); }

protected:
  size_t mNumOfParticles;
  size_t mNumOfMaxParticles;
  CudaArray<float3> mPos;
  CudaArray<size_t> mParticle2Cell;
};

typedef SharedPtr<CudaParticles> CudaParticlesPtr;
} // namespace KIRI

#endif