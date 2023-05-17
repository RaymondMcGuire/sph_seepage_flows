/*** 
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-05-14 20:01:11
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-05-17 15:32:20
 * @FilePath: \sph_seepage_flows\seepage_flow_cuda\include\kiri_pbs_cuda\particle\cuda_particles.cuh
 * @Description: 
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved. 
 */

#ifndef _CUDA_PARTICLES_CUH_
#define _CUDA_PARTICLES_CUH_

#pragma once

#include <kiri_pbs_cuda/data/cuda_array.cuh>

namespace KIRI {
class CudaParticles {
public:
  explicit CudaParticles(const size_t numOfMaxParticles)
      : mPos(numOfMaxParticles),mId(numOfMaxParticles), mParticle2Cell(numOfMaxParticles),
        mNumOfParticles(numOfMaxParticles),
        mNumOfMaxParticles(numOfMaxParticles) {}

  explicit CudaParticles(const Vec_Float3 &p);
  explicit CudaParticles(const size_t numOfMaxParticles, const Vec_Float3 &p);

  CudaParticles(const CudaParticles &) = delete;
  CudaParticles &operator=(const CudaParticles &) = delete;

  virtual ~CudaParticles() noexcept {}

  inline size_t Size() const { return mNumOfParticles; }
  inline size_t MaxSize() const { return mNumOfMaxParticles; }

  inline float3 *GetPosPtr() const { return mPos.Data(); }
  inline int *GetIdPtr() const { return mId.Data(); }

  inline size_t *GetParticle2CellPtr() const { return mParticle2Cell.Data(); }

protected:
  size_t mNumOfParticles;
  size_t mNumOfMaxParticles;

  CudaArray<int> mId;
  CudaArray<float3> mPos;
  CudaArray<size_t> mParticle2Cell;
};

typedef SharedPtr<CudaParticles> CudaParticlesPtr;
} // namespace KIRI

#endif