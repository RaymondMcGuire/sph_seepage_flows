/*** 
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-05-19 20:19:33
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-06-08 23:49:02
 * @FilePath: \sph_seepage_flows\seepage_flow_cuda\src\kiri_pbs_cuda\particle\cuda_particles.cu
 * @Description: 
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved. 
 */

#include <kiri_pbs_cuda/particle/cuda_particles.cuh>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
namespace KIRI {

CudaParticles::CudaParticles(const Vec_Float3 &p)
    : mPos(p.size()), mId(p.size()), mParticle2Cell(p.size()),
      mNumOfParticles(p.size()), mNumOfMaxParticles(p.size()) {
  thrust::device_ptr<size_t> id_ptr(mId.Data());
  thrust::device_ptr<size_t> id_end_ptr(mId.Data() + p.size());
  thrust::sequence(id_ptr, id_end_ptr, 1);
  KIRI_CUCALL(cudaMemcpy(mPos.Data(), &p[0], sizeof(float3) * p.size(),
                         cudaMemcpyHostToDevice));
}

CudaParticles::CudaParticles(const size_t numOfMaxParticles,
                             const Vec_Float3 &p)
    : mPos(numOfMaxParticles), mId(numOfMaxParticles),
      mParticle2Cell(numOfMaxParticles), mNumOfParticles(p.size()),
      mNumOfMaxParticles(numOfMaxParticles) {

  thrust::device_ptr<size_t> id_ptr(mId.Data());
  thrust::device_ptr<size_t> id_end_ptr(mId.Data() + p.size());
  thrust::sequence(id_ptr, id_end_ptr, 1);
  KIRI_CUCALL(cudaMemcpy(mPos.Data(), &p[0], sizeof(float3) * p.size(),
                         cudaMemcpyHostToDevice));
}

} // namespace KIRI
