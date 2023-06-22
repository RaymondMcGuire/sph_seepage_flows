/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-06-15 10:01:37
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-06-19 15:06:21
 * @FilePath:
 * \sph_seepage_flows\seepage_flow_cuda\src\kiri_pbs_cuda\particle\cuda_sf_particles.cu
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */

#include <kiri_pbs_cuda/particle/cuda_sf_particles.cuh>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>

namespace KIRI {

void CudaSFParticles::Advect(const float dt, const float damping) {
  auto sandAccDataTuple =
      thrust::make_tuple(mLabel.Data(), mAcc.Data(), mVel.Data());
  auto sandAccDataIterator = thrust::make_zip_iterator(sandAccDataTuple);

  thrust::transform(thrust::device, sandAccDataIterator,
                    sandAccDataIterator + Size(), mAcc.Data(),
                    AccDampingForSand(dt, damping));

  auto sandAngularAccDataTuple =
      thrust::make_tuple(mLabel.Data(), mAngularAcc.Data(), mAngularVel.Data());
  auto sandAngularAccDataIterator =
      thrust::make_zip_iterator(sandAngularAccDataTuple);

  thrust::transform(thrust::device, sandAngularAccDataIterator,
                    sandAngularAccDataIterator + Size(), mAngularAcc.Data(),
                    AccDampingForSand(dt, damping));

  thrust::transform(
      thrust::device, mVel.Data(), mVel.Data() + Size(), mAcc.Data(),
      mVel.Data(), [dt] __host__ __device__(const float3 &lv, const float3 &a) {
        return lv + dt * a;
      });

  thrust::transform(
      thrust::device, mAngularVel.Data(), mAngularVel.Data() + Size(),
      mAngularAcc.Data(), mAngularVel.Data(),
      [dt] __host__ __device__(const float3 &lv, const float3 &a) {
        return lv + dt * a;
      });

  thrust::transform(
      thrust::device, mPos.Data(), mPos.Data() + Size(), mVel.Data(),
      mPos.Data(), [dt] __host__ __device__(const float3 &lp, const float3 &v) {
        return lp + dt * v;
      });

  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();
}

void CudaSFParticles::AddSphParticles(Vec_Float3 pos, float3 col, float3 vel,
                                      float mass, float radius) {
  size_t num = pos.size();

  if (this->Size() + num >= this->MaxSize()) {
    printf("Current SPH particles numbers exceed maximum particles. \n");
    return;
  }

  KIRI_CUCALL(cudaMemcpy(this->GetPosPtr() + this->Size(), &pos[0],
                         sizeof(float3) * num, cudaMemcpyHostToDevice));

  // add id
  thrust::device_vector<size_t> new_ids(num);
  thrust::sequence(new_ids.begin(), new_ids.end(), this->Size() + 1);
  thrust::copy(new_ids.begin(), new_ids.end(), this->GetIdPtr() + this->Size());

  thrust::fill(thrust::device, this->GetLabelPtr() + this->Size(),
               this->GetLabelPtr() + this->Size() + num, 0);
  thrust::fill(thrust::device, this->GetColPtr() + this->Size(),
               this->GetColPtr() + this->Size() + num, col);
  thrust::fill(thrust::device, this->GetVelPtr() + this->Size(),
               this->GetVelPtr() + this->Size() + num, vel);
  thrust::fill(thrust::device, this->GetMassPtr() + this->Size(),
               this->GetMassPtr() + this->Size() + num, mass);
  thrust::fill(thrust::device, this->GetRadiusPtr() + this->Size(),
               this->GetRadiusPtr() + this->Size() + num, radius);
  thrust::fill(thrust::device, this->GetInertiaPtr() + this->Size(),
               this->GetInertiaPtr() + this->Size() + num,
               2.f / 5.f * mass * radius * radius);
  mNumOfParticles += num;
}

} // namespace KIRI
