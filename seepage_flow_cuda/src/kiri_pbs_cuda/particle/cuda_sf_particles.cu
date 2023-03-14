/*
 * @Author: Xu.WANG
 * @Date: 2021-02-03 14:33:32
 * @LastEditTime: 2021-07-19 00:17:39
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\particle\cuda_sf_particles.cu
 */

#include <kiri_pbs_cuda/particle/cuda_sf_particles.cuh>

#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>

namespace KIRI
{

    void CudaSFParticles::Advect(const float dt, const float damping)
    {
        auto sfDataTuple = thrust::make_tuple(mLabel.Data(), mAcc.Data(), mVel.Data());
        auto sfDataIterator = thrust::make_zip_iterator(sfDataTuple);

        thrust::transform(thrust::device,
                          sfDataIterator, sfDataIterator + Size(),
                          mAcc.Data(),
                          AccDampingForSand(dt, damping));

        thrust::transform(thrust::device,
                          mVel.Data(), mVel.Data() + Size(),
                          mAcc.Data(),
                          mVel.Data(),
                          [dt] __host__ __device__(const float3 &lv, const float3 &a)
                          {
                              return lv + dt * a;
                          });

        thrust::transform(thrust::device,
                          mPos.Data(), mPos.Data() + Size(),
                          mVel.Data(),
                          mPos.Data(),
                          [dt] __host__ __device__(const float3 &lp, const float3 &v)
                          {
                              return lp + dt * v;
                          });

        KIRI_CUCALL(cudaDeviceSynchronize());
        KIRI_CUKERNAL();
    }

    void CudaSFParticles::AddSphParticles(Vec_Float3 pos, float3 col, float3 vel, float mass, float radius)
    {
        size_t num = pos.size();

        if (this->Size() + num >= this->MaxSize())
        {
            printf("Current SPH particles numbers exceed maximum particles. \n");
            return;
        }

        KIRI_CUCALL(cudaMemcpy(this->GetPosPtr() + this->Size(), &pos[0], sizeof(float3) * num, cudaMemcpyHostToDevice));
        thrust::fill(thrust::device, this->GetLabelPtr() + this->Size(), this->GetLabelPtr() + this->Size() + num, 0);
        thrust::fill(thrust::device, this->GetColPtr() + this->Size(), this->GetColPtr() + this->Size() + num, col);
        thrust::fill(thrust::device, this->GetVelPtr() + this->Size(), this->GetVelPtr() + this->Size() + num, vel);
        thrust::fill(thrust::device, this->GetMassPtr() + this->Size(), this->GetMassPtr() + this->Size() + num, mass);
        thrust::fill(thrust::device, this->GetRadiusPtr() + this->Size(), this->GetRadiusPtr() + this->Size() + num, radius);

        mNumOfParticles += num;
    }

} // namespace KIRI
