/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-04-17 15:08:41
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-12-13 22:48:45
 * @FilePath: \sph_seepage_flows\seepage_flows_cuda\src\kiri_pbs_cuda\particle\cuda_sf_particles.cu
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
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
        auto num = pos.size();
        auto current_size = this->Size();
        if (current_size + num >= this->MaxSize())
        {
            printf("Current SPH particles numbers exceed maximum particles. \n");
            return;
        }

        // generate id
        Vec_SizeT id;
        for (auto i = 0; i < num; i++)
            id.emplace_back(mNumOfParticles + i + 1);

        KIRI_CUCALL(cudaMemcpy(this->GetIdPtr() + current_size, &id[0], sizeof(size_t) * num, cudaMemcpyHostToDevice));
        KIRI_CUCALL(cudaMemcpy(this->GetPosPtr() + current_size, &pos[0], sizeof(float3) * num, cudaMemcpyHostToDevice));
        thrust::fill(thrust::device, this->GetLabelPtr() + current_size, this->GetLabelPtr() + current_size + num, 0);
        thrust::fill(thrust::device, this->GetColPtr() + current_size, this->GetColPtr() + current_size + num, col);
        thrust::fill(thrust::device, this->GetVelPtr() + current_size, this->GetVelPtr() + current_size + num, vel);
        thrust::fill(thrust::device, this->GetMassPtr() + current_size, this->GetMassPtr() + current_size + num, mass);
        thrust::fill(thrust::device, this->GetRadiusPtr() + current_size, this->GetRadiusPtr() + current_size + num, radius);

        mNumOfParticles += num;
    }

} // namespace KIRI
