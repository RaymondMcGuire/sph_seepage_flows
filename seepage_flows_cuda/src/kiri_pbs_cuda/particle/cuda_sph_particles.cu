/*
 * @Author: Xu.WANG
 * @Date: 2021-02-03 14:33:32
 * @LastEditTime: 2021-03-19 23:51:05
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\particle\cuda_sph_particles.cu
 */

#include <kiri_pbs_cuda/particle/cuda_sph_particles.cuh>
namespace KIRI
{
    void CudaSphParticles::AddSphParticles(Vec_Float3 pos, float3 col, float3 vel, float mass)
    {
        size_t num = pos.size();

        if (this->Size() + num >= this->MaxSize())
        {
            printf("Current SPH particles numbers exceed maximum particles. \n");
            return;
        }

        KIRI_CUCALL(cudaMemcpy(this->GetPosPtr() + this->Size(), &pos[0], sizeof(float3) * num, cudaMemcpyHostToDevice));
        thrust::fill(thrust::device, this->GetColPtr() + this->Size(), this->GetColPtr() + this->Size() + num, col);
        thrust::fill(thrust::device, this->GetVelPtr() + this->Size(), this->GetVelPtr() + this->Size() + num, vel);
        thrust::fill(thrust::device, this->GetMassPtr() + this->Size(), this->GetMassPtr() + this->Size() + num, mass);

        mNumOfParticles += num;
    }

    void CudaSphParticles::Advect(const float dt)
    {
        thrust::transform(thrust::device,
                          mVel.Data(), mVel.Data() + Size(),
                          mAcc.Data(),
                          mVel.Data(),
                          [dt] __host__ __device__(const float3 &lv, const float3 &a) {
                              return lv + dt * a;
                          });

        thrust::transform(thrust::device,
                          mPos.Data(), mPos.Data() + Size(),
                          mVel.Data(),
                          mPos.Data(),
                          [dt] __host__ __device__(const float3 &lp, const float3 &v) {
                              return lp + dt * v;
                          });

        KIRI_CUCALL(cudaDeviceSynchronize());
        KIRI_CUKERNAL();
    }

} // namespace KIRI
