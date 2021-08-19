/*
 * @Author: Xu.WANG
 * @Date: 2021-02-03 14:33:32
 * @LastEditTime: 2021-03-15 23:40:03
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\particle\cuda_dem_particles.cu
 */

#include <kiri_pbs_cuda/particle/cuda_dem_particles.cuh>
namespace KIRI
{

    void CudaDemParticles::Advect(const float dt, const float damping)
    {
        thrust::transform(thrust::device,
                          mAcc.Data(), mAcc.Data() + Size(),
                          mVel.Data(),
                          mAcc.Data(),
                          [dt, damping] __host__ __device__(const float3 &a, const float3 &lv) {
                              return a * (make_float3(1.f) - damping * sgn(a * (lv + 0.5f * dt * a)));
                          });

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
