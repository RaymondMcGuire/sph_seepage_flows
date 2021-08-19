/*
 * @Author: Xu.WANG
 * @Date: 2021-02-03 14:33:32
 * @LastEditTime: 2021-03-17 21:33:58
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\particle\cuda_sph_bn_particles.cu
 */

#include <kiri_pbs_cuda/particle/cuda_sph_bn_particles.cuh>
#include <kiri_pbs_cuda/particle/cuda_sph_bn_particles_gpu.cuh>

#include <thrust/device_ptr.h>

namespace KIRI
{

    void CudaSphBNParticles::Advect(const float dt)
    {
        uint num = this->Size();

        auto densityArray = thrust::device_pointer_cast(this->GetDensityPtr());
        // float minDensity = *(thrust::min_element(densityArray, densityArray + num));
        // float maxDensity = *(thrust::max_element(densityArray, densityArray + num));

        float minDensity = 200.f;
        float maxDensity = 2000.f;

        BNAdvect_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
            this->GetPosPtr(),
            this->GetVelPtr(),
            this->GetTmpPosPtr(),
            this->GetTmpVelPtr(),
            this->GetColPtr(),
            this->GetDensityPtr(),
            this->GetNormalPtr(),
            this->GetIsBoundaryPtr(),
            this->Size(),
            dt,
            minDensity,
            maxDensity,
            mEpsilon);

        KIRI_CUCALL(cudaDeviceSynchronize());
        KIRI_CUKERNAL();
    }

} // namespace KIRI
