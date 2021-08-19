/*
 * @Author: Xu.WANG
 * @Date: 2021-02-03 14:33:32
 * @LastEditTime: 2021-07-19 00:35:14
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\particle\cuda_iisf_particles.cu
 */

#include <kiri_pbs_cuda/particle/cuda_iisf_particles.cuh>

#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>

namespace KIRI
{
    typedef thrust::tuple<size_t, float3, float3> IISFWaterVelData;
    struct WaterVelAdv
    {
        float mDt;
        __host__ __device__ WaterVelAdv(
            const float dt)
            : mDt(dt)
        {
        }

        __host__ __device__ float3 operator()(const IISFWaterVelData &data) const
        {
            size_t label = data.get<0>();
            float3 acc = data.get<1>();
            float3 lv = data.get<2>();

            if (label == 0)
                return (lv + acc * mDt);
            else
                return lv;
        }
    };

    typedef thrust::tuple<size_t, float3, float3, float3> IISFAccData;
    struct IISFAccAdv
    {
        float mDt;
        __host__ __device__ IISFAccAdv(
            const float dt)
            : mDt(dt)
        {
        }

        __host__ __device__ float3 operator()(const IISFAccData &data) const
        {
            size_t label = data.get<0>();
            float3 acc = data.get<1>();
            float3 pacc = data.get<2>();
            float3 lv = data.get<3>();

            if (label == 0)
                return (lv + pacc * mDt);
            else
                return (lv + acc * mDt);
        }
    };

    void CudaIISFParticles::PredictVelAdvect(const float dt)
    {
        auto waterVelAdvTuple = thrust::make_tuple(mLabel.Data(), mAcc.Data(), mVel.Data());
        auto waterVelAdvIter = thrust::make_zip_iterator(waterVelAdvTuple);

        thrust::transform(thrust::device,
                          waterVelAdvIter, waterVelAdvIter + Size(),
                          mVel.Data(),
                          WaterVelAdv(dt));

        KIRI_CUCALL(cudaDeviceSynchronize());
        KIRI_CUKERNAL();
    }

    void CudaIISFParticles::Advect(const float dt, const float damping)
    {
        auto sfDataTuple = thrust::make_tuple(mLabel.Data(), mAcc.Data(), mVel.Data());
        auto sfDataIterator = thrust::make_zip_iterator(sfDataTuple);

        // sand acc
        thrust::transform(thrust::device,
                          sfDataIterator, sfDataIterator + Size(),
                          mAcc.Data(),
                          AccDampingForSand(dt, damping));

        auto iisfAccTuple = thrust::make_tuple(mLabel.Data(), mAcc.Data(), mPressureAcc.Data(), mVel.Data());
        auto iisfAccIter = thrust::make_zip_iterator(iisfAccTuple);
        // sand water vel
        thrust::transform(thrust::device,
                          iisfAccIter, iisfAccIter + Size(),
                          mVel.Data(),
                          IISFAccAdv(dt));

        // sand water pos
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

} // namespace KIRI
