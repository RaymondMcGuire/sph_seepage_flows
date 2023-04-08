/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-03-23 15:57:38
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-03-25 23:12:35
 * @FilePath:
 * \sph_seepage_flows\seepage_flow_cuda\src\kiri_pbs_cuda\particle\cuda_dfsf_particles.cu
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */

#include <kiri_pbs_cuda/particle/cuda_dfsf_particles.cuh>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

namespace KIRI {

typedef thrust::tuple<size_t, float3, float3> DFSFAccDataType;
struct AdvectVelForDFFluid {
  float mDt;
  __host__ __device__ AdvectVelForDFFluid(const float dt) : mDt(dt) {}

  __host__ __device__ float3 operator()(const DFSFAccDataType &data) const {
    size_t label = data.get<0>();
    float3 acc = data.get<1>();
    float3 lv = data.get<2>();

    if (label == 0)
      return lv + mDt * acc;
    else
      return lv;
  }
};

struct AdvectVelForDFSand {
  float mDt;
  __host__ __device__ AdvectVelForDFSand(const float dt) : mDt(dt) {}

  __host__ __device__ float3 operator()(const DFSFAccDataType &data) const {
    size_t label = data.get<0>();
    float3 acc = data.get<1>();
    float3 lv = data.get<2>();

    if (label == 1)
      return lv + mDt * acc;
    else
      return lv;
  }
};

void CudaDFSFParticles::AdvectFluidVel(const float dt) {

  auto waterAccDataTuple =
      thrust::make_tuple(mLabel.Data(), mAcc.Data(), mVel.Data());
  auto waterAccDataIterator = thrust::make_zip_iterator(waterAccDataTuple);

  thrust::transform(thrust::device, waterAccDataIterator,
                    waterAccDataIterator + Size(), mVel.Data(),
                    AdvectVelForDFFluid(dt));

  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();
}

void CudaDFSFParticles::Advect(const float dt, const float damping) {

  // sand damping
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
      thrust::device, mAngularVel.Data(), mAngularVel.Data() + Size(),
      mAngularAcc.Data(), mAngularVel.Data(),
      [dt] __host__ __device__(const float3 &lv, const float3 &a) {
        return lv + dt * a;
      });

  // sand vel advect
  auto sandVelDataTuple =
      thrust::make_tuple(mLabel.Data(), mAcc.Data(), mVel.Data());
  auto sandVelDataIterator = thrust::make_zip_iterator(sandVelDataTuple);

  thrust::transform(thrust::device, sandVelDataIterator,
                    sandVelDataIterator + Size(), mVel.Data(),
                    AdvectVelForDFSand(dt));

  // water and sand pos advect
  thrust::transform(
      thrust::device, mPos.Data(), mPos.Data() + Size(), mVel.Data(),
      mPos.Data(), [dt] __host__ __device__(const float3 &lp, const float3 &v) {
        return lp + dt * v;
      });

  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();
}

} // namespace KIRI
