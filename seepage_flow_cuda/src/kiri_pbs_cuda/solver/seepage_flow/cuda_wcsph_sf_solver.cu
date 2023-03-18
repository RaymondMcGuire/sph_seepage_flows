/*
 * @Author: Xu.WANG
 * @Date: 2021-02-03 17:49:11
 * @LastEditTime: 2021-04-08 00:05:31
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath:
 * \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\solver\seepage_flow\cuda_wcsph_sf_solver.cu
 */

#include <kiri_pbs_cuda/solver/seepageflow/cuda_wcsph_sf_solver.cuh>
#include <kiri_pbs_cuda/solver/seepageflow/cuda_wcsph_sf_solver_gpu.cuh>
#include <kiri_pbs_cuda/thrust_helper/helper_thrust.cuh>
#include <thrust/device_ptr.h>
namespace KIRI {

void CudaWCSphSFSolver::ComputePressure(CudaSFParticlesPtr &particles,
                                        const float rho0, const float stiff) {
  _ComputeSFPressureByTait_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
      particles->GetLabelPtr(), particles->GetDensityPtr(),
      particles->GetPressurePtr(), particles->Size(), rho0, stiff,
      mNegativeScale);

  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();
}

void CudaWCSphSFSolver::ComputeSubTimeStepsByCFL(CudaSFParticlesPtr &particles,
                                                 const float sphMass,
                                                 const float dt,
                                                 const float kernelRadius,
                                                 float renderInterval) {

  auto accArray = thrust::device_pointer_cast(particles->GetAccPtr());
  float3 maxAcc =
      *(thrust::max_element(accArray, accArray + particles->Size(),
                            ThrustHelper::CompareLengthCuda<float3>()));

  float maxForceMagnitude = length(maxAcc) * sphMass;
  float timeStepLimitBySpeed =
      mTimeStepLimitBySpeedFactor * kernelRadius / mSpeedOfSound;
  float timeStepLimitByForce =
      mTimeStepLimitByForceFactor *
      std::sqrt(kernelRadius * sphMass / maxForceMagnitude);
  float desiredTimeStep =
      std::min(mTimeStepLimitScale *
                   std::min(timeStepLimitBySpeed, timeStepLimitByForce),
               dt);

  mNumOfSubTimeSteps = static_cast<size_t>(renderInterval / desiredTimeStep);

  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();
}

} // namespace KIRI
