/*** 
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-05-14 20:01:11
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-05-17 20:34:28
 * @FilePath: \sph_seepage_flows\seepage_flow_cuda\src\kiri_pbs_cuda\solver\seepage_flow\cuda_dfsph_sf_solver.cu
 * @Description: 
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved. 
 */
#include <kiri_pbs_cuda/solver/seepageflow/cuda_dfsph_sf_solver.cuh>
#include <kiri_pbs_cuda/solver/seepageflow/cuda_dfsph_sf_solver_gpu.cuh>
#include <kiri_pbs_cuda/solver/seepageflow/cuda_sph_sf_solver_gpu.cuh>
#include <kiri_pbs_cuda/thrust_helper/helper_thrust.cuh>
#include <thrust/device_ptr.h>
namespace KIRI {

void CudaDFSphSFSolver::ComputeDensity(
    CudaSFParticlesPtr &particles, CudaBoundaryParticlesPtr &boundaries,
    const float rho0, const float rho1, const CudaArray<size_t> &cellStart,
    const CudaArray<size_t> &boundaryCellStart, const float3 lowestPoint,
    const float kernelRadius, const int3 gridSize) {

  auto data = std::dynamic_pointer_cast<CudaDFSFParticles>(particles);
  _ComputeSFDensity_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
      data->GetDensityPtr(), data->GetLabelPtr(), data->GetPosPtr(),
      data->GetMassPtr(), rho0, rho1, data->Size(), cellStart.Data(),
      boundaries->GetPosPtr(), boundaries->GetVolumePtr(),
      boundaryCellStart.Data(), gridSize,
      ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
      ThrustHelper::GridXYZ2GridHash(gridSize), CubicKernel(kernelRadius));
  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();
}

void CudaDFSphSFSolver::ComputeDFPressure(CudaDFSFParticlesPtr &particles,
                                          const float rho0) {
  _ComputeDFSFPressure_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
      particles->GetPressurePtr(), particles->GetLabelPtr(),
      particles->GetDensityPtr(), particles->GetStiffPtr(), particles->Size(),
      rho0, 1.f);
  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();
}

void CudaDFSphSFSolver::AdvectDFSPHVelocity(CudaDFSFParticlesPtr &fluids) {

  fluids->AdvectFluidVel(mDt);
}

void CudaDFSphSFSolver::ComputeTimeStepsByCFL(
    CudaDFSFParticlesPtr &fluids, const float particleRadius, const float dt,
    const float timeIntervalInSeconds) {

  _ComputeVelMag_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
      fluids->GetVelMagPtr(), fluids->GetLabelPtr(), fluids->GetVelPtr(),
      fluids->GetAccPtr(), mDt, fluids->Size());

  auto vel_mag_array = thrust::device_pointer_cast(fluids->GetVelMagPtr());
  float max_vel_mag =
      *(thrust::max_element(vel_mag_array, vel_mag_array + fluids->Size()));

  auto diam = 2.f * particleRadius;
  mDt = CFL_FACTOR * 0.4f * (diam / sqrt(max_vel_mag));
  mDt = max(mDt, CFL_MIN_TIMESTEP_SIZE);
  mDt = min(mDt, CFL_MAX_TIMESTEP_SIZE);
  mDt = min(mDt, dt);

  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();
}

void CudaDFSphSFSolver::ComputeDFSPHAlpha(
    CudaDFSFParticlesPtr &fluids, CudaBoundaryParticlesPtr &boundaries,
    const float rho0, const CudaArray<size_t> &cellStart,
    const CudaArray<size_t> &boundaryCellStart, const float3 lowestPoint,
    const float kernelRadius, const int3 gridSize) {

  _ComputeAlpha_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
      fluids->GetAlphaPtr(), fluids->GetLabelPtr(), fluids->GetPosPtr(),
      fluids->GetMassPtr(), fluids->GetDensityPtr(), rho0, fluids->Size(),
      cellStart.Data(), boundaries->GetPosPtr(), boundaries->GetVolumePtr(),
      boundaryCellStart.Data(), gridSize,
      ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
      ThrustHelper::GridXYZ2GridHash(gridSize), CubicKernelGrad(kernelRadius));

  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();
}

size_t CudaDFSphSFSolver::ApplyDivergenceSolver(
    CudaDFSFParticlesPtr &fluids, CudaBoundaryParticlesPtr &boundaries,
    const float rho0, const CudaArray<size_t> &cellStart,
    const CudaArray<size_t> &boundaryCellStart, const float3 lowestPoint,
    const float kernelRadius, const int3 gridSize) {

  auto num = fluids->Size();

  // Compute velocity of density change
  _ComputeDivgenceError_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
      fluids->GetStiffPtr(), fluids->GetDensityErrorPtr(),
      fluids->GetLabelPtr(), fluids->GetAlphaPtr(), fluids->GetPosPtr(),
      fluids->GetVelPtr(), fluids->GetMassPtr(), fluids->GetDensityPtr(), rho0,
      mDt, num, cellStart.Data(), boundaries->GetPosPtr(),
      boundaries->GetVolumePtr(), boundaryCellStart.Data(), gridSize,
      ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
      ThrustHelper::GridXYZ2GridHash(gridSize), CubicKernelGrad(kernelRadius));

  auto iter = 0;
  auto total_error = std::numeric_limits<float>::max();

  while ((total_error > mDivergenceErrorThreshold * num * rho0 ||
          (iter < mDivergenceMinIter)) &&
         (iter < mDivergenceMaxIter)) {

    _CorrectDivergenceByJacobi_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        fluids->GetVelPtr(), fluids->GetLabelPtr(), fluids->GetStiffPtr(),
        fluids->GetPosPtr(), fluids->GetMassPtr(), rho0, num, cellStart.Data(),
        boundaries->GetPosPtr(), boundaries->GetVolumePtr(),
        boundaryCellStart.Data(), gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize),
        CubicKernelGrad(kernelRadius));

    _ComputeDivgenceError_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        fluids->GetStiffPtr(), fluids->GetDensityErrorPtr(),
        fluids->GetLabelPtr(), fluids->GetAlphaPtr(), fluids->GetPosPtr(),
        fluids->GetVelPtr(), fluids->GetMassPtr(), fluids->GetDensityPtr(),
        rho0, mDt, num, cellStart.Data(), boundaries->GetPosPtr(),
        boundaries->GetVolumePtr(), boundaryCellStart.Data(), gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize),
        CubicKernelGrad(kernelRadius));

    iter++;

    total_error = thrust::reduce(
        thrust::device_ptr<float>(fluids->GetDensityErrorPtr()),
        thrust::device_ptr<float>(fluids->GetDensityErrorPtr() + num), 0.f,
        ThrustHelper::AbsPlus<float>());
  }

  // printf("divergence iter=%d, total_error=%.6f \n", iter,
  //        total_error);
  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();

  return iter;
}

size_t CudaDFSphSFSolver::ApplyPressureSolver(
    CudaDFSFParticlesPtr &fluids, CudaBoundaryParticlesPtr &boundaries,
    const float rho0, const CudaArray<size_t> &cellStart,
    const CudaArray<size_t> &boundaryCellStart, const float3 lowestPoint,
    const float kernelRadius, const int3 gridSize) {

  auto num = fluids->Size();

  // use warm stiff
  _CorrectPressureByJacobi_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
      fluids->GetVelPtr(), fluids->GetLabelPtr(), fluids->GetWarmStiffPtr(),
      fluids->GetPosPtr(), fluids->GetMassPtr(), rho0, mDt, num,
      cellStart.Data(), boundaries->GetPosPtr(), boundaries->GetVolumePtr(),
      boundaryCellStart.Data(), gridSize,
      ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
      ThrustHelper::GridXYZ2GridHash(gridSize), CubicKernelGrad(kernelRadius));

  _ComputeDensityError_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
      fluids->GetDensityErrorPtr(), fluids->GetStiffPtr(),
      fluids->GetLabelPtr(), fluids->GetAlphaPtr(), fluids->GetPosPtr(),
      fluids->GetVelPtr(), fluids->GetMassPtr(), fluids->GetDensityPtr(), rho0,
      mDt, num, cellStart.Data(), boundaries->GetPosPtr(),
      boundaries->GetVolumePtr(), boundaryCellStart.Data(), gridSize,
      ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
      ThrustHelper::GridXYZ2GridHash(gridSize), CubicKernelGrad(kernelRadius));

  // reset warm stiffness
  KIRI_CUCALL(cudaMemcpy(fluids->GetWarmStiffPtr(), fluids->GetStiffPtr(),
                         sizeof(float) * num, cudaMemcpyDeviceToDevice));

  auto iter = 0;
  auto total_error = std::numeric_limits<float>::max();

  while ((total_error > mPressureErrorThreshold * num * rho0 ||
          (iter < mPressureMinIter)) &&
         (iter < mPressureMaxIter)) {

    _CorrectPressureByJacobi_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        fluids->GetVelPtr(), fluids->GetLabelPtr(), fluids->GetStiffPtr(),
        fluids->GetPosPtr(), fluids->GetMassPtr(), rho0, mDt, num,
        cellStart.Data(), boundaries->GetPosPtr(), boundaries->GetVolumePtr(),
        boundaryCellStart.Data(), gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize),
        CubicKernelGrad(kernelRadius));

    _ComputeDensityError_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        fluids->GetDensityErrorPtr(), fluids->GetStiffPtr(),
        fluids->GetLabelPtr(), fluids->GetAlphaPtr(), fluids->GetPosPtr(),
        fluids->GetVelPtr(), fluids->GetMassPtr(), fluids->GetDensityPtr(),
        rho0, mDt, num, cellStart.Data(), boundaries->GetPosPtr(),
        boundaries->GetVolumePtr(), boundaryCellStart.Data(), gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize),
        CubicKernelGrad(kernelRadius));

    thrust::transform(thrust::device, fluids->GetWarmStiffPtr(),
                      fluids->GetWarmStiffPtr() + num, fluids->GetStiffPtr(),
                      fluids->GetWarmStiffPtr(), thrust::plus<float>());
    iter++;

    if (iter >= mPressureMinIter) {
      total_error = thrust::reduce(
          thrust::device_ptr<float>(fluids->GetDensityErrorPtr()),
          thrust::device_ptr<float>(fluids->GetDensityErrorPtr() + num), 0.f,
          ThrustHelper::AbsPlus<float>());
    }
  }

  //   printf("Total Iteration Num=%d; Total Error=%.6f; Threshold=%.6f \n",
  //   iter,
  //          total_error, mPressureErrorThreshold * num * rho0);

  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();

  return iter;
}

void CudaDFSphSFSolver::ComputeSFSandLinearMomentum(
    CudaSFParticlesPtr &particles, CudaBoundaryParticlesPtr &boundaries,
    const CudaArray<size_t> &cellStart,
    const CudaArray<size_t> &boundaryCellStart, const float boundaryRadius,
    const float maxForceFactor, const float young, const float poisson,
    const float tanFrictionAngle, const float c0, const float csat,
    const float cmc, const float cmcp, const float cd, const float gravity,
    const float rho0, const float3 lowestPoint, const float3 highestPoint,
    const float kernelRadius, const int3 gridSize) {
  auto data = std::dynamic_pointer_cast<CudaDFSFParticles>(particles);
  _ComputeDFSFSandLinearMomentum_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
      data->GetAvgDragForcePtr(), data->GetAccPtr(), data->GetAngularAccPtr(),
      data->GetLabelPtr(), data->GetPosPtr(), data->GetVelPtr(),
      data->GetAngularVelPtr(), data->GetMassPtr(), data->GetInertiaPtr(),
      data->GetDensityPtr(), data->GetStiffPtr(), data->GetVoidagePtr(),
      data->GetSaturationPtr(), data->GetAvgFlowVelPtr(),
      data->GetAvgAdhesionForcePtr(), data->GetRadiusPtr(), boundaryRadius,
      maxForceFactor, young, poisson, tanFrictionAngle, cd, gravity, rho0,
      data->Size(), lowestPoint, highestPoint, cellStart.Data(),
      boundaries->GetPosPtr(), boundaries->GetLabelPtr(),
      boundaryCellStart.Data(), gridSize,
      ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
      ThrustHelper::GridXYZ2GridHash(gridSize),
      QuadraticBezierCoeff(c0, cmc, cmcp, csat), CubicKernel(kernelRadius),
      CubicKernelGrad(kernelRadius));
  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();
}

void CudaDFSphSFSolver::ComputeMultiSFSandLinearMomentum(
    CudaSFParticlesPtr &particles, CudaBoundaryParticlesPtr &boundaries,
    const CudaArray<size_t> &cellStart,
    const CudaArray<size_t> &boundaryCellStart, const float boundaryRadius,
    const float maxForceFactor, const float young, const float poisson,
    const float tanFrictionAngle, const float c0, const float csat,
    const float cmc, const float cmcp, const float gravity, const float rho0,
    const float3 lowestPoint, const float3 highestPoint,
    const float kernelRadius, const int3 gridSize) {
  auto data = std::dynamic_pointer_cast<CudaDFSFParticles>(particles);
  _ComputeMultiDFSFSandLinearMomentum_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
      data->GetAvgDragForcePtr(), data->GetAccPtr(), data->GetAngularAccPtr(),
      data->GetLabelPtr(), data->GetPosPtr(), data->GetVelPtr(),
      data->GetAngularVelPtr(), data->GetMassPtr(), data->GetInertiaPtr(),
      data->GetDensityPtr(), data->GetStiffPtr(), data->GetVoidagePtr(),
      data->GetSaturationPtr(), data->GetAvgFlowVelPtr(),
      data->GetAvgAdhesionForcePtr(), data->GetRadiusPtr(), boundaryRadius,
      maxForceFactor, young, poisson, tanFrictionAngle, data->GetCdA0AsatPtr(),
      gravity, rho0, data->Size(), lowestPoint, highestPoint, cellStart.Data(),
      boundaries->GetPosPtr(), boundaries->GetLabelPtr(),
      boundaryCellStart.Data(), gridSize,
      ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
      ThrustHelper::GridXYZ2GridHash(gridSize),
      QuadraticBezierCoeff(c0, cmc, cmcp, csat), CubicKernel(kernelRadius),
      CubicKernelGrad(kernelRadius));
  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();
}

void CudaDFSphSFSolver::ComputeSFWaterLinearMomentum(
    CudaSFParticlesPtr &particles, CudaBoundaryParticlesPtr &boundaries,
    const CudaArray<size_t> &cellStart,
    const CudaArray<size_t> &boundaryCellStart, const float rho0,
    const float nu, const float bnu, const float3 lowestPoint,
    const float kernelRadius, const int3 gridSize) {
  auto data = std::dynamic_pointer_cast<CudaDFSFParticles>(particles);
  _ComputeDFSFWaterLinearMomentum_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
      data->GetAccPtr(), data->GetLabelPtr(), data->GetPosPtr(),
      data->GetVelPtr(), data->GetMassPtr(), data->GetDensityPtr(),
      data->GetStiffPtr(), data->GetVoidagePtr(), data->GetAvgDragForcePtr(),
      data->GetAdhesionForcePtr(), rho0, nu, bnu, data->Size(),
      cellStart.Data(), boundaries->GetPosPtr(), boundaries->GetVolumePtr(),
      boundaryCellStart.Data(), gridSize,
      ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
      ThrustHelper::GridXYZ2GridHash(gridSize), CubicKernel(kernelRadius),
      CubicKernelGrad(kernelRadius));
  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();
}

void CudaDFSphSFSolver::Advect(CudaSFParticlesPtr &particles,
                               CudaBoundaryParticlesPtr &boundaries,
                               const CudaArray<size_t> &boundaryCellStart,
                               const float waterRadius, const float dt,
                               const float damping, const float3 lowestPoint,
                               const float3 highestPoint,
                               const float kernelRadius, const int3 gridSize) {
  auto data = std::dynamic_pointer_cast<CudaDFSFParticles>(particles);
  size_t num = data->Size();
  data->Advect(dt, damping);
  // _SFWaterBoundaryConstrain_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
  //     data->GetPosPtr(), data->GetVelPtr(), data->GetLabelPtr(), num,
  //     lowestPoint, highestPoint, waterRadius, boundaries->GetPosPtr(),
  //     boundaries->GetLabelPtr(), boundaryCellStart.Data(), gridSize,
  //     ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
  //     ThrustHelper::GridXYZ2GridHash(gridSize));

  thrust::fill(thrust::device, data->GetAdhesionForcePtr(),
               data->GetAdhesionForcePtr() + num, make_float3(0.f));
  thrust::fill(thrust::device, data->GetAvgAdhesionForcePtr(),
               data->GetAvgAdhesionForcePtr() + num, make_float3(0.f));
  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();
}

} // namespace KIRI
