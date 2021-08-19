/*
 * @Author: Xu.WANG
 * @Date: 2021-02-03 17:49:11
 * @LastEditTime: 2021-08-16 00:06:16
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\solver\seepage_flow\cuda_iisph_sf_solver.cu
 */
#include <kiri_pbs_cuda/solver/seepageflow/cuda_iisph_sf_solver.cuh>
#include <kiri_pbs_cuda/solver/seepageflow/cuda_iisph_sf_solver_gpu.cuh>

#include <kiri_pbs_cuda/solver/seepageflow/cuda_sf_utils.cuh>
#include <kiri_pbs_cuda/solver/seepageflow/cuda_sph_sf_solver_gpu.cuh>
#include <kiri_pbs_cuda/particle/cuda_iisf_particles.cuh>

#include <kiri_pbs_cuda/thrust_helper/helper_thrust.cuh>
#include <thrust/device_ptr.h>
namespace KIRI
{

  void CudaIISphSFSolver::PredictVelAdvect(
      CudaSFParticlesPtr &particles,
      const float dt)
  {

    auto data = std::dynamic_pointer_cast<CudaIISFParticles>(particles);
    data->PredictVelAdvect(dt);

    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaIISphSFSolver::ComputeDiiTerm(
      CudaSFParticlesPtr &particles,
      CudaBoundaryParticlesPtr &boundaries,
      const CudaArray<size_t> &cellStart,
      const CudaArray<size_t> &boundaryCellStart,
      const float rho0,
      const float3 lowestPoint,
      const float kernelRadius,
      const int3 gridSize)
  {
    auto data = std::dynamic_pointer_cast<CudaIISFParticles>(particles);
    ComputeIISFDiiTerm_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        data->GetDiiPtr(),
        data->GetLabelPtr(),
        data->GetPosPtr(),
        data->GetMassPtr(),
        data->GetDensityPtr(),
        rho0,
        data->Size(),
        cellStart.Data(),
        boundaries->GetPosPtr(),
        boundaries->GetVolumePtr(),
        boundaryCellStart.Data(),
        gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize),
        SpikyKernelGrad(kernelRadius));

    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaIISphSFSolver::ComputeAiiTerm(
      CudaSFParticlesPtr &particles,
      CudaBoundaryParticlesPtr &boundaries,
      const CudaArray<size_t> &cellStart,
      const CudaArray<size_t> &boundaryCellStart,
      const float rho0,
      const float dt,
      const float3 lowestPoint,
      const float kernelRadius,
      const int3 gridSize)
  {
    auto data = std::dynamic_pointer_cast<CudaIISFParticles>(particles);
    ComputeIISFAiiTerm_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        data->GetAiiPtr(),
        data->GetDensityAdvPtr(),
        data->GetPressurePtr(),
        data->GetLabelPtr(),
        data->GetDiiPtr(),
        data->GetPosPtr(),
        data->GetVelPtr(),
        data->GetAccPtr(),
        data->GetMassPtr(),
        data->GetDensityPtr(),
        data->GetLastPressurePtr(),
        rho0,
        dt,
        data->Size(),
        cellStart.Data(),
        boundaries->GetPosPtr(),
        boundaries->GetVolumePtr(),
        boundaryCellStart.Data(),
        gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize),
        SpikyKernelGrad(kernelRadius));

    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  size_t CudaIISphSFSolver::PressureSolver(
      CudaSFParticlesPtr &particles,
      CudaBoundaryParticlesPtr &boundaries,
      const float rho0,
      const float dt,
      const CudaArray<size_t> &cellStart,
      const CudaArray<size_t> &boundaryCellStart,
      const float3 lowestPoint,
      const float kernelRadius,
      const int3 gridSize)
  {
    auto data = std::dynamic_pointer_cast<CudaIISFParticles>(particles);
    auto num = data->Size();
    auto error = 0.f;
    auto iter = 0;
    auto flag = false;

    while ((!flag || (iter < mMinIter)) && (iter < mMaxIter))
    {
      flag = true;

      error = 0.f;

      ComputeIISFDijPjTerm_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
          data->GetDijPjPtr(),
          data->GetLabelPtr(),
          data->GetPosPtr(),
          data->GetMassPtr(),
          data->GetDensityPtr(),
          data->GetLastPressurePtr(),
          data->Size(),
          cellStart.Data(),
          gridSize,
          ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
          ThrustHelper::GridXYZ2GridHash(gridSize),
          SpikyKernelGrad(kernelRadius));

      ComputeIISFPressureByJacobi_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
          data->GetPressurePtr(),
          data->GetLastPressurePtr(),
          data->GetDensityErrorPtr(),
          data->GetLabelPtr(),
          data->GetAiiPtr(),
          data->GetDijPjPtr(),
          data->GetDiiPtr(),
          data->GetDensityAdvPtr(),
          data->GetPosPtr(),
          data->GetMassPtr(),
          data->GetDensityPtr(),
          rho0,
          dt,
          num,
          cellStart.Data(),
          boundaries->GetPosPtr(),
          boundaries->GetVolumePtr(),
          boundaryCellStart.Data(),
          gridSize,
          ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
          ThrustHelper::GridXYZ2GridHash(gridSize),
          SpikyKernelGrad(kernelRadius));

      error = thrust::reduce(thrust::device_ptr<float>(data->GetDensityErrorPtr()), thrust::device_ptr<float>(data->GetDensityErrorPtr() + num));
      error /= num;

      auto eta = mMaxError * 0.01f * rho0;
      flag = flag && (error <= eta);

      //printf("iter=%d, error=%.6f \n", iter, error);

      iter++;
    }

    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();

    return iter;
  }

  void CudaIISphSFSolver::ComputePressureAcceleration(
      CudaSFParticlesPtr &particles,
      CudaBoundaryParticlesPtr &boundaries,
      const CudaArray<size_t> &cellStart,
      const CudaArray<size_t> &boundaryCellStart,
      const float rho0,
      const float3 lowestPoint,
      const float kernelRadius,
      const int3 gridSize)
  {
    auto data = std::dynamic_pointer_cast<CudaIISFParticles>(particles);
    ComputeIISFPressureAcceleration_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        data->GetPressureAccPtr(),
        data->GetLabelPtr(),
        data->GetPosPtr(),
        data->GetMassPtr(),
        data->GetDensityPtr(),
        data->GetPressurePtr(),
        rho0,
        data->Size(),
        cellStart.Data(),
        boundaries->GetPosPtr(),
        boundaries->GetVolumePtr(),
        boundaryCellStart.Data(),
        gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize),
        SpikyKernelGrad(kernelRadius));

    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaIISphSFSolver::ComputeIISFSandLMNB(
      CudaSFParticlesPtr &particles,
      CudaBoundaryParticlesPtr &boundaries,
      const CudaArray<size_t> &cellStart,
      const CudaArray<size_t> &boundaryCellStart,
      const float sandRadius,
      const float waterRadius,
      const float young,
      const float poisson,
      const float tanFrictionAngle,
      const float c0,
      const float csat,
      const float cmc,
      const float cmcp,
      const float cd,
      const float gravity,
      const float rho0,
      const float3 lowestPoint,
      const float3 highestPoint,
      const float kernelRadius,
      const int3 gridSize)
  {
    ComputeIISFSandLMNB_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        particles->GetLabelPtr(),
        particles->GetPosPtr(),
        particles->GetVelPtr(),
        particles->GetAccPtr(),
        particles->GetMassPtr(),
        particles->GetDensityPtr(),
        particles->GetVoidagePtr(),
        particles->GetSaturationPtr(),
        particles->GetAvgFlowVelPtr(),
        particles->GetAvgDragForcePtr(),
        particles->GetAvgAdhesionForcePtr(),
        sandRadius,
        waterRadius,
        young,
        poisson,
        tanFrictionAngle,
        cd,
        gravity,
        rho0,
        particles->Size(),
        lowestPoint,
        highestPoint,
        cellStart.Data(),
        boundaries->GetPosPtr(),
        boundaries->GetLabelPtr(),
        boundaryCellStart.Data(),
        gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize),
        QuadraticBezierCoeff(c0, cmc, cmcp, csat),
        Poly6Kernel(kernelRadius));
    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaIISphSFSolver::ComputeIISFSandLMWB(
      CudaSFParticlesPtr &particles,
      const CudaArray<size_t> &cellStart,
      const float3 lowestPoint,
      const float kernelRadius,
      const int3 gridSize)
  {
    ComputeIISFSandLMWB_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        particles->GetAccPtr(),
        particles->GetLabelPtr(),
        particles->GetPosPtr(),
        particles->GetMassPtr(),
        particles->GetDensityPtr(),
        particles->GetPressurePtr(),
        particles->GetVoidagePtr(),
        particles->Size(),
        cellStart.Data(),
        gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize),
        SpikyKernelGrad(kernelRadius));
    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaIISphSFSolver::ComputeIISFWaterLMNB(
      CudaSFParticlesPtr &particles,
      CudaBoundaryParticlesPtr &boundaries,
      const CudaArray<size_t> &cellStart,
      const CudaArray<size_t> &boundaryCellStart,
      const float rho0,
      const float nu,
      const float bnu,
      const float3 lowestPoint,
      const float kernelRadius,
      const int3 gridSize)
  {
    ComputeIISFWaterLMNB_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        particles->GetAccPtr(),
        particles->GetLabelPtr(),
        particles->GetPosPtr(),
        particles->GetVelPtr(),
        particles->GetMassPtr(),
        particles->GetDensityPtr(),
        particles->GetVoidagePtr(),
        particles->GetAvgDragForcePtr(),
        particles->GetAdhesionForcePtr(),
        rho0,
        nu,
        bnu,
        particles->Size(),
        cellStart.Data(),
        boundaries->GetPosPtr(),
        boundaries->GetVolumePtr(),
        boundaryCellStart.Data(),
        gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize),
        Poly6Kernel(kernelRadius),
        SpikyKernelGrad(kernelRadius));
    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaIISphSFSolver::ComputeIISFWaterLMWB(
      CudaSFParticlesPtr &particles,
      const CudaArray<size_t> &cellStart,
      const float3 lowestPoint,
      const float kernelRadius,
      const int3 gridSize)
  {
    auto data = std::dynamic_pointer_cast<CudaIISFParticles>(particles);
    ComputeIISFWaterLMWB_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        data->GetPressureAccPtr(),
        data->GetLabelPtr(),
        data->GetPosPtr(),
        data->GetMassPtr(),
        data->GetDensityPtr(),
        data->GetPressurePtr(),
        data->GetVoidagePtr(),
        data->Size(),
        cellStart.Data(),
        gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize),
        SpikyKernelGrad(kernelRadius));
    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaIISphSFSolver::Advect(
      CudaSFParticlesPtr &particles,
      CudaBoundaryParticlesPtr &boundaries,
      const CudaArray<size_t> &boundaryCellStart,
      const float waterRadius,
      const float dt,
      const float damping,
      const float3 lowestPoint,
      const float3 highestPoint,
      const float kernelRadius,
      const int3 gridSize)
  {
    auto data = std::dynamic_pointer_cast<CudaIISFParticles>(particles);
    size_t num = data->Size();
    data->Advect(dt, damping);

    SFWaterBoundaryConstrain_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        data->GetLabelPtr(),
        data->GetPosPtr(),
        data->GetVelPtr(),
        num,
        lowestPoint,
        highestPoint,
        waterRadius,
        boundaries->GetPosPtr(),
        boundaries->GetLabelPtr(),
        boundaryCellStart.Data(),
        gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize));

    thrust::fill(thrust::device, data->GetDensityPtr(), data->GetDensityPtr() + num, 0.f);
    thrust::fill(thrust::device, data->GetAccPtr(), data->GetAccPtr() + num, make_float3(0.f));
    thrust::fill(thrust::device, data->GetAdhesionForcePtr(), data->GetAdhesionForcePtr() + num, make_float3(0.f));
    thrust::fill(thrust::device, data->GetAvgAdhesionForcePtr(), data->GetAvgAdhesionForcePtr() + num, make_float3(0.f));

    // iisph
    thrust::fill(thrust::device, data->GetAiiPtr(), data->GetAiiPtr() + num, 0.f);
    thrust::fill(thrust::device, data->GetDiiPtr(), data->GetDiiPtr() + num, make_float3(0.f));
    thrust::fill(thrust::device, data->GetDijPjPtr(), data->GetDijPjPtr() + num, make_float3(0.f));
    thrust::fill(thrust::device, data->GetDensityAdvPtr(), data->GetDensityAdvPtr() + num, 0.f);
    thrust::fill(thrust::device, data->GetDensityErrorPtr(), data->GetDensityErrorPtr() + num, 0.f);
    thrust::fill(thrust::device, data->GetPressureAccPtr(), data->GetPressureAccPtr() + num, make_float3(0.f));

    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

} // namespace KIRI
