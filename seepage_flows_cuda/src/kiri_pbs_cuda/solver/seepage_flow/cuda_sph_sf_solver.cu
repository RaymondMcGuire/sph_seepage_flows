/*** 
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-02-28 22:45:47
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-02-28 23:00:33
 * @FilePath: \sph_seepage_flows\seepage_flows_cuda\src\kiri_pbs_cuda\solver\seepage_flow\cuda_sph_sf_solver.cu
 * @Description: 
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved. 
 */

#include <kiri_pbs_cuda/solver/seepageflow/cuda_sph_sf_solver.cuh>

#include <kiri_pbs_cuda/thrust_helper/helper_thrust.cuh>
#include <kiri_pbs_cuda/solver/cuda_solver_common_gpu.cuh>
#include <kiri_pbs_cuda/solver/seepageflow/cuda_sf_utils.cuh>
#include <kiri_pbs_cuda/solver/seepageflow/cuda_sph_sf_solver_gpu.cuh>

namespace KIRI
{
  void CudaSphSFSolver::ComputeDensity(
      CudaSFParticlesPtr &particles,
      CudaBoundaryParticlesPtr &boundaries,
      const float rho0,
      const float rho1,
      const CudaArray<size_t> &cellStart,
      const CudaArray<size_t> &boundaryCellStart,
      const float3 lowestPoint,
      const float kernelRadius,
      const int3 gridSize)
  {
    ComputeSFDensity_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        particles->GetLabelPtr(),
        particles->GetPosPtr(),
        particles->GetMassPtr(),
        particles->GetDensityPtr(),
        rho0,
        rho1,
        particles->Size(),
        cellStart.Data(),
        boundaries->GetPosPtr(),
        boundaries->GetVolumePtr(),
        boundaryCellStart.Data(),
        gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize),
        Poly6Kernel(kernelRadius));
    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaSphSFSolver::ComputePressure(
      CudaSFParticlesPtr &particles,
      const float rho0,
      const float stiff)
  {
    ComputeSFPressure_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        particles->GetLabelPtr(),
        particles->GetDensityPtr(),
        particles->GetPressurePtr(),
        particles->Size(),
        rho0,
        stiff,
        1.f);
    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaSphSFSolver::ComputeAvgFlowVelocity(
      CudaSFParticlesPtr &particles,
      const CudaArray<size_t> &cellStart,
      const float3 lowestPoint,
      const float kernelRadius,
      const int3 gridSize)
  {

    ComputeSFAvgFlow_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        particles->GetLabelPtr(),
        particles->GetPosPtr(),
        particles->GetVelPtr(),
        particles->GetMassPtr(),
        particles->GetDensityPtr(),
        particles->GetAvgFlowVelPtr(),
        particles->GetVoidagePtr(),
        particles->GetSaturationPtr(),
        particles->Size(),
        cellStart.Data(),
        gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize),
        Poly6Kernel(kernelRadius));
    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaSphSFSolver::ComputeSFWaterAdhesion(
      CudaSFParticlesPtr &particles,
      const CudaArray<size_t> &cellStart,
      const float a0,
      const float asat,
      const float amc,
      const float amcp,
      const float3 lowestPoint,
      const float kernelRadius,
      const int3 gridSize)
  {
    ComputeSFWaterAdhesionForces_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        particles->GetLabelPtr(),
        particles->GetPosPtr(),
        particles->GetMassPtr(),
        particles->GetDensityPtr(),
        particles->GetSaturationPtr(),
        particles->GetAdhesionForcePtr(),
        particles->GetAvgAdhesionForcePtr(),
        particles->Size(),
        cellStart.Data(),
        gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize),
        QuadraticBezierCoeff(a0, amc, amcp, asat),
        Poly6Kernel(kernelRadius),
        AdhesionAkinci13(kernelRadius));
    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaSphSFSolver::ComputeMultiSFWaterAdhesion(
      CudaSFParticlesPtr &particles,
      const CudaArray<size_t> &cellStart,
      const float3 lowestPoint,
      const float kernelRadius,
      const int3 gridSize)
  {
    ComputeMultiSFWaterAdhesionForces_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        particles->GetLabelPtr(),
        particles->GetPosPtr(),
        particles->GetMassPtr(),
        particles->GetDensityPtr(),
        particles->GetSaturationPtr(),
        particles->GetAdhesionForcePtr(),
        particles->GetAvgAdhesionForcePtr(),
        particles->GetCdA0AsatPtr(),
        particles->GetAmcAmcpPtr(),
        particles->Size(),
        cellStart.Data(),
        gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize),
        Poly6Kernel(kernelRadius),
        AdhesionAkinci13(kernelRadius));
    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaSphSFSolver::ComputeSFWetSandColor(
      CudaSFParticlesPtr &particles,
      const float3 dryColor,
      const float3 wetColor)
  {

    ComputeSFWetSandColor_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        particles->GetLabelPtr(),
        particles->GetSaturationPtr(),
        particles->GetMaxSaturationPtr(),
        particles->GetColPtr(),
        particles->Size(),
        dryColor,
        wetColor);
    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaSphSFSolver::ComputeSFSandVoidage(
      CudaSFParticlesPtr &particles,
      const CudaArray<size_t> &cellStart,
      const float3 lowestPoint,
      const float kernelRadius,
      const int3 gridSize)
  {

    ComputeSFSandVoidage_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        particles->GetLabelPtr(),
        particles->GetPosPtr(),
        particles->GetMassPtr(),
        particles->GetDensityPtr(),
        particles->GetVoidagePtr(),
        particles->Size(),
        cellStart.Data(),
        gridSize,
        ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
        ThrustHelper::GridXYZ2GridHash(gridSize),
        Poly6Kernel(kernelRadius));
    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaSphSFSolver::ComputeSFSandLinearMomentum(
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
    ComputeSFSandLinearMomentum_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        particles->GetLabelPtr(),
        particles->GetPosPtr(),
        particles->GetVelPtr(),
        particles->GetAccPtr(),
        particles->GetMassPtr(),
        particles->GetDensityPtr(),
        particles->GetPressurePtr(),
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
        Poly6Kernel(kernelRadius),
        SpikyKernelGrad(kernelRadius));
    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaSphSFSolver::ComputeMultiSFSandLinearMomentum(
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
      const float gravity,
      const float rho0,
      const float3 lowestPoint,
      const float3 highestPoint,
      const float kernelRadius,
      const int3 gridSize)
  {
    ComputeMultiSFSandLinearMomentum_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        particles->GetLabelPtr(),
        particles->GetPosPtr(),
        particles->GetVelPtr(),
        particles->GetAccPtr(),
        particles->GetMassPtr(),
        particles->GetDensityPtr(),
        particles->GetPressurePtr(),
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
        particles->GetCdA0AsatPtr(),
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
        Poly6Kernel(kernelRadius),
        SpikyKernelGrad(kernelRadius));
    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaSphSFSolver::ComputeSFWaterLinearMomentum(
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
    ComputeSFWaterLinearMomentum_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        particles->GetLabelPtr(),
        particles->GetPosPtr(),
        particles->GetVelPtr(),
        particles->GetAccPtr(),
        particles->GetMassPtr(),
        particles->GetDensityPtr(),
        particles->GetPressurePtr(),
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

  void CudaSphSFSolver::Advect(
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
    size_t num = particles->Size();
    particles->Advect(dt, damping);
    SFWaterBoundaryConstrain_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
        particles->GetLabelPtr(),
        particles->GetPosPtr(),
        particles->GetVelPtr(),
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

    thrust::fill(thrust::device, particles->GetAdhesionForcePtr(), particles->GetAdhesionForcePtr() + num, make_float3(0.f));
    thrust::fill(thrust::device, particles->GetAvgAdhesionForcePtr(), particles->GetAvgAdhesionForcePtr() + num, make_float3(0.f));
    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

  void CudaSphSFSolver::ExtraForces(
      CudaSFParticlesPtr &particles,
      const float3 gravity)
  {
    thrust::fill(thrust::device, particles->GetAccPtr(), particles->GetAccPtr() + particles->Size(), make_float3(0.f));
    thrust::transform(thrust::device,
                      particles->GetAccPtr(), particles->GetAccPtr() + particles->Size(),
                      particles->GetAccPtr(),
                      ThrustHelper::Plus<float3>(gravity));

    KIRI_CUCALL(cudaDeviceSynchronize());
    KIRI_CUKERNAL();
  }

} // namespace KIRI
