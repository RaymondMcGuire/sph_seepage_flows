/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-03-25 22:02:18
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-04-08 12:05:09
 * @FilePath:
 * \sph_seepage_flows\seepage_flow_cuda\src\kiri_pbs_cuda\system\cuda_sf_system.cu
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */
#include <kiri_pbs_cuda/system/cuda_base_system_gpu.cuh>
#include <kiri_pbs_cuda/system/cuda_sf_system.cuh>
#include <kiri_pbs_cuda/system/cuda_sph_system_gpu.cuh>
#include <kiri_pbs_cuda/thrust_helper/helper_thrust.cuh>
namespace KIRI {

CudaSFSystem::CudaSFSystem(CudaSFParticlesPtr &particles,
                           CudaBoundaryParticlesPtr &boundaryParticles,
                           CudaSphSFSolverPtr &solver,
                           CudaGNSearcherPtr &searcher,
                           CudaGNBoundarySearcherPtr &boundarySearcher,
                           CudaEmitterPtr &emitter,
                           const bool adaptiveSubTimeStep)
    : CudaBaseSystem(
          boundaryParticles, std::static_pointer_cast<CudaBaseSolver>(solver),
          boundarySearcher, particles->MaxSize(), adaptiveSubTimeStep),
      mParticles(std::move(particles)), mSearcher(std::move(searcher)),
      mEmitter(std::move(emitter)), mEmitterElapsedTime(0.f),
      mNextEmitTime(0.f),
      mCudaGridSize(CuCeilDiv(particles->MaxSize(), KIRI_CUBLOCKSIZE)) {

  if (CUDA_SPH_EMITTER_PARAMS.enable) {
    switch (CUDA_SPH_EMITTER_PARAMS.emit_type) {
    case CudaSphEmitterType::SQUARE:
      mEmitter->BuildSquareEmitter(CUDA_SEEPAGEFLOW_PARAMS.sph_particle_radius,
                                   CUDA_SPH_EMITTER_PARAMS.emit_radius);
      break;
    case CudaSphEmitterType::CIRCLE:
      mEmitter->BuildCircleEmitter(CUDA_SEEPAGEFLOW_PARAMS.sph_particle_radius,
                                   CUDA_SPH_EMITTER_PARAMS.emit_radius);
      break;
    case CudaSphEmitterType::RECTANGLE:
      mEmitter->BuildRectangleEmitter(
          CUDA_SEEPAGEFLOW_PARAMS.sph_particle_radius,
          CUDA_SPH_EMITTER_PARAMS.emit_width,
          CUDA_SPH_EMITTER_PARAMS.emit_height);
      break;
    }
  }

  // compute boundary volume(Akinci2012)
  ComputeBoundaryVolume();
}

void CudaSFSystem::OnUpdateSolver(float renderInterval) {
  mSearcher->BuildGNSearcher(mParticles);

  CudaSphSFSolverPtr solver;
  if (CUDA_SEEPAGEFLOW_PARAMS.solver_type == SPH_SOLVER)
    solver = std::dynamic_pointer_cast<CudaSphSFSolver>(mSolver);
  else if (CUDA_SEEPAGEFLOW_PARAMS.solver_type == WCSPH_SOLVER)
    solver = std::dynamic_pointer_cast<CudaWCSphSFSolver>(mSolver);
  else if (CUDA_SEEPAGEFLOW_PARAMS.solver_type == DFSPH_SOLVER)
    solver = std::dynamic_pointer_cast<CudaDFSphSFSolver>(mSolver);

  solver->UpdateSolver(mParticles, mBoundaries, mSearcher->GetCellStart(),
                       mBoundarySearcher->GetCellStart(), renderInterval,
                       CUDA_SEEPAGEFLOW_PARAMS, CUDA_BOUNDARY_PARAMS);

  // emitter
  if (mEmitter->GetEmitterStatus() && CUDA_SPH_EMITTER_PARAMS.run) {

    mEmitterElapsedTime += this->GetCurrentTimeStep();
    if (mEmitterElapsedTime > mNextEmitTime) {
      auto p = mEmitter->Emit();
      if (mParticles->Size() + p.size() < mParticles->MaxSize())
        mParticles->AddSphParticles(
            p, CUDA_SPH_EMITTER_PARAMS.emit_col, mEmitter->GetEmitterVelocity(),
            CUDA_SEEPAGEFLOW_PARAMS.sph_mass,
            CUDA_SEEPAGEFLOW_PARAMS.sph_particle_radius);
      else
        mEmitter->SetEmitterStatus(false);

      mNextEmitTime += 2.f * CUDA_SEEPAGEFLOW_PARAMS.sph_particle_radius /
                       length(mEmitter->GetEmitterVelocity());
    }
  }

  cudaDeviceSynchronize();
  KIRI_CUKERNAL();
}

void CudaSFSystem::ComputeBoundaryVolume() {
  auto CUDA_BOUNDARY_GRIDSIZE =
      CuCeilDiv(mBoundaries->Size(), KIRI_CUBLOCKSIZE);

  ComputeBoundaryVolume_CUDA<<<CUDA_BOUNDARY_GRIDSIZE, KIRI_CUBLOCKSIZE>>>(
      mBoundaries->GetPosPtr(), mBoundaries->GetVolumePtr(),
      mBoundaries->Size(), mBoundarySearcher->GetCellStartPtr(),
      mBoundarySearcher->GetGridSize(),
      ThrustHelper::Pos2GridXYZ<float3>(mBoundarySearcher->GetLowestPoint(),
                                        mBoundarySearcher->GetCellSize(),
                                        mBoundarySearcher->GetGridSize()),
      ThrustHelper::GridXYZ2GridHash(mBoundarySearcher->GetGridSize()),
      Poly6Kernel(mBoundarySearcher->GetCellSize()));
  cudaDeviceSynchronize();
  KIRI_CUKERNAL();
}
} // namespace KIRI
