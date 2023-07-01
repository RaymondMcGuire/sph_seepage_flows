/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-03-22 15:40:25
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-04-08 11:54:36
 * @FilePath:
 * \sph_seepage_flows\seepage_flow_cuda\src\kiri_pbs_cuda\solver\seepage_flow\cuda_dfsph_sf_solver.cpp
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */
#include <kiri_pbs_cuda/solver/seepageflow/cuda_dfsph_sf_solver.cuh>

namespace KIRI {

void CudaDFSphSFSolver::UpdateSolver(CudaSFParticlesPtr &particles,
                                     CudaBoundaryParticlesPtr &boundaries,
                                     const CudaArray<size_t> &cellStart,
                                     const CudaArray<size_t> &boundaryCellStart,
                                     float renderInterval,
                                     CudaSeepageflowParams params,
                                     CudaBoundaryParams bparams) {

  ExtraForces(particles, params.gravity);

  ComputeDensity(particles, boundaries, params.sph_density, params.dem_density,
                 cellStart, boundaryCellStart, bparams.lowest_point,
                 bparams.kernel_radius, bparams.grid_size);
  ComputePressure(particles, params.sph_density, params.sph_stiff);

  auto data = std::dynamic_pointer_cast<CudaDFSFParticles>(particles);

  ComputeDFSPHAlpha(data, boundaries, params.sph_density, cellStart,
                    boundaryCellStart, bparams.lowest_point,
                    bparams.kernel_radius, bparams.grid_size);

  ApplyDivergenceSolver(data, boundaries, params.sph_density, cellStart,
                        boundaryCellStart, bparams.lowest_point,
                        bparams.kernel_radius, bparams.grid_size);

  // ComputeDFPressure(data,params.sph_density);
  ComputeAvgFlowVelocity(particles, cellStart, bparams.lowest_point,
                         bparams.kernel_radius, bparams.grid_size);

  ComputeSFSandVoidage(particles, cellStart, bparams.lowest_point,
                       bparams.kernel_radius, bparams.grid_size);

  if (params.sf_type == SF) {
    ComputeSFWaterAdhesion(particles, cellStart, params.sf_a0, params.sf_asat,
                           params.sf_amc, params.sf_amc_p, bparams.lowest_point,
                           bparams.kernel_radius, bparams.grid_size);

    ComputeSFSandLinearMomentum(
        particles, boundaries, cellStart, boundaryCellStart,
        params.boundary_particle_radius, params.max_force_factor,
        params.dem_young, params.dem_poisson, params.dem_tan_friction_angle,
        params.sf_c0, params.sf_csat, params.sf_cmc, params.sf_cmc_p,
        params.sf_cd, abs(params.gravity.y), params.sph_density,
        bparams.lowest_point, bparams.highest_point, bparams.kernel_radius,
        bparams.grid_size);
  } else if (params.sf_type == MULTI_SF) {
    ComputeMultiSFWaterAdhesion(particles, cellStart, bparams.lowest_point,
                                bparams.kernel_radius, bparams.grid_size);

    ComputeMultiSFSandLinearMomentum(
        particles, boundaries, cellStart, boundaryCellStart,
        params.boundary_particle_radius, params.max_force_factor,
        params.dem_young, params.dem_poisson, params.dem_tan_friction_angle,
        params.sf_c0, params.sf_csat, params.sf_cmc, params.sf_cmc_p,
        abs(params.gravity.y), params.sph_density, bparams.lowest_point,
        bparams.highest_point, bparams.kernel_radius, bparams.grid_size);
  }

  ComputeSFWaterLinearMomentum(
      particles, boundaries, cellStart, boundaryCellStart, params.sph_density,
      params.sph_nu, params.sph_bnu, bparams.lowest_point,
      bparams.kernel_radius, bparams.grid_size);

  AdvectDFSPHVelocity(data);

  ApplyPressureSolver(data, boundaries, params.sph_density, cellStart,
                      boundaryCellStart, bparams.lowest_point,
                      bparams.kernel_radius, bparams.grid_size);

  ComputeSFWetSandColor(particles, params.sf_dry_sand_color,
                        params.sf_wet_sand_color);

  ComputeTimeStepsByCFL(data, params.sph_particle_radius, params.dt,
                        renderInterval);

  Advect(particles, boundaries, boundaryCellStart, params.sph_particle_radius,
         mDt, params.dem_damping, bparams.lowest_point, bparams.highest_point,
         bparams.kernel_radius, bparams.grid_size);
}

} // namespace KIRI