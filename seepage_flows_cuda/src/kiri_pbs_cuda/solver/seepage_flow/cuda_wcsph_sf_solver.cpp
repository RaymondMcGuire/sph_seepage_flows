/*** 
 * @Author: Xu.WANG
 * @Date: 2021-02-03 16:35:31
 * @LastEditTime: 2021-08-18 21:31:15
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\solver\seepage_flow\cuda_wcsph_sf_solver.cpp
 */

#include <kiri_pbs_cuda/solver/seepageflow/cuda_wcsph_sf_solver.cuh>

namespace KIRI
{
    float CudaWCSphSFSolver::GetSpeedOfSound() const
    {
        return mSpeedOfSound;
    }

    void CudaWCSphSFSolver::SetSpeedOfSound(float newSpeedOfSound)
    {
        mSpeedOfSound = std::max(newSpeedOfSound, KIRI_EPSILON);
    }

    float CudaWCSphSFSolver::GetTimeStepLimitScale() const
    {
        return mTimeStepLimitScale;
    }

    void CudaWCSphSFSolver::SetTimeStepLimitScale(float newScale)
    {
        mTimeStepLimitScale = std::max(newScale, 0.f);
    }

    void CudaWCSphSFSolver::UpdateSolver(
        CudaSFParticlesPtr &particles,
        CudaBoundaryParticlesPtr &boundaries,
        const CudaArray<size_t> &cellStart,
        const CudaArray<size_t> &boundaryCellStart,
        float renderInterval,
        CudaSeepageflowParams params,
        CudaBoundaryParams bparams)
    {

        ExtraForces(
            particles,
            params.gravity);

        ComputeDensity(
            particles,
            boundaries,
            params.sph_density,
            params.dem_density,
            cellStart,
            boundaryCellStart,
            bparams.lowest_point,
            bparams.kernel_radius,
            bparams.grid_size);

        ComputePressure(
            particles,
            params.sph_density,
            params.sph_stiff);

        ComputeAvgFlowVelocity(
            particles,
            cellStart,
            bparams.lowest_point,
            bparams.kernel_radius,
            bparams.grid_size);

        ComputeSFSandVoidage(
            particles,
            cellStart,
            bparams.lowest_point,
            bparams.kernel_radius,
            bparams.grid_size);

      
            ComputeSFWaterAdhesion(
                particles,
                cellStart,
                params.sf_a0,
                params.sf_asat,
                params.sf_amc,
                params.sf_amc_p,
                bparams.lowest_point,
                bparams.kernel_radius,
                bparams.grid_size);

            ComputeSFSandLinearMomentum(
                particles,
                boundaries,
                cellStart,
                boundaryCellStart,
                params.dem_particle_radius,
                params.sph_particle_radius,
                params.dem_young,
                params.dem_poisson,
                params.dem_tan_friction_angle,
                params.sf_c0,
                params.sf_csat,
                params.sf_cmc,
                params.sf_cmc_p,
                params.sf_cd,
                abs(params.gravity.y),
                params.sph_density,
                bparams.lowest_point,
                bparams.highest_point,
                bparams.kernel_radius,
                bparams.grid_size);
       
        ComputeSFWaterLinearMomentum(
            particles,
            boundaries,
            cellStart,
            boundaryCellStart,
            params.sph_density,
            params.sph_nu,
            params.sph_bnu,
            bparams.lowest_point,
            bparams.kernel_radius,
            bparams.grid_size);

        ComputeSFWetSandColor(
            particles,
            params.sf_dry_sand_color,
            params.sf_wet_sand_color);

        ComputeSubTimeStepsByCFL(
            particles,
            params.sph_mass,
            params.dt,
            bparams.kernel_radius,
            renderInterval);

        Advect(particles,
               boundaries,
               boundaryCellStart,
               params.sph_particle_radius,
               renderInterval / static_cast<float>(mNumOfSubTimeSteps),
               params.dem_damping,
               bparams.lowest_point,
               bparams.highest_point,
               bparams.kernel_radius,
               bparams.grid_size);
    }

} // namespace KIRI