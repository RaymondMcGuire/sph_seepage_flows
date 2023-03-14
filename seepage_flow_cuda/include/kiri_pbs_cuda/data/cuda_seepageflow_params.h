/***
 * @Author: Xu.WANG
 * @Date: 2021-02-10 15:29:35
 * @LastEditTime: 2021-08-21 18:14:53
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath: \sph_seepage_flows\seepage_flows_cuda\include\kiri_pbs_cuda\data\cuda_seepageflow_params.h
 */

#ifndef _CUDA_SEEPAGEFLOW_PARAMS_H_
#define _CUDA_SEEPAGEFLOW_PARAMS_H_

#pragma once

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>
#include <kiri_pbs_cuda/data/cuda_sph_params.h>
namespace KIRI
{
    enum CudaSeepageflowType
    {
        SF,
        MULTI_SF
    };

    struct CudaSeepageflowParams
    {
        // water(SPH)
        float sph_mass;
        float sph_density;
        float sph_particle_radius;
        float sph_kernel_radius;

        float sph_stiff;
        float sph_visc;
        float sph_nu;
        float sph_bnu;

        // sand(DEM)
        float dem_mass;
        float dem_density;
        float dem_particle_radius;
        float dem_kernel_radius;
        float dem_young;
        float dem_poisson;
        float dem_tan_friction_angle;
        float dem_damping;

        // seepage flow
        float sf_c0;
        float sf_csat;
        float sf_cmc;
        float sf_cmc_p;
        float sf_cd;

        float sf_a0;
        float sf_asat;
        float sf_amc;
        float sf_amc_p;

        float3 sf_dry_sand_color;
        float3 sf_wet_sand_color;

        //
        float3 gravity;
        float dt;

        CudaSphSolverType solver_type;
        CudaSeepageflowType sf_type;
    };

    struct CudaSeepageflowAppParams
    {
        size_t max_num = 100000;

        bool run = false;
        bool run_offline = false;

        int scene_data_idx = 0;
        char bgeo_export_folder[320] = "default";
        bool bgeo_export = false;
    };

    extern CudaSeepageflowParams CUDA_SEEPAGEFLOW_PARAMS;
    extern CudaSeepageflowAppParams CUDA_SEEPAGEFLOW_APP_PARAMS;

} // namespace KIRI

#endif