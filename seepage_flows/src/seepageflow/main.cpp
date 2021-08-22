/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-27 00:49:33
 * @LastEditTime: 2021-08-22 14:32:50
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \sph_seepage_flows\seepage_flows\src\seepageflow\main.cpp
 */

#include <sf_cuda_define.h>
#include <kiri_utils.h>

using namespace KIRI;

// global params
const UInt RunLiquidNumber = 0;
const UInt TotalFrameNumber = 300;
UInt SimCount = 0;
float TotalFrameTime = 0.f;
float RenderInterval = 1.f / 30.f;

KiriTimer PerFrameTimer;
CudaSFSystemPtr SFSystem;

void SetupParams()
{
    KIRI_LOG_DEBUG("Seepageflow: SetupParams");

    strcpy(CUDA_SEEPAGEFLOW_APP_PARAMS.bgeo_export_folder, (String(EXPORT_PATH) + "bgeo/seepageflow_bunny_wcsph").c_str());

    // scene config
    auto cuda_lowest_point = make_float3(0.f);
    auto cuda_highest_point = make_float3(1.5f, 1.5f, 2.5f);
    auto cuda_world_size = cuda_highest_point - cuda_lowest_point;
    auto cuda_world_center = (cuda_highest_point + cuda_lowest_point) / 2.f;
    CUDA_SEEPAGEFLOW_APP_PARAMS.max_num = 500000;

    // sph params
    CUDA_SEEPAGEFLOW_PARAMS.sph_density = 1000.f;
    CUDA_SEEPAGEFLOW_PARAMS.sph_particle_radius = 0.01f;
    CUDA_SEEPAGEFLOW_PARAMS.sph_kernel_radius = 4.f * CUDA_SEEPAGEFLOW_PARAMS.sph_particle_radius;

    auto diam = 2.f * CUDA_SEEPAGEFLOW_PARAMS.sph_particle_radius;
    CUDA_SEEPAGEFLOW_PARAMS.sph_mass = 0.8f * diam * diam * diam * CUDA_SEEPAGEFLOW_PARAMS.sph_density;

    auto visc = 0.05f;
    auto sound_speed = 100.f;
    auto nu = (visc + visc) * CUDA_SEEPAGEFLOW_PARAMS.sph_kernel_radius * sound_speed;
    auto boundary_friction = 0.2f;
    auto bnu = boundary_friction * CUDA_SEEPAGEFLOW_PARAMS.sph_kernel_radius * sound_speed;

    CUDA_SEEPAGEFLOW_PARAMS.sph_stiff = 0.001f * sound_speed * sound_speed * CUDA_SEEPAGEFLOW_PARAMS.sph_density / 7.f;
    CUDA_SEEPAGEFLOW_PARAMS.sph_visc = visc;
    CUDA_SEEPAGEFLOW_PARAMS.sph_nu = nu;
    CUDA_SEEPAGEFLOW_PARAMS.sph_bnu = bnu;

    // dem params
    CUDA_SEEPAGEFLOW_PARAMS.dem_density = 2700.f;
    CUDA_SEEPAGEFLOW_PARAMS.dem_particle_radius = 0.005f;
    CUDA_SEEPAGEFLOW_PARAMS.dem_mass = CUDA_SEEPAGEFLOW_PARAMS.dem_density * ((4.f / 3.f) * 3.1415926f * std::powf(CUDA_SEEPAGEFLOW_PARAMS.dem_particle_radius, 3.f));

    CUDA_SEEPAGEFLOW_PARAMS.dem_young = 1e5f;
    CUDA_SEEPAGEFLOW_PARAMS.dem_poisson = 0.3f;
    CUDA_SEEPAGEFLOW_PARAMS.dem_tan_friction_angle = 0.5f;
    CUDA_SEEPAGEFLOW_PARAMS.dem_damping = 0.4f;

    CUDA_SEEPAGEFLOW_PARAMS.sf_c0 = 0.7f;
    CUDA_SEEPAGEFLOW_PARAMS.sf_cd = 0.5f;
    CUDA_SEEPAGEFLOW_PARAMS.sf_csat = 0.f;
    CUDA_SEEPAGEFLOW_PARAMS.sf_cmc = 1.f;
    CUDA_SEEPAGEFLOW_PARAMS.sf_cmc_p = 0.01f;

    CUDA_SEEPAGEFLOW_PARAMS.sf_a0 = 0.f;
    CUDA_SEEPAGEFLOW_PARAMS.sf_asat = 0.8f;
    CUDA_SEEPAGEFLOW_PARAMS.sf_amc = 1.5f;
    CUDA_SEEPAGEFLOW_PARAMS.sf_amc_p = 0.5f;

    CUDA_SEEPAGEFLOW_PARAMS.sf_dry_sand_color = make_float3(0.88f, 0.79552f, 0.5984f);
    CUDA_SEEPAGEFLOW_PARAMS.sf_wet_sand_color = make_float3(0.38f, 0.29f, 0.14f);

    CUDA_SEEPAGEFLOW_PARAMS.dt = 0.5f * CUDA_SEEPAGEFLOW_PARAMS.dem_particle_radius / std::sqrtf(CUDA_SEEPAGEFLOW_PARAMS.dem_young / CUDA_SEEPAGEFLOW_PARAMS.dem_density);
    CUDA_SEEPAGEFLOW_PARAMS.gravity = make_float3(0.0f, -9.8f, 0.0f);

    // sph emitter
    CUDA_SPH_EMITTER_PARAMS.enable = true;
    CUDA_SPH_EMITTER_PARAMS.run = false;
    CUDA_SPH_EMITTER_PARAMS.emit_pos = make_float3(cuda_world_center.x + cuda_world_size.x / 10.f, cuda_world_center.y + cuda_world_size.y / 5.f, cuda_world_center.z + cuda_world_size.z / 2.3f);
    CUDA_SPH_EMITTER_PARAMS.emit_vel = make_float3(0.f, 0.f, -5.f);
    CUDA_SPH_EMITTER_PARAMS.emit_col = make_float3(127.f, 205.f, 255.f) / 255.f;

    CUDA_SPH_EMITTER_PARAMS.emit_radius = 0.17f;
    CUDA_SPH_EMITTER_PARAMS.emit_width = 0.22f;
    CUDA_SPH_EMITTER_PARAMS.emit_height = 0.18f;
    CUDA_SPH_EMITTER_PARAMS.emit_type = CudaSphEmitterType::CIRCLE;

    // scene data
    CUDA_BOUNDARY_PARAMS.lowest_point = cuda_lowest_point;
    CUDA_BOUNDARY_PARAMS.highest_point = cuda_highest_point;
    CUDA_BOUNDARY_PARAMS.world_size = cuda_world_size;
    CUDA_BOUNDARY_PARAMS.world_center = cuda_world_center;

    CUDA_BOUNDARY_PARAMS.kernel_radius = CUDA_SEEPAGEFLOW_PARAMS.sph_kernel_radius;
    CUDA_BOUNDARY_PARAMS.grid_size = make_int3((CUDA_BOUNDARY_PARAMS.highest_point - CUDA_BOUNDARY_PARAMS.lowest_point) / CUDA_BOUNDARY_PARAMS.kernel_radius);

    // init emitter
    CudaEmitterPtr emitter = std::make_shared<CudaEmitter>(
        CUDA_SPH_EMITTER_PARAMS.emit_pos,
        CUDA_SPH_EMITTER_PARAMS.emit_vel,
        CUDA_SPH_EMITTER_PARAMS.enable);

    // boundary sampling
    BoundaryData boundaryData;
    auto boundaryEmitter = std::make_shared<CudaBoundaryEmitter>();

    boundaryEmitter->BuildWorldBoundary(boundaryData, CUDA_BOUNDARY_PARAMS.lowest_point, CUDA_BOUNDARY_PARAMS.highest_point, CUDA_SEEPAGEFLOW_PARAMS.sph_particle_radius);

    // shape sampling
    SeepageflowVolumeData volumeData;
    auto volumeEmitter = std::make_shared<CudaVolumeEmitter>();

    Vec_String shape_folders, shape_files;
    shape_folders.emplace_back("bunny");
    shape_files.emplace_back("bunny");

    auto sandShapes = ReadMultiBgeoFilesForGPU(shape_folders, shape_files);

    volumeEmitter->BuildSeepageflowShapeVolume(
        volumeData,
        sandShapes,
        CUDA_SEEPAGEFLOW_PARAMS.sf_dry_sand_color,
        CUDA_SEEPAGEFLOW_PARAMS.dem_density,
        true,
        CUDA_BOUNDARY_PARAMS.lowest_point.y);

    // dt
    CUDA_SEEPAGEFLOW_PARAMS.dem_particle_radius = volumeData.sandMinRadius;
    CUDA_SEEPAGEFLOW_PARAMS.dt = 0.5f * volumeData.sandMinRadius / std::sqrtf(CUDA_SEEPAGEFLOW_PARAMS.dem_young / CUDA_SEEPAGEFLOW_PARAMS.dem_density);
    KIRI_LOG_INFO("Number of total particles = {0}, dt={1}", volumeData.pos.size(), CUDA_SEEPAGEFLOW_PARAMS.dt);

    // init spatial searcher
    CudaSFParticlesPtr particles;
    CudaGNSearcherPtr searcher;
    searcher = std::make_shared<CudaGNSearcher>(
        CUDA_BOUNDARY_PARAMS.lowest_point,
        CUDA_BOUNDARY_PARAMS.highest_point,
        particles->MaxSize(),
        CUDA_BOUNDARY_PARAMS.kernel_radius,
        SearcherParticleType::SEEPAGE);

    particles =
        std::make_shared<CudaSFParticles>(
            CUDA_SEEPAGEFLOW_APP_PARAMS.max_num,
            volumeData.pos,
            volumeData.col,
            volumeData.label,
            volumeData.mass,
            volumeData.radius);

    auto boundary_particles = std::make_shared<CudaBoundaryParticles>(boundaryData.pos, boundaryData.label);
    KIRI_LOG_INFO("Number of Boundary Particles = {0}", boundary_particles->Size());

    // wcsph
    bool adaptive_sub_timestep = true;
    CudaSphSFSolverPtr pSolver;
    pSolver = std::make_shared<CudaWCSphSFSolver>(
        particles->MaxSize());
    CUDA_SEEPAGEFLOW_PARAMS.solver_type = WCSPH_SOLVER;
    KIRI_LOG_INFO("Current Fluid Solver= WCSPH");

    // bgeo file export & render FPS
    CUDA_SEEPAGEFLOW_APP_PARAMS.bgeo_export = true;

    CudaGNBoundarySearcherPtr boundary_searcher = std::make_shared<CudaGNBoundarySearcher>(
        CUDA_BOUNDARY_PARAMS.lowest_point,
        CUDA_BOUNDARY_PARAMS.highest_point,
        boundary_particles->MaxSize(),
        CUDA_BOUNDARY_PARAMS.kernel_radius);

    SFSystem = std::make_shared<CudaSFSystem>(
        particles,
        boundary_particles,
        pSolver,
        searcher,
        boundary_searcher,
        emitter,
        adaptive_sub_timestep);
}

void Update()
{
    if (CUDA_SEEPAGEFLOW_APP_PARAMS.run && SimCount < TotalFrameNumber + RunLiquidNumber)
    {
        if (SimCount == RunLiquidNumber)
        {
            // export bgeo file
            CUDA_SEEPAGEFLOW_APP_PARAMS.bgeo_export = CUDA_SEEPAGEFLOW_APP_PARAMS.run;
            if (CUDA_SPH_EMITTER_PARAMS.enable)
                CUDA_SPH_EMITTER_PARAMS.run = CUDA_SEEPAGEFLOW_APP_PARAMS.run;
        }

        if (SFSystem->GetAdaptiveSubTimeStep())
        {
            float remaining_time = RenderInterval;
            KIRI_LOG_INFO("Simulation Frame={0}, Adaptive Sub-Simulation", ++SimCount);
            PerFrameTimer.Restart();
            size_t i = 0;
            while (remaining_time > KIRI_EPSILON)
            {
                KIRI_LOG_INFO("Current Sub-Simulation RemainTime={0},Sub-Simulation Step={1}", remaining_time, ++i);
                SFSystem->UpdateSystem(remaining_time);
                remaining_time -= remaining_time / static_cast<float>(SFSystem->GetNumOfSubTimeSteps());
            }
        }
        else
        {
            auto numOfSubTimeSteps = SFSystem->GetNumOfSubTimeSteps();
            KIRI_LOG_INFO("Simulation Frame={0}, Sub-Simulation Total Number={1}", ++SimCount, numOfSubTimeSteps);

            PerFrameTimer.Restart();
            for (size_t i = 0; i < numOfSubTimeSteps; i++)
            {
                KIRI_LOG_INFO("Current Sub-Simulation/ Total Number ={0}/{1}", i + 1, numOfSubTimeSteps);
                SFSystem->UpdateSystem(RenderInterval);
            }
        }

        KIRI_LOG_INFO("Time Per Frame={0}", PerFrameTimer.Elapsed());
        TotalFrameTime += PerFrameTimer.Elapsed();

        if (CUDA_SEEPAGEFLOW_APP_PARAMS.bgeo_export)
        {
            auto particles = SFSystem->GetSFParticles();
            ExportBgeoFileCUDA(
                CUDA_SEEPAGEFLOW_APP_PARAMS.bgeo_export_folder,
                UInt2Str4Digit(SimCount - RunLiquidNumber),
                particles->GetPosPtr(),
                particles->GetColPtr(),
                particles->GetRadiusPtr(),
                particles->GetLabelPtr(),
                particles->Size());
        }
    }
    else if (CUDA_SEEPAGEFLOW_APP_PARAMS.run)
    {
        CUDA_SEEPAGEFLOW_APP_PARAMS.run = false;

        KIRI_LOG_INFO("Average Per Frame={0}", TotalFrameTime / (TotalFrameNumber + RunLiquidNumber));
    }
}

void main()
{
    KiriLog::Init();

    SetupParams();

    CUDA_SEEPAGEFLOW_APP_PARAMS.run = true;

    while (CUDA_SEEPAGEFLOW_APP_PARAMS.run)
        Update();

    return;
}
