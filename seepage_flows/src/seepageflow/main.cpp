/*** 
 * @Author: Xu.WANG
 * @Date: 2020-10-27 00:49:33
 * @LastEditTime: 2021-08-19 00:31:30
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriExamples\src\seepageflow\seepageflow_offline.cpp
 */

#include <seepageflow/seepageflow_offline.h>
#include <imgui/include/imgui.h>

#include <kiri_pbs_cuda/solver/seepageflow/cuda_wcsph_sf_solver.cuh>
#include <kiri_pbs_cuda/solver/seepageflow/cuda_iisph_sf_solver.cuh>
#include <kiri_pbs_cuda/particle/cuda_iisf_particles.cuh>

#include <fbs/generated/cuda_seepageflow_app_generated.h>
#include <fbs/fbs_helper.h>

#include <kiri_pbs_cuda/emitter/cuda_volume_emitter.cuh>
#include <kiri_pbs_cuda/emitter/cuda_boundary_emitter.cuh>

#include <kiri_pbs_cuda/solver/seepageflow/cuda_sf_utils.cuh>

namespace KIRI
{
    void KiriSeepageFlowOffline::SetupPBSParams()
    {
        KIRI_LOG_DEBUG("Seepageflow Offline:SetupPBSParams");

        strcpy(CUDA_SEEPAGEFLOW_APP_PARAMS.bgeo_export_folder, (KiriUtils::GetDefaultExportPath() + "/bgeo/" + mName).c_str());

        auto scene_config_data = KIRI::FlatBuffers::GetCudaSeepageFlowApp(mSceneConfigData.data());

        // max number of particles
        CUDA_SEEPAGEFLOW_APP_PARAMS.max_num = scene_config_data->max_particles_num();

        // water params
        auto seepageflow_data = scene_config_data->seepage_flow_data();
        CUDA_SEEPAGEFLOW_PARAMS.sph_density = seepageflow_data->sph_density();
        CUDA_SEEPAGEFLOW_PARAMS.sph_mass = seepageflow_data->sph_mass();
        CUDA_SEEPAGEFLOW_PARAMS.sph_kernel_radius = seepageflow_data->sph_kernel_radius();
        CUDA_SEEPAGEFLOW_PARAMS.sph_particle_radius = seepageflow_data->sph_particle_radius();

        CUDA_SEEPAGEFLOW_PARAMS.sph_stiff = seepageflow_data->sph_stiff();
        CUDA_SEEPAGEFLOW_PARAMS.sph_visc = seepageflow_data->sph_visc();
        CUDA_SEEPAGEFLOW_PARAMS.sph_nu = seepageflow_data->sph_nu();
        CUDA_SEEPAGEFLOW_PARAMS.sph_bnu = seepageflow_data->sph_bnu();

        // sand params
        CUDA_SEEPAGEFLOW_PARAMS.dem_mass = seepageflow_data->dem_mass();
        CUDA_SEEPAGEFLOW_PARAMS.dem_density = seepageflow_data->dem_density();
        CUDA_SEEPAGEFLOW_PARAMS.dem_particle_radius = seepageflow_data->dem_particle_radius();

        CUDA_SEEPAGEFLOW_PARAMS.dem_young = seepageflow_data->dem_young();
        CUDA_SEEPAGEFLOW_PARAMS.dem_poisson = seepageflow_data->dem_poisson();
        CUDA_SEEPAGEFLOW_PARAMS.dem_tan_friction_angle = std::tanf(seepageflow_data->dem_friction_angle());
        CUDA_SEEPAGEFLOW_PARAMS.dem_damping = seepageflow_data->dem_damping();

        CUDA_SEEPAGEFLOW_PARAMS.sf_c0 = seepageflow_data->sf_c0();
        CUDA_SEEPAGEFLOW_PARAMS.sf_cd = seepageflow_data->sf_cd();
        CUDA_SEEPAGEFLOW_PARAMS.sf_csat = seepageflow_data->sf_csat();
        CUDA_SEEPAGEFLOW_PARAMS.sf_cmc = seepageflow_data->sf_cmc();
        CUDA_SEEPAGEFLOW_PARAMS.sf_cmc_p = seepageflow_data->sf_cmc_p();

        CUDA_SEEPAGEFLOW_PARAMS.sf_a0 = seepageflow_data->sf_a0();
        CUDA_SEEPAGEFLOW_PARAMS.sf_asat = seepageflow_data->sf_asat();
        CUDA_SEEPAGEFLOW_PARAMS.sf_amc = seepageflow_data->sf_amc();
        CUDA_SEEPAGEFLOW_PARAMS.sf_amc_p = seepageflow_data->sf_amc_p();

        KIRI_LOG_DEBUG("data={0},{1},{2},{3}",
                       CUDA_SEEPAGEFLOW_PARAMS.sf_cmc,
                       CUDA_SEEPAGEFLOW_PARAMS.sf_cmc_p,
                       CUDA_SEEPAGEFLOW_PARAMS.sf_amc,
                       CUDA_SEEPAGEFLOW_PARAMS.sf_amc_p);

        CUDA_SEEPAGEFLOW_PARAMS.sf_dry_sand_color = FbsToKiriCUDA(*seepageflow_data->sf_dry_sand_color());
        CUDA_SEEPAGEFLOW_PARAMS.sf_wet_sand_color = FbsToKiriCUDA(*seepageflow_data->sf_wet_sand_color());

        CUDA_SEEPAGEFLOW_PARAMS.dt = 0.5f * CUDA_SEEPAGEFLOW_PARAMS.dem_particle_radius / std::sqrtf(CUDA_SEEPAGEFLOW_PARAMS.dem_young / CUDA_SEEPAGEFLOW_PARAMS.dem_density);
        CUDA_SEEPAGEFLOW_PARAMS.gravity = FbsToKiriCUDA(*seepageflow_data->gravity());

        // init water box volume particles
        auto water_box = scene_config_data->water_box_volume();
        auto water_box_size = FbsToKiriCUDA(*water_box->box_size());
        auto water_box_lower = FbsToKiriCUDA(*water_box->box_lower());
        auto water_box_color = FbsToKiriCUDA(*water_box->box_color());

        // init sand box volume particles
        auto sand_box = scene_config_data->sand_box_volume();
        auto sand_box_size = FbsToKiriCUDA(*sand_box->box_size());
        auto sand_box_lower = FbsToKiriCUDA(*sand_box->box_lower());
        auto sand_box_color = FbsToKiriCUDA(*sand_box->box_color());

        // sph emitter
        auto sph_emitter = scene_config_data->sph_emitter();
        CUDA_SPH_EMITTER_PARAMS.enable = sph_emitter->enable();
        CUDA_SPH_EMITTER_PARAMS.run = false;
        CUDA_SPH_EMITTER_PARAMS.emit_pos = FbsToKiriCUDA(*sph_emitter->emit_pos());
        CUDA_SPH_EMITTER_PARAMS.emit_vel = FbsToKiriCUDA(*sph_emitter->emit_vel());
        CUDA_SPH_EMITTER_PARAMS.emit_col = water_box_color;

        CUDA_SPH_EMITTER_PARAMS.emit_radius = sph_emitter->emit_radius();
        CUDA_SPH_EMITTER_PARAMS.emit_width = sph_emitter->emit_width();
        CUDA_SPH_EMITTER_PARAMS.emit_height = sph_emitter->emit_height();
        switch (sph_emitter->emit_type())
        {
        case FlatBuffers::CudaSphEmitterType::CudaSphEmitterType_SQUARE:
            CUDA_SPH_EMITTER_PARAMS.emit_type = CudaSphEmitterType::SQUARE;
            break;
        case FlatBuffers::CudaSphEmitterType::CudaSphEmitterType_CIRCLE:
            CUDA_SPH_EMITTER_PARAMS.emit_type = CudaSphEmitterType::CIRCLE;
            break;
        case FlatBuffers::CudaSphEmitterType::CudaSphEmitterType_RECTANGLE:
            CUDA_SPH_EMITTER_PARAMS.emit_type = CudaSphEmitterType::RECTANGLE;
            break;
        }

        // scene data
        auto app_data = scene_config_data->app_data();
        auto scene_data = app_data->scene();
        CUDA_BOUNDARY_PARAMS.lowest_point = FbsToKiriCUDA(*scene_data->world_lower());
        CUDA_BOUNDARY_PARAMS.highest_point = FbsToKiriCUDA(*scene_data->world_upper());
        CUDA_BOUNDARY_PARAMS.world_size = FbsToKiriCUDA(*scene_data->world_size());
        CUDA_BOUNDARY_PARAMS.world_center = FbsToKiriCUDA(*scene_data->world_center());

        CUDA_BOUNDARY_PARAMS.kernel_radius = CUDA_SEEPAGEFLOW_PARAMS.sph_kernel_radius;
        CUDA_BOUNDARY_PARAMS.grid_size = make_int3((CUDA_BOUNDARY_PARAMS.highest_point - CUDA_BOUNDARY_PARAMS.lowest_point) / CUDA_BOUNDARY_PARAMS.kernel_radius);

        mRunLiquidNumber = 0;
        mTotalFrameNumber = 300;
        CudaEmitterPtr emitter = std::make_shared<CudaEmitter>(
            CUDA_SPH_EMITTER_PARAMS.emit_pos,
            CUDA_SPH_EMITTER_PARAMS.emit_vel,
            CUDA_SPH_EMITTER_PARAMS.enable);

        // boundary sampling
        BoundaryData boundaryData;
        auto boundaryEmitter = std::make_shared<CudaBoundaryEmitter>();

        boundaryEmitter->BuildWorldBoundary(boundaryData, CUDA_BOUNDARY_PARAMS.lowest_point, CUDA_BOUNDARY_PARAMS.highest_point, CUDA_SEEPAGEFLOW_PARAMS.sph_particle_radius);
        auto boundary_shapes = scene_config_data->boundary_shape_volume();
        if (boundary_shapes->folders()->size() != 0)
        {
            auto fbs_folders = boundary_shapes->folders();
            auto fbs_files = boundary_shapes->files();
            for (size_t i = 0; i < fbs_folders->size(); i++)
                boundaryEmitter->BuildBoundaryShapeVolume(boundaryData, KiriUtils::ReadBgeoFileForGPU(fbs_folders->GetAsString(i)->c_str(), fbs_files->GetAsString(i)->c_str()));
        }

        // sf type
        switch (scene_config_data->seepage_flow_type())
        {
        case CudaSeepageflowType::SF:
            CUDA_SEEPAGEFLOW_PARAMS.sf_type = SF;
            break;
        case CudaSeepageflowType::MULTI_SF:
            CUDA_SEEPAGEFLOW_PARAMS.sf_type = MULTI_SF;
            break;
        }

        // init volume data
        SeepageflowVolumeData volumeData;
        SeepageflowMultiVolumeData multiVolumeData;

        auto volumeEmitter = std::make_shared<CudaVolumeEmitter>();

        // init box volume particles
        if (sand_box_size.x != 0 && sand_box_size.y != 0 && sand_box_size.z != 0)
            volumeEmitter->BuildSeepageflowBoxVolume(
                volumeData,
                sand_box_lower,
                sand_box_size,
                CUDA_SEEPAGEFLOW_PARAMS.dem_particle_radius,
                sand_box_color,
                CUDA_SEEPAGEFLOW_PARAMS.dem_mass,
                1,
                CUDA_SEEPAGEFLOW_PARAMS.dem_particle_radius * 0.001f);
        if (water_box_size.x != 0 && water_box_size.y != 0 && water_box_size.z != 0)
            volumeEmitter->BuildSeepageflowBoxVolume(
                volumeData,
                water_box_lower,
                water_box_size,
                CUDA_SEEPAGEFLOW_PARAMS.sph_particle_radius,
                water_box_color,
                CUDA_SEEPAGEFLOW_PARAMS.sph_mass,
                0);

        // init shape volume particles
        Vec_String folders, files;
        auto sand_shapes = scene_config_data->sand_shape_volume();

        //
        CudaSFParticlesPtr particles;
        CudaGNSearcherPtr searcher;

        auto fbs_folders = sand_shapes->folders();
        auto fbs_files = sand_shapes->files();

        auto shapeFileSize = fbs_folders->size();

        auto sph_solver_type = scene_config_data->sph_solver_type();

        if (CUDA_SEEPAGEFLOW_PARAMS.sf_type == SF)
        {
            if (shapeFileSize != 0)
            {
                for (size_t i = 0; i < shapeFileSize; i++)
                {
                    folders.emplace_back(fbs_folders->GetAsString(i)->c_str());
                    files.emplace_back(fbs_files->GetAsString(i)->c_str());
                }
                auto sandShapes = KiriUtils::ReadMultiBgeoFilesForGPU(folders, files);

                volumeEmitter->BuildSeepageflowShapeVolume(
                    volumeData,
                    sandShapes,
                    CUDA_SEEPAGEFLOW_PARAMS.sf_dry_sand_color,
                    CUDA_SEEPAGEFLOW_PARAMS.dem_density,
                    sand_shapes->offset_ground(),
                    CUDA_BOUNDARY_PARAMS.lowest_point.y);
            }

            CUDA_SEEPAGEFLOW_PARAMS.dem_particle_radius = volumeData.sandMinRadius;
            CUDA_SEEPAGEFLOW_PARAMS.dt = 0.5f * volumeData.sandMinRadius / std::sqrtf(CUDA_SEEPAGEFLOW_PARAMS.dem_young / CUDA_SEEPAGEFLOW_PARAMS.dem_density);
            KIRI_LOG_INFO("Number of total particles = {0}, dt={1}", volumeData.pos.size(), CUDA_SEEPAGEFLOW_PARAMS.dt);

            if (sph_solver_type == FlatBuffers::CudaSphType::CudaSphType_IISPH)
            {
                particles =
                    std::make_shared<CudaIISFParticles>(
                        CUDA_SEEPAGEFLOW_APP_PARAMS.max_num,
                        volumeData.pos,
                        volumeData.col,
                        volumeData.label,
                        volumeData.mass,
                        volumeData.radius);

                searcher = std::make_shared<CudaGNSearcher>(
                    CUDA_BOUNDARY_PARAMS.lowest_point,
                    CUDA_BOUNDARY_PARAMS.highest_point,
                    particles->MaxSize(),
                    CUDA_BOUNDARY_PARAMS.kernel_radius,
                    SearcherParticleType::IISEEPAGE);
            }
            else
            {
                particles =
                    std::make_shared<CudaSFParticles>(
                        CUDA_SEEPAGEFLOW_APP_PARAMS.max_num,
                        volumeData.pos,
                        volumeData.col,
                        volumeData.label,
                        volumeData.mass,
                        volumeData.radius);

                searcher = std::make_shared<CudaGNSearcher>(
                    CUDA_BOUNDARY_PARAMS.lowest_point,
                    CUDA_BOUNDARY_PARAMS.highest_point,
                    particles->MaxSize(),
                    CUDA_BOUNDARY_PARAMS.kernel_radius,
                    SearcherParticleType::SEEPAGE);
            }
        }
        else if (CUDA_SEEPAGEFLOW_PARAMS.sf_type == MULTI_SF)
        {
            if (shapeFileSize != 0)
            {
                for (size_t i = 0; i < shapeFileSize; i++)
                {
                    auto cda0asat = FbsToKiriCUDA(*(scene_config_data->seepage_flow_cda0asat()->GetAs<FlatBuffers::float3>(i)));
                    auto amcamcp = FbsToKiriCUDA(*(scene_config_data->seepage_flow_amcamcp()->GetAs<FlatBuffers::float2>(i)));
                    auto sandShape = KiriUtils::ReadBgeoFileForGPU(fbs_folders->GetAsString(i)->c_str(), fbs_files->GetAsString(i)->c_str());

                    volumeEmitter->BuildSeepageflowShapeMultiVolume(
                        multiVolumeData,
                        sandShape,
                        CUDA_SEEPAGEFLOW_PARAMS.sf_dry_sand_color,
                        CUDA_SEEPAGEFLOW_PARAMS.dem_density,
                        cda0asat,
                        amcamcp,
                        sand_shapes->offset_ground(),
                        CUDA_BOUNDARY_PARAMS.lowest_point.y);

                    KIRI_LOG_DEBUG("Cd A0 Asat Amc Amcp = {0}, {1}, {2}, {3}, {4}", cda0asat.x, cda0asat.y, cda0asat.z, amcamcp.x, amcamcp.y);
                }
            }

            CUDA_SEEPAGEFLOW_PARAMS.dem_particle_radius = multiVolumeData.sandMinRadius;
            CUDA_SEEPAGEFLOW_PARAMS.dt = 0.5f * CUDA_SEEPAGEFLOW_PARAMS.dem_particle_radius / std::sqrtf(CUDA_SEEPAGEFLOW_PARAMS.dem_young / CUDA_SEEPAGEFLOW_PARAMS.dem_density);
            KIRI_LOG_INFO("Number of total particles = {0}", multiVolumeData.pos.size());

            particles =
                std::make_shared<CudaSFParticles>(
                    CUDA_SEEPAGEFLOW_APP_PARAMS.max_num,
                    multiVolumeData.pos,
                    multiVolumeData.col,
                    multiVolumeData.label,
                    multiVolumeData.mass,
                    multiVolumeData.radius,
                    multiVolumeData.cda0asat,
                    multiVolumeData.amcamcp);

            searcher = std::make_shared<CudaGNSearcher>(
                CUDA_BOUNDARY_PARAMS.lowest_point,
                CUDA_BOUNDARY_PARAMS.highest_point,
                particles->MaxSize(),
                CUDA_BOUNDARY_PARAMS.kernel_radius,
                SearcherParticleType::SEEPAGE_MULTI);
        }

        auto boundaryParticles = std::make_shared<CudaBoundaryParticles>(boundaryData.pos, boundaryData.label);
        KIRI_LOG_INFO("Number of Boundary Particles = {0}", boundaryParticles->Size());

        bool adaptiveSubTimeStep = false;
        CudaSphSFSolverPtr pSolver;
        CUDA_SEEPAGEFLOW_PARAMS.solver_type = SPH_SOLVER;
        switch (sph_solver_type)
        {
        case FlatBuffers::CudaSphType::CudaSphType_SPH:
            pSolver = std::make_shared<CudaSphSFSolver>(
                particles->MaxSize());
            KIRI_LOG_INFO("Current Fluid Solver= SPH");
            break;
        case FlatBuffers::CudaSphType::CudaSphType_WCSPH:
            adaptiveSubTimeStep = true;
            pSolver = std::make_shared<CudaWCSphSFSolver>(
                particles->MaxSize());
            CUDA_SEEPAGEFLOW_PARAMS.solver_type = WCSPH_SOLVER;
            KIRI_LOG_INFO("Current Fluid Solver= WCSPH");
            break;
        case FlatBuffers::CudaSphType::CudaSphType_IISPH:
            pSolver = std::make_shared<CudaIISphSFSolver>(
                particles->MaxSize());
            CUDA_SEEPAGEFLOW_PARAMS.solver_type = IISPH_SOLVER;
            KIRI_LOG_INFO("Current Fluid Solver= IISPH");
            break;
        default:
            pSolver = std::make_shared<CudaSphSFSolver>(
                particles->MaxSize());
            KIRI_LOG_INFO("Current Fluid Solver= SPH");
            break;
        }

        // bgeo file export & render FPS
        CUDA_SEEPAGEFLOW_APP_PARAMS.bgeo_export = app_data->bgeo_export_mode_enable();
        if (app_data->render_mode_enable())
            SetRenderFps(app_data->render_mode_fps());
        else
            SetRenderFps(1.f / CUDA_SEEPAGEFLOW_PARAMS.dt);

        CudaGNBoundarySearcherPtr boundarySearcher = std::make_shared<CudaGNBoundarySearcher>(
            CUDA_BOUNDARY_PARAMS.lowest_point,
            CUDA_BOUNDARY_PARAMS.highest_point,
            boundaryParticles->MaxSize(),
            CUDA_BOUNDARY_PARAMS.kernel_radius);

        mSystem = std::make_shared<CudaSFSystem>(
            particles,
            boundaryParticles,
            pSolver,
            searcher,
            boundarySearcher,
            emitter,
            adaptiveSubTimeStep);

        // String ystr = "";
        // String xstr = "";
        // auto G = QuadraticBezierCoeff(0.f, 1.5f, 0.5f, 0.8f);
        // for (float i = 0.f; i < 1.f; i += 0.001f)
        // {
        //     auto yres = G(i);
        //     xstr += std::to_string(i) + ",";
        //     ystr += std::to_string(yres) + ",";
        // }

        // KIRI_LOG_DEBUG("x=[{0}]", xstr);
        // KIRI_LOG_DEBUG("y=[{0}]", ystr);
    }

    void KiriSeepageFlowOffline::OnPBSUpdate(const KIRI::KiriTimeStep &DeltaTime)
    {
        if (CUDA_SEEPAGEFLOW_APP_PARAMS.run && mSimCount < mTotalFrameNumber + mRunLiquidNumber)
        {
            if (mSimCount == mRunLiquidNumber)
            {
                // export bgeo file
                CUDA_SEEPAGEFLOW_APP_PARAMS.bgeo_export = CUDA_SEEPAGEFLOW_APP_PARAMS.run;
                if (CUDA_SPH_EMITTER_PARAMS.enable)
                    CUDA_SPH_EMITTER_PARAMS.run = CUDA_SEEPAGEFLOW_APP_PARAMS.run;
            }

            if (mSystem->GetAdaptiveSubTimeStep())
            {
                float remainingTime = mRenderInterval;
                KIRI_LOG_INFO("Simulation Frame={0}, Adaptive Sub-Simulation", ++mSimCount);
                mPerFrameTimer.Restart();
                size_t i = 0;
                while (remainingTime > KIRI_EPSILON)
                {
                    KIRI_LOG_INFO("Current Sub-Simulation RemainTime={0},Sub-Simulation Step={1}", remainingTime, ++i);
                    mSystem->UpdateSystem(remainingTime);
                    remainingTime -= remainingTime / static_cast<float>(mSystem->GetNumOfSubTimeSteps());
                }
            }
            else
            {
                auto numOfSubTimeSteps = mSystem->GetNumOfSubTimeSteps();
                KIRI_LOG_INFO("Simulation Frame={0}, Sub-Simulation Total Number={1}", ++mSimCount, numOfSubTimeSteps);

                mPerFrameTimer.Restart();
                for (size_t i = 0; i < numOfSubTimeSteps; i++)
                {
                    KIRI_LOG_INFO("Current Sub-Simulation/ Total Number ={0}/{1}", i + 1, numOfSubTimeSteps);
                    mSystem->UpdateSystem(mRenderInterval);
                }
            }

            KIRI_LOG_INFO("Time Per Frame={0}", mPerFrameTimer.Elapsed());
            mTotalFrameTime += mPerFrameTimer.Elapsed();

            if (CUDA_SEEPAGEFLOW_APP_PARAMS.bgeo_export)
            {
                auto particles = mSystem->GetSFParticles();
                KiriUtils::ExportBgeoFileCUDA(
                    CUDA_SEEPAGEFLOW_APP_PARAMS.bgeo_export_folder,
                    KiriUtils::UInt2Str4Digit(mSimCount - mRunLiquidNumber),
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

            KIRI_LOG_INFO("Average Per Frame={0}", mTotalFrameTime / (mTotalFrameNumber + mRunLiquidNumber));
        }
    }

    void KiriSeepageFlowOffline::SetupPBSScene()
    {
        KIRI_LOG_DEBUG("Seepageflow Offline:SetupPBSScene");
    }

    void KiriSeepageFlowOffline::OnImguiRender()
    {
        static bool p_open = true;
        if (p_open)
        {
            ImGui::SetNextWindowSize(ImVec2(430, 450), ImGuiCond_FirstUseEver);
            if (ImGui::Begin("Seepageflow Examples", &p_open))
            {
                if (ImGui::CollapsingHeader("Example Scene"))
                {
                    const char *items[] = {
                        "seepageflow_valley_dam_breach_wcsph",
                        "seepageflow_bunny_wcsph",
                        "seepageflow_bunny_iisph",
                        "seepageflow_chess_wcsph",
                        "seepageflow_funnel",
                        "seepageflow_dam_breach_wcsph",
                        "seepageflow_dam_breach",
                        "seepageflow_bunny"};

                    ImGui::Combo("Scene Config Data File", &CUDA_SEEPAGEFLOW_APP_PARAMS.scene_data_idx, items, IM_ARRAYSIZE(items));
                    ChangeSceneConfigData(items[CUDA_SEEPAGEFLOW_APP_PARAMS.scene_data_idx]);
                    ImGui::InputText("Export Folder Path", CUDA_SEEPAGEFLOW_APP_PARAMS.bgeo_export_folder, 320);
                }

                if (ImGui::CollapsingHeader("Simulation"))
                {
                    if (CUDA_SPH_EMITTER_PARAMS.enable)
                        ImGui::Checkbox("Emit Particles", &CUDA_SPH_EMITTER_PARAMS.run);

                    ImGui::Checkbox("Run", &CUDA_SEEPAGEFLOW_APP_PARAMS.run);
                }
                ImGui::End();
            }
        }
    }
} // namespace KIRI