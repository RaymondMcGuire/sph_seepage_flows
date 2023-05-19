/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-05-14 20:01:11
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-05-17 21:04:09
 * @FilePath: \sph_seepage_flows\seepage_flow\src\seepageflow\main.cpp
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */
// clang-format off
#include <sf_cuda_define.h>
#include <kiri_utils.h>
#include<cuda/cuda_helper.h>
#include <vtk_helper/vtk_polygonal_writer.h>
#include <vtk_helper/vtk_reader.h>
#include <filesystem>
// clang-format on
using namespace KIRI;

// global params
auto ExampleName = "seepageflow_uni_slide_dfsph";

auto RunLiquidNumber = 0;
auto TotalFrameNumber = 180;
auto SimCount = 0;
auto TotalFrameTime = 0.f;
auto RenderInterval = 1.f / 60.f;

KiriTimer PerFrameTimer;
CudaSFSystemPtr SFSystem;

void Seepage_Uni_Slide_DFSPH() {

  KIRI_LOG_DEBUG("Example:  Seepage_Uni_Slide_DFSPH");
  auto vtk_export_path = String(EXPORT_PATH) + "vtk/" + ExampleName + "/";

  // scene config
  auto cuda_lowest_point =
      make_float3(-2.f, -2.f, -2.f); //左下角与右上角 坐标顺序 x y z
  auto cuda_highest_point = make_float3(15.f, 10.f, 25.f);
  auto cuda_world_size = cuda_highest_point - cuda_lowest_point;
  auto cuda_world_center = (cuda_highest_point + cuda_lowest_point) / 2.f;
  CUDA_SEEPAGEFLOW_APP_PARAMS.max_num = 8000000;

  // sph params
  CUDA_SEEPAGEFLOW_PARAMS.sph_density = 1000.f;
  CUDA_SEEPAGEFLOW_PARAMS.sph_particle_radius = 0.1f;
  CUDA_SEEPAGEFLOW_PARAMS.sph_kernel_radius =
      4.f * CUDA_SEEPAGEFLOW_PARAMS.sph_particle_radius;

  auto diam = 2.f * CUDA_SEEPAGEFLOW_PARAMS.sph_particle_radius;
  CUDA_SEEPAGEFLOW_PARAMS.sph_mass =
      0.8f * diam * diam * diam * CUDA_SEEPAGEFLOW_PARAMS.sph_density;

  auto visc = 0.05f;
  auto sound_speed = 100.f;
  auto nu =
      (visc + visc) * CUDA_SEEPAGEFLOW_PARAMS.sph_kernel_radius * sound_speed;
  auto boundary_friction = 0.2f;
  auto bnu = boundary_friction * CUDA_SEEPAGEFLOW_PARAMS.sph_kernel_radius *
             sound_speed;

  CUDA_SEEPAGEFLOW_PARAMS.sph_stiff = 0.001f * sound_speed * sound_speed *
                                      CUDA_SEEPAGEFLOW_PARAMS.sph_density / 7.f;
  CUDA_SEEPAGEFLOW_PARAMS.sph_visc = visc;
  CUDA_SEEPAGEFLOW_PARAMS.sph_nu = nu;
  CUDA_SEEPAGEFLOW_PARAMS.sph_bnu = bnu;

  // dem params
  CUDA_SEEPAGEFLOW_PARAMS.dem_density = 2700.f;
  CUDA_SEEPAGEFLOW_PARAMS.dem_young = 1e5f;
  CUDA_SEEPAGEFLOW_PARAMS.dem_poisson = 0.3f;
  CUDA_SEEPAGEFLOW_PARAMS.dem_tan_friction_angle = std::tanf(0.5f);
  CUDA_SEEPAGEFLOW_PARAMS.dem_damping = 0.6f;

  CUDA_SEEPAGEFLOW_PARAMS.sf_c0 = 0.7f;
  CUDA_SEEPAGEFLOW_PARAMS.sf_cd = 0.005f;
  CUDA_SEEPAGEFLOW_PARAMS.sf_csat = 0.f;
  CUDA_SEEPAGEFLOW_PARAMS.sf_cmc = 1.f;
  CUDA_SEEPAGEFLOW_PARAMS.sf_cmc_p = 0.01f;

  CUDA_SEEPAGEFLOW_PARAMS.sf_a0 = 2.f;
  CUDA_SEEPAGEFLOW_PARAMS.sf_asat = 1.f;
  CUDA_SEEPAGEFLOW_PARAMS.sf_amc = 2.1f;
  CUDA_SEEPAGEFLOW_PARAMS.sf_amc_p = 0.8f;

  CUDA_SEEPAGEFLOW_PARAMS.sf_dry_sand_color =
      make_float3(0.88f, 0.79552f, 0.5984f);
  CUDA_SEEPAGEFLOW_PARAMS.sf_wet_sand_color = make_float3(0.38f, 0.29f, 0.14f);

  CUDA_SEEPAGEFLOW_PARAMS.gravity = make_float3(0.0f, -9.8f, 0.0f);
  CUDA_SEEPAGEFLOW_PARAMS.max_force_factor = 8.f;

  // sph emitter
  CUDA_SPH_EMITTER_PARAMS.enable = false;
  CUDA_SPH_EMITTER_PARAMS.run = false;
  CUDA_SPH_EMITTER_PARAMS.emit_pos = make_float3(0.38f, 1.0f, 3.8f);
  CUDA_SPH_EMITTER_PARAMS.emit_vel = make_float3(0.f, -1.2f, 0.f);
  CUDA_SPH_EMITTER_PARAMS.emit_col = make_float3(127.f, 205.f, 255.f) / 255.f;

  // 发射器形状参数 圆形 矩形.
  CUDA_SPH_EMITTER_PARAMS.emit_radius = 0.6f;
  CUDA_SPH_EMITTER_PARAMS.emit_width = 0.6f;
  CUDA_SPH_EMITTER_PARAMS.emit_height = 0.6f;
  CUDA_SPH_EMITTER_PARAMS.emit_type = CudaSphEmitterType::SQUARE;

  // scene data
  CUDA_BOUNDARY_PARAMS.lowest_point = cuda_lowest_point;
  CUDA_BOUNDARY_PARAMS.highest_point = cuda_highest_point;
  CUDA_BOUNDARY_PARAMS.world_size = cuda_world_size;
  CUDA_BOUNDARY_PARAMS.world_center = cuda_world_center;

  CUDA_BOUNDARY_PARAMS.kernel_radius =
      CUDA_SEEPAGEFLOW_PARAMS.sph_kernel_radius;
  CUDA_BOUNDARY_PARAMS.grid_size = make_int3(
      (CUDA_BOUNDARY_PARAMS.highest_point - CUDA_BOUNDARY_PARAMS.lowest_point) /
      CUDA_BOUNDARY_PARAMS.kernel_radius);

  // boudnary particle radius
  CUDA_SEEPAGEFLOW_PARAMS.boundary_particle_radius = 0.1f;

  // init emitter
  CudaEmitterPtr emitter = std::make_shared<CudaEmitter>(
      CUDA_SPH_EMITTER_PARAMS.emit_pos, CUDA_SPH_EMITTER_PARAMS.emit_vel,
      CUDA_SPH_EMITTER_PARAMS.enable);

  // boundary sampling
  BoundaryData boundary_data;
  auto boundary_emitter = std::make_shared<CudaBoundaryEmitter>();

  // build world boundary data
  boundary_emitter->BuildWorldBoundary(
      boundary_data, CUDA_BOUNDARY_PARAMS.lowest_point,
      CUDA_BOUNDARY_PARAMS.highest_point,
      CUDA_SEEPAGEFLOW_PARAMS.boundary_particle_radius);

  // build custom boundary data
  auto boundary_vtk_file_path = String(DB_PBR_PATH) + "vtk/boundary.vtk";
  auto boundary_particles_data = VTKReader::ReadPoints(boundary_vtk_file_path);
  boundary_emitter->BuildBoundaryShapeVolume(boundary_data,
                                             boundary_particles_data, true);

  auto vtk_world_boundary = vtk_export_path + "world_boundary.vtk";
  auto vtk_custom_boundary = vtk_export_path + "custom_boundary.vtk";

  auto vtk_world_boundary_writer = std::make_shared<VTKPolygonalWriter>(
      vtk_world_boundary, boundary_data.pos, boundary_data.label, 0);
  vtk_world_boundary_writer->WriteToFile();

  auto vtk_custom_boundary_writer = std::make_shared<VTKPolygonalWriter>(
      vtk_custom_boundary, boundary_data.pos, boundary_data.label, 1);
  vtk_custom_boundary_writer->WriteToFile();

  // material type (SF: unified material; MULTI_SF: multiple types of materials)
  CUDA_SEEPAGEFLOW_PARAMS.sf_type = MULTI_SF;

  // shape sampling
  auto offset2Ground = false;
  SeepageflowMultiVolumeData multi_volume_data;
  auto volumeEmitter = std::make_shared<CudaVolumeEmitter>();

  // multiple sand objects
  Vec_String sand_shape_files;

  // object 1:
  sand_shape_files.emplace_back(String(DB_PBR_PATH) + "vtk/slide.vtk");

  // object 2: dam/dam2.bego
  // shape_folders.emplace_back("dam");
  // shape_files.emplace_back("dam2");

  std::vector<float3> cd_a0_asat;
  std::vector<float2> amc_amcp;

  // params(cd, a0, asat, amc, amcp) for object1
  cd_a0_asat.emplace_back(make_float3(CUDA_SEEPAGEFLOW_PARAMS.sf_cd,
                                      CUDA_SEEPAGEFLOW_PARAMS.sf_a0,
                                      CUDA_SEEPAGEFLOW_PARAMS.sf_asat));
  amc_amcp.emplace_back(make_float2(CUDA_SEEPAGEFLOW_PARAMS.sf_amc,
                                    CUDA_SEEPAGEFLOW_PARAMS.sf_amc_p));

  // params(cd, a0, asat, amc, amcp) for object2
  // cd_a0_asat.emplace_back(make_float3(0.015f, CUDA_SEEPAGEFLOW_PARAMS.sf_a0,
  // CUDA_SEEPAGEFLOW_PARAMS.sf_asat));
  // amc_amcp.emplace_back(make_float2(CUDA_SEEPAGEFLOW_PARAMS.sf_amc,
  // CUDA_SEEPAGEFLOW_PARAMS.sf_amc_p));

  multi_volume_data.min_radius = CUDA_SEEPAGEFLOW_PARAMS.sph_particle_radius;
  for (auto i = 0; i < sand_shape_files.size(); i++) {
    auto cda0asat = cd_a0_asat[i];
    auto amcamcp = amc_amcp[i];

    auto sand_shape = VTKReader::ReadPoints(sand_shape_files[i]);

    volumeEmitter->BuildSeepageflowShapeMultiVolume(
        multi_volume_data, sand_shape, 0.1f,
        CUDA_SEEPAGEFLOW_PARAMS.sf_dry_sand_color,
        CUDA_SEEPAGEFLOW_PARAMS.dem_density, cda0asat, amcamcp, true,
        offset2Ground, CUDA_BOUNDARY_PARAMS.lowest_point.y,
        make_float2(0.f, 0.f));

    KIRI_LOG_DEBUG(
        "Object({0}) Params: Cd A0 Asat Amc Amcp = {1}, {2}, {3}, {4}, {5}",
        i + 1, cda0asat.x, cda0asat.y, cda0asat.z, amcamcp.x, amcamcp.y);
  }

  // dt
  CUDA_SEEPAGEFLOW_PARAMS.dt = 0.5f * multi_volume_data.min_radius /
                               std::sqrtf(CUDA_SEEPAGEFLOW_PARAMS.dem_young /
                                          CUDA_SEEPAGEFLOW_PARAMS.dem_density);

  KIRI_LOG_INFO(
      "Number of total sand particles = {0}; minimum radius ={1}; dt ={2}",
      multi_volume_data.pos.size(), multi_volume_data.min_radius,
      CUDA_SEEPAGEFLOW_PARAMS.dt);

  // spatial searcher & particles
  CudaSFParticlesPtr particles;

  // dfsf
  particles = std::make_shared<CudaDFSFParticles>(
      CUDA_SEEPAGEFLOW_APP_PARAMS.max_num, multi_volume_data.pos,
      multi_volume_data.col, multi_volume_data.label, multi_volume_data.mass,
      multi_volume_data.inertia, multi_volume_data.radius,
      multi_volume_data.cda0asat, multi_volume_data.amcamcp);

  CudaGNSearcherPtr searcher;
  searcher = std::make_shared<CudaGNSearcher>(
      CUDA_BOUNDARY_PARAMS.lowest_point, CUDA_BOUNDARY_PARAMS.highest_point,
      particles->MaxSize(), CUDA_BOUNDARY_PARAMS.kernel_radius,
      SearcherParticleType::DFSF_MULTI);

  auto boundary_particles = std::make_shared<CudaBoundaryParticles>(
      boundary_data.pos, boundary_data.label);
  KIRI_LOG_INFO("Number of Boundary Particles = {0}",
                boundary_particles->Size());

  bool adaptive_sub_timestep = true;
  CudaSphSFSolverPtr pSolver;

  // dfsph
  pSolver = std::make_shared<CudaDFSphSFSolver>(particles->MaxSize(),
                                                CUDA_SEEPAGEFLOW_PARAMS.dt);
  CUDA_SEEPAGEFLOW_PARAMS.solver_type = DFSPH_SOLVER;
  KIRI_LOG_INFO("Current Fluid Solver= DFSPH");

  // bgeo file export & render FPS
  CUDA_SEEPAGEFLOW_APP_PARAMS.enable_write2file = true;

  CudaGNBoundarySearcherPtr boundary_searcher =
      std::make_shared<CudaGNBoundarySearcher>(
          CUDA_BOUNDARY_PARAMS.lowest_point, CUDA_BOUNDARY_PARAMS.highest_point,
          boundary_particles->MaxSize(), CUDA_BOUNDARY_PARAMS.kernel_radius);

  SFSystem = std::make_shared<CudaSFSystem>(
      particles, boundary_particles, pSolver, searcher, boundary_searcher,
      emitter, adaptive_sub_timestep);
}

void main() {
  KiriLog::Init();

  // vtk exporter params
  auto vtk_export_path = String(EXPORT_PATH) + "vtk/" + ExampleName + "/";

  std::error_code error_code;
  std::filesystem::create_directories(vtk_export_path, error_code);

  Seepage_Uni_Slide_DFSPH();

  CUDA_SEEPAGEFLOW_APP_PARAMS.run = true;

  while (CUDA_SEEPAGEFLOW_APP_PARAMS.run) {
    if (CUDA_SEEPAGEFLOW_APP_PARAMS.run &&
        SimCount < TotalFrameNumber + RunLiquidNumber) {
      if (SimCount == RunLiquidNumber) {
        // export bgeo file
        CUDA_SEEPAGEFLOW_APP_PARAMS.enable_write2file =
            CUDA_SEEPAGEFLOW_APP_PARAMS.run;
        if (CUDA_SPH_EMITTER_PARAMS.enable)
          CUDA_SPH_EMITTER_PARAMS.run = CUDA_SEEPAGEFLOW_APP_PARAMS.run;
      }

      auto remaining_time = RenderInterval;
      KIRI_LOG_INFO("Simulation Frame={0}, Adaptive Sub-Simulation",
                    ++SimCount);
      PerFrameTimer.Restart();
      auto sub_timestep = 0;
      while (remaining_time > KIRI_EPSILON) {
        KIRI_LOG_INFO(
            "Current Sub-Simulation RemainTime={0},Sub-Simulation Step={1}",
            remaining_time, ++sub_timestep);
        SFSystem->UpdateSystem(remaining_time);
        remaining_time -= SFSystem->GetCurrentTimeStep();
      }

      KIRI_LOG_INFO("Time Per Frame={0}", PerFrameTimer.Elapsed());
      TotalFrameTime += PerFrameTimer.Elapsed();

      if (CUDA_SEEPAGEFLOW_APP_PARAMS.enable_write2file) {
        auto particles = SFSystem->GetSFParticles();
        auto cpu_pos =
            TransferGPUData2CPU(particles->GetPosPtr(), particles->Size());
        auto cpu_label =
            TransferGPUData2CPU(particles->GetLabelPtr(), particles->Size());
        auto cpu_id =
            TransferGPUData2CPU(particles->GetIdPtr(), particles->Size());
        auto cpu_radius =
            TransferGPUData2CPU(particles->GetRadiusPtr(), particles->Size());
        auto cpu_vel =
            TransferGPUData2CPU(particles->GetVelPtr(), particles->Size());
        auto cpu_acc =
            TransferGPUData2CPU(particles->GetAccPtr(), particles->Size());
        auto cpu_sat = TransferGPUData2CPU(particles->GetSaturationPtr(),
                                           particles->Size());
        auto cpu_rho =
            TransferGPUData2CPU(particles->GetDensityPtr(), particles->Size());

        auto vtk_file_water = vtk_export_path + "water_" +
                              UInt2Str4Digit(SimCount - RunLiquidNumber) +
                              ".vtk";
        auto vtk_file_sand = vtk_export_path + "sand_" +
                             UInt2Str4Digit(SimCount - RunLiquidNumber) +
                             ".vtk";

        auto vtk_water_writer = std::make_shared<VTKPolygonalWriter>(
            vtk_file_water, cpu_pos, cpu_label, 0);
        vtk_water_writer->AddIntData("Id", cpu_id);
        vtk_water_writer->AddVectorFloatData("Velocity", cpu_vel);
        vtk_water_writer->AddVectorFloatData("Accelerate", cpu_acc);
        vtk_water_writer->AddFloatData("Saturation", cpu_sat);
        vtk_water_writer->AddFloatData("Rho", cpu_rho);
        vtk_water_writer->AddFloatData("Radius", cpu_radius);

        vtk_water_writer->WriteToFile();

        auto vtk_sand_writer = std::make_shared<VTKPolygonalWriter>(
            vtk_file_sand, cpu_pos, cpu_label, 1);
        vtk_sand_writer->AddIntData("Id", cpu_id);
        vtk_sand_writer->AddVectorFloatData("Velocity", cpu_vel);
        vtk_sand_writer->AddVectorFloatData("Accelerate", cpu_acc);
        vtk_sand_writer->AddFloatData("Saturation", cpu_sat);
        vtk_sand_writer->AddFloatData("Rho", cpu_rho);
        vtk_sand_writer->AddFloatData("Radius", cpu_radius);
        vtk_sand_writer->WriteToFile();
      }
    } else if (CUDA_SEEPAGEFLOW_APP_PARAMS.run) {
      CUDA_SEEPAGEFLOW_APP_PARAMS.run = false;

      KIRI_LOG_INFO("Average Per Frame={0}",
                    TotalFrameTime / (TotalFrameNumber + RunLiquidNumber));
    }
  }

  return;
}
