/*** 
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-03-21 12:33:24
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-03-31 17:51:49
 * @FilePath: \sph_seepage_flows\seepage_flow\src\seepageflow\main.cpp
 * @Description: 
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved. 
 */
// clang-format off
#include <sf_cuda_define.h>
#include <kiri_utils.h>
#include <abc/particles-alembic-manager.h>
#include <filesystem>
// clang-format on
using namespace KIRI;

// global params
auto ExampleName = "seepageflow_bunny_wcsph";

auto RunLiquidNumber = 0;
auto TotalFrameNumber = 360;
auto SimCount = 0;
auto TotalFrameTime = 0.f;
auto RenderInterval = 1.f / 60.f;

KiriTimer PerFrameTimer;
CudaSFSystemPtr SFSystem;

void SetupExample1() {

  KIRI_LOG_DEBUG("Seepageflow: Example1 SetupParams");
     ExampleName = "seepageflow_bunny_wcsph";
  // export path
  strcpy(CUDA_SEEPAGEFLOW_APP_PARAMS.bgeo_export_folder,
         (String(EXPORT_PATH) + "bgeo/" + ExampleName).c_str());

  // scene config
  auto cuda_lowest_point = make_float3(0.f);
  auto cuda_highest_point = make_float3(2.f, 2.f, 3.f);
  auto cuda_world_size = cuda_highest_point - cuda_lowest_point;
  auto cuda_world_center = (cuda_highest_point + cuda_lowest_point) / 2.f;
  CUDA_SEEPAGEFLOW_APP_PARAMS.max_num = 450000;

  // sph params
  CUDA_SEEPAGEFLOW_PARAMS.sph_density = 1000.f;
  CUDA_SEEPAGEFLOW_PARAMS.sph_particle_radius = 0.01f;
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
  CUDA_SEEPAGEFLOW_PARAMS.dem_tan_friction_angle = 0.5f;
  CUDA_SEEPAGEFLOW_PARAMS.dem_damping = 0.4f;

  CUDA_SEEPAGEFLOW_PARAMS.sf_c0 = 4.f;
  CUDA_SEEPAGEFLOW_PARAMS.sf_cd = 0.5f;
  CUDA_SEEPAGEFLOW_PARAMS.sf_csat = 0.f;
  CUDA_SEEPAGEFLOW_PARAMS.sf_cmc = 4.1f;
  CUDA_SEEPAGEFLOW_PARAMS.sf_cmc_p = 0.01f;

  CUDA_SEEPAGEFLOW_PARAMS.sf_a0 = 0.f;
  CUDA_SEEPAGEFLOW_PARAMS.sf_asat = 0.8f;
  CUDA_SEEPAGEFLOW_PARAMS.sf_amc = 1.5f;
  CUDA_SEEPAGEFLOW_PARAMS.sf_amc_p = 0.5f;

  CUDA_SEEPAGEFLOW_PARAMS.sf_dry_sand_color =
      make_float3(0.88f, 0.79552f, 0.5984f);
  CUDA_SEEPAGEFLOW_PARAMS.sf_wet_sand_color = make_float3(0.38f, 0.29f, 0.14f);

  CUDA_SEEPAGEFLOW_PARAMS.gravity = make_float3(0.0f, -9.8f, 0.0f);
  CUDA_SEEPAGEFLOW_PARAMS.max_force_factor = 15.f;

  // sph emitter
  CUDA_SPH_EMITTER_PARAMS.enable = true;
  CUDA_SPH_EMITTER_PARAMS.run = false;
  CUDA_SPH_EMITTER_PARAMS.emit_pos =
      make_float3(cuda_world_center.x + cuda_world_size.x / 10.f,
                  cuda_world_center.y + cuda_world_size.y / 5.f,
                  cuda_world_center.z + cuda_world_size.z / 2.3f);
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

  CUDA_BOUNDARY_PARAMS.kernel_radius =
      CUDA_SEEPAGEFLOW_PARAMS.sph_kernel_radius;
  CUDA_BOUNDARY_PARAMS.grid_size = make_int3(
      (CUDA_BOUNDARY_PARAMS.highest_point - CUDA_BOUNDARY_PARAMS.lowest_point) /
      CUDA_BOUNDARY_PARAMS.kernel_radius);

  CUDA_SEEPAGEFLOW_PARAMS.boundary_particle_radius =
      CUDA_SEEPAGEFLOW_PARAMS.sph_particle_radius;

  // init emitter
  CudaEmitterPtr emitter = std::make_shared<CudaEmitter>(
      CUDA_SPH_EMITTER_PARAMS.emit_pos, CUDA_SPH_EMITTER_PARAMS.emit_vel,
      CUDA_SPH_EMITTER_PARAMS.enable);

  // boundary sampling
  BoundaryData boundaryData;
  auto boundaryEmitter = std::make_shared<CudaBoundaryEmitter>();

  boundaryEmitter->BuildWorldBoundary(
      boundaryData, CUDA_BOUNDARY_PARAMS.lowest_point,
      CUDA_BOUNDARY_PARAMS.highest_point,
      CUDA_SEEPAGEFLOW_PARAMS.boundary_particle_radius);

  // material type (SF: unified material; MULTI_SF: multiple types of materials)
  CUDA_SEEPAGEFLOW_PARAMS.sf_type = MULTI_SF;

  // shape sampling
  auto offset2Ground = true;
  SeepageflowMultiVolumeData multiVolumeData;
  auto volumeEmitter = std::make_shared<CudaVolumeEmitter>();

  // multiple sand objects
  Vec_String shape_folders, shape_files;

  // object 1: bunny/bunny.bego
  shape_folders.emplace_back("bunny");
  shape_files.emplace_back("uni_bunny_0.06");

  // object 2: dam/dam2.bego
  // shape_folders.emplace_back("dam");
  // shape_files.emplace_back("dam2");

  std::vector<float3> cd_a0_asat;
  std::vector<float2> amc_amcp;

  // params(cd, a0, asat, amc, amcp) for object1
  cd_a0_asat.emplace_back(make_float3(0.5f, CUDA_SEEPAGEFLOW_PARAMS.sf_a0,
                                      CUDA_SEEPAGEFLOW_PARAMS.sf_asat));
  amc_amcp.emplace_back(make_float2(CUDA_SEEPAGEFLOW_PARAMS.sf_amc,
                                    CUDA_SEEPAGEFLOW_PARAMS.sf_amc_p));

  // params(cd, a0, asat, amc, amcp) for object2
  // cd_a0_asat.emplace_back(make_float3(0.015f, CUDA_SEEPAGEFLOW_PARAMS.sf_a0,
  // CUDA_SEEPAGEFLOW_PARAMS.sf_asat));
  // amc_amcp.emplace_back(make_float2(CUDA_SEEPAGEFLOW_PARAMS.sf_amc,
  // CUDA_SEEPAGEFLOW_PARAMS.sf_amc_p));
 
  multiVolumeData.sandMinRadius =  CUDA_SEEPAGEFLOW_PARAMS.sph_particle_radius;
  for (auto i = 0; i < shape_folders.size(); i++) {
    auto cda0asat = cd_a0_asat[i];
    auto amcamcp = amc_amcp[i];
    auto sandShape = ReadBgeoFileForGPU(shape_folders[i], shape_files[i]);

    volumeEmitter->BuildSeepageflowShapeMultiVolume(
        multiVolumeData, sandShape, CUDA_SEEPAGEFLOW_PARAMS.sf_dry_sand_color,
        CUDA_SEEPAGEFLOW_PARAMS.dem_density, cda0asat, amcamcp, offset2Ground,
        CUDA_BOUNDARY_PARAMS.lowest_point.y, make_float2(1.2f, 0.6f));

    KIRI_LOG_DEBUG(
        "Object({0}) Params: Cd A0 Asat Amc Amcp = {1}, {2}, {3}, {4}, {5}",
        i + 1, cda0asat.x, cda0asat.y, cda0asat.z, amcamcp.x, amcamcp.y);
  }

  // dt
  CUDA_SEEPAGEFLOW_PARAMS.dt = 0.5f * multiVolumeData.sandMinRadius /
                               std::sqrtf(CUDA_SEEPAGEFLOW_PARAMS.dem_young /
                                          CUDA_SEEPAGEFLOW_PARAMS.dem_density);
  KIRI_LOG_INFO(
      "Number of total sand particles = {0}; minimum radius ={1}; dt ={2}",
      multiVolumeData.pos.size(), multiVolumeData.sandMinRadius,
      CUDA_SEEPAGEFLOW_PARAMS.dt);

  // spatial searcher & particles
  CudaSFParticlesPtr particles;

  //dfsf
//   particles = std::make_shared<CudaDFSFParticles>(
//       CUDA_SEEPAGEFLOW_APP_PARAMS.max_num, multiVolumeData.pos,
//       multiVolumeData.col, multiVolumeData.label, multiVolumeData.mass,multiVolumeData.inertia,
//       multiVolumeData.radius, multiVolumeData.cda0asat,
//       multiVolumeData.amcamcp);

    // wcsph
  particles = std::make_shared<CudaSFParticles>(
      CUDA_SEEPAGEFLOW_APP_PARAMS.max_num, multiVolumeData.pos,
      multiVolumeData.col, multiVolumeData.label, multiVolumeData.mass,multiVolumeData.inertia,
      multiVolumeData.radius, multiVolumeData.cda0asat,
      multiVolumeData.amcamcp);

  CudaGNSearcherPtr searcher;
  searcher = std::make_shared<CudaGNSearcher>(
      CUDA_BOUNDARY_PARAMS.lowest_point, CUDA_BOUNDARY_PARAMS.highest_point,
      particles->MaxSize(), CUDA_BOUNDARY_PARAMS.kernel_radius,
      SearcherParticleType::SEEPAGE_MULTI);

  auto boundary_particles = std::make_shared<CudaBoundaryParticles>(
      boundaryData.pos, boundaryData.label);
  KIRI_LOG_INFO("Number of Boundary Particles = {0}",
                boundary_particles->Size());


  bool adaptive_sub_timestep = true;
  CudaSphSFSolverPtr pSolver;

// wcsph
  pSolver = std::make_shared<CudaWCSphSFSolver>(particles->MaxSize());
  CUDA_SEEPAGEFLOW_PARAMS.solver_type = WCSPH_SOLVER;
  KIRI_LOG_INFO("Current Fluid Solver= WCSPH");

    //dfsph
//     pSolver = std::make_shared<CudaDFSphSFSolver>(particles->MaxSize(),CUDA_SEEPAGEFLOW_PARAMS.dt);
//   CUDA_SEEPAGEFLOW_PARAMS.solver_type = DFSPH_SOLVER;
//   KIRI_LOG_INFO("Current Fluid Solver= DFSPH");

  // bgeo file export & render FPS
  CUDA_SEEPAGEFLOW_APP_PARAMS.bgeo_export = true;

  CudaGNBoundarySearcherPtr boundary_searcher =
      std::make_shared<CudaGNBoundarySearcher>(
          CUDA_BOUNDARY_PARAMS.lowest_point, CUDA_BOUNDARY_PARAMS.highest_point,
          boundary_particles->MaxSize(), CUDA_BOUNDARY_PARAMS.kernel_radius);

  SFSystem = std::make_shared<CudaSFSystem>(
      particles, boundary_particles, pSolver, searcher, boundary_searcher,
      emitter, adaptive_sub_timestep);
}

void SetupExample2() {

  KIRI_LOG_DEBUG("Seepageflow: Example2 SetupParams");
     ExampleName = "seepageflow_dam_wcsph_0.5";
  // export path
  strcpy(CUDA_SEEPAGEFLOW_APP_PARAMS.bgeo_export_folder,
         (String(EXPORT_PATH) + "bgeo/" + ExampleName).c_str());

  // scene config
  auto cuda_lowest_point = make_float3(0.f);
  auto cuda_highest_point = make_float3(1.43f, 2.f, 3.f);
  auto cuda_world_size = cuda_highest_point - cuda_lowest_point;
  auto cuda_world_center = (cuda_highest_point + cuda_lowest_point) / 2.f;
  CUDA_SEEPAGEFLOW_APP_PARAMS.max_num = 200000;

  // sph params
  CUDA_SEEPAGEFLOW_PARAMS.sph_density = 1000.f;
  CUDA_SEEPAGEFLOW_PARAMS.sph_particle_radius = 0.01f;
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
  CUDA_SEEPAGEFLOW_PARAMS.dem_tan_friction_angle = 0.5f;
  CUDA_SEEPAGEFLOW_PARAMS.dem_damping = 0.4f;

  CUDA_SEEPAGEFLOW_PARAMS.sf_c0 = 2.f;
  CUDA_SEEPAGEFLOW_PARAMS.sf_cd = 0.5f;
  CUDA_SEEPAGEFLOW_PARAMS.sf_csat = 0.f;
  CUDA_SEEPAGEFLOW_PARAMS.sf_cmc =2.1f;
  CUDA_SEEPAGEFLOW_PARAMS.sf_cmc_p = 0.01f;

  CUDA_SEEPAGEFLOW_PARAMS.sf_a0 = 2.f;
  CUDA_SEEPAGEFLOW_PARAMS.sf_asat = 1.f;
  CUDA_SEEPAGEFLOW_PARAMS.sf_amc = 2.f;
  CUDA_SEEPAGEFLOW_PARAMS.sf_amc_p = 0.8f;

  CUDA_SEEPAGEFLOW_PARAMS.sf_dry_sand_color =
      make_float3(0.88f, 0.79552f, 0.5984f);
  CUDA_SEEPAGEFLOW_PARAMS.sf_wet_sand_color = make_float3(0.38f, 0.29f, 0.14f);

  CUDA_SEEPAGEFLOW_PARAMS.gravity = make_float3(0.0f, -9.8f, 0.0f);
  CUDA_SEEPAGEFLOW_PARAMS.max_force_factor = 15.f;

  // sph emitter
  CUDA_SPH_EMITTER_PARAMS.enable = true;
  CUDA_SPH_EMITTER_PARAMS.run = false;
  CUDA_SPH_EMITTER_PARAMS.emit_pos =
      make_float3(cuda_world_center.x,
                  cuda_world_center.y,
                  cuda_world_center.z + cuda_world_size.z / 2.5f);
  CUDA_SPH_EMITTER_PARAMS.emit_vel = make_float3(0.f, -5.f, 0.f);
  CUDA_SPH_EMITTER_PARAMS.emit_col = make_float3(127.f, 205.f, 255.f) / 255.f;

  CUDA_SPH_EMITTER_PARAMS.emit_radius = 0.25f;
  CUDA_SPH_EMITTER_PARAMS.emit_width = 0.22f;
  CUDA_SPH_EMITTER_PARAMS.emit_height = 0.18f;
  CUDA_SPH_EMITTER_PARAMS.emit_type = CudaSphEmitterType::CIRCLE;

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

  CUDA_SEEPAGEFLOW_PARAMS.boundary_particle_radius =
      CUDA_SEEPAGEFLOW_PARAMS.sph_particle_radius;

  // init emitter
  CudaEmitterPtr emitter = std::make_shared<CudaEmitter>(
      CUDA_SPH_EMITTER_PARAMS.emit_pos, CUDA_SPH_EMITTER_PARAMS.emit_vel,
      CUDA_SPH_EMITTER_PARAMS.enable);

  // boundary sampling
  BoundaryData boundaryData;
  auto boundaryEmitter = std::make_shared<CudaBoundaryEmitter>();

  boundaryEmitter->BuildWorldBoundary(
      boundaryData, CUDA_BOUNDARY_PARAMS.lowest_point,
      CUDA_BOUNDARY_PARAMS.highest_point,
      CUDA_SEEPAGEFLOW_PARAMS.boundary_particle_radius);

  // material type (SF: unified material; MULTI_SF: multiple types of materials)
  CUDA_SEEPAGEFLOW_PARAMS.sf_type = MULTI_SF;

  // shape sampling
  auto offset2Ground = true;
  SeepageflowMultiVolumeData multiVolumeData;
  auto volumeEmitter = std::make_shared<CudaVolumeEmitter>();

  // multiple sand objects
  Vec_String shape_folders, shape_files;

  // object 1: bunny/bunny.bego
  shape_folders.emplace_back("dam");
  shape_files.emplace_back("dam");

  std::vector<float3> cd_a0_asat;
  std::vector<float2> amc_amcp;

  cd_a0_asat.emplace_back(make_float3(CUDA_SEEPAGEFLOW_PARAMS.sf_cd, CUDA_SEEPAGEFLOW_PARAMS.sf_a0,
                                      CUDA_SEEPAGEFLOW_PARAMS.sf_asat));
  amc_amcp.emplace_back(make_float2(CUDA_SEEPAGEFLOW_PARAMS.sf_amc,
                                    CUDA_SEEPAGEFLOW_PARAMS.sf_amc_p));

  multiVolumeData.sandMinRadius =  CUDA_SEEPAGEFLOW_PARAMS.sph_particle_radius;
//   for (auto i = 0; i < shape_folders.size(); i++) {
//     auto cda0asat = cd_a0_asat[i];
//     auto amcamcp = amc_amcp[i];
//     auto sandShape = ReadBgeoFileForGPU(shape_folders[i], shape_files[i]);

//     volumeEmitter->BuildSeepageflowShapeMultiVolume(
//         multiVolumeData, sandShape, CUDA_SEEPAGEFLOW_PARAMS.sf_dry_sand_color,
//         CUDA_SEEPAGEFLOW_PARAMS.dem_density, cda0asat, amcamcp, offset2Ground,
//         CUDA_BOUNDARY_PARAMS.lowest_point.y, make_float2(0.75f, 2.f));

//     KIRI_LOG_DEBUG(
//         "Object({0}) Params: Cd A0 Asat Amc Amcp = {1}, {2}, {3}, {4}, {5}",
//         i + 1, cda0asat.x, cda0asat.y, cda0asat.z, amcamcp.x, amcamcp.y);
//   }

  // dt
  CUDA_SEEPAGEFLOW_PARAMS.dt = 0.5f * multiVolumeData.sandMinRadius /
                               std::sqrtf(CUDA_SEEPAGEFLOW_PARAMS.dem_young /
                                          CUDA_SEEPAGEFLOW_PARAMS.dem_density);
  KIRI_LOG_INFO(
      "Number of total sand particles = {0}; minimum radius ={1}; dt ={2}",
      multiVolumeData.pos.size(), multiVolumeData.sandMinRadius,
      CUDA_SEEPAGEFLOW_PARAMS.dt);

  // spatial searcher & particles
  CudaSFParticlesPtr particles;

    // wcsph
  particles = std::make_shared<CudaSFParticles>(
      CUDA_SEEPAGEFLOW_APP_PARAMS.max_num, multiVolumeData.pos,
      multiVolumeData.col, multiVolumeData.label, multiVolumeData.mass,multiVolumeData.inertia,
      multiVolumeData.radius, multiVolumeData.cda0asat,
      multiVolumeData.amcamcp);

  CudaGNSearcherPtr searcher;
  searcher = std::make_shared<CudaGNSearcher>(
      CUDA_BOUNDARY_PARAMS.lowest_point, CUDA_BOUNDARY_PARAMS.highest_point,
      particles->MaxSize(), CUDA_BOUNDARY_PARAMS.kernel_radius,
      SearcherParticleType::SEEPAGE_MULTI);

  auto boundary_particles = std::make_shared<CudaBoundaryParticles>(
      boundaryData.pos, boundaryData.label);
  KIRI_LOG_INFO("Number of Boundary Particles = {0}",
                boundary_particles->Size());


  bool adaptive_sub_timestep = true;
  CudaSphSFSolverPtr pSolver;

// wcsph
  pSolver = std::make_shared<CudaWCSphSFSolver>(particles->MaxSize());
  CUDA_SEEPAGEFLOW_PARAMS.solver_type = WCSPH_SOLVER;
  KIRI_LOG_INFO("Current Fluid Solver= WCSPH");

  // bgeo file export & render FPS
  CUDA_SEEPAGEFLOW_APP_PARAMS.bgeo_export = true;

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

  SetupExample1();

  // abc exporter params
  auto AbcDtScale = 120.f * RenderInterval;
  auto AbcExportorPath = String(EXPORT_PATH) + "abc/" + ExampleName;
  auto AbcExportorPos = AbcExportorPath + "/" + "pos" + ".abc";
  auto AbcExportorScale = AbcExportorPath + "/" + "scale" + ".abc";
  auto AbcExportorColor = AbcExportorPath + "/" + "color" + ".abc";

  std::error_code ErrorCode;
  std::filesystem::create_directories(AbcExportorPath, ErrorCode);

  auto AbcPosData = std::make_shared<ParticlesAlembicManager>(
      AbcExportorPos, RenderInterval * AbcDtScale, "particle_pos");
  auto AbcScaleData = std::make_shared<ParticlesAlembicManager>(
      AbcExportorScale, RenderInterval * AbcDtScale, "particle_scale");
  auto AbcColorData = std::make_shared<ParticlesAlembicManager>(
      AbcExportorColor, RenderInterval * AbcDtScale, "particle_cd");

  CUDA_SEEPAGEFLOW_APP_PARAMS.run = true;

  while (CUDA_SEEPAGEFLOW_APP_PARAMS.run) {
    if (CUDA_SEEPAGEFLOW_APP_PARAMS.run &&
        SimCount < TotalFrameNumber + RunLiquidNumber) {
      if (SimCount == RunLiquidNumber) {
        // export bgeo file
        CUDA_SEEPAGEFLOW_APP_PARAMS.bgeo_export =
            CUDA_SEEPAGEFLOW_APP_PARAMS.run;
        if (CUDA_SPH_EMITTER_PARAMS.enable)
          CUDA_SPH_EMITTER_PARAMS.run = CUDA_SEEPAGEFLOW_APP_PARAMS.run;
      }

      if (SFSystem->GetAdaptiveSubTimeStep()) {
        float remaining_time = RenderInterval;
        KIRI_LOG_INFO("Simulation Frame={0}, Adaptive Sub-Simulation",
                      ++SimCount);
        PerFrameTimer.Restart();
        size_t i = 0;
        while (remaining_time > KIRI_EPSILON) {
          KIRI_LOG_INFO(
              "Current Sub-Simulation RemainTime={0},Sub-Simulation Step={1}",
              remaining_time, ++i);
          SFSystem->UpdateSystem(remaining_time);
          remaining_time -=
              remaining_time /
              static_cast<float>(SFSystem->GetNumOfSubTimeSteps());
        }
      } else {
        auto numOfSubTimeSteps = SFSystem->GetNumOfSubTimeSteps();
        KIRI_LOG_INFO("Simulation Frame={0}, Sub-Simulation Total Number={1}",
                      ++SimCount, numOfSubTimeSteps);

        PerFrameTimer.Restart();
        for (size_t i = 0; i < numOfSubTimeSteps; i++) {
          KIRI_LOG_INFO("Current Sub-Simulation/ Total Number ={0}/{1}", i + 1,
                        numOfSubTimeSteps);
          SFSystem->UpdateSystem(RenderInterval);
        }
      }

      KIRI_LOG_INFO("Time Per Frame={0}", PerFrameTimer.Elapsed());
      TotalFrameTime += PerFrameTimer.Elapsed();

      if (CUDA_SEEPAGEFLOW_APP_PARAMS.bgeo_export) {
        auto particles = SFSystem->GetSFParticles();
        // ExportBgeoFileCUDA(
        //     CUDA_SEEPAGEFLOW_APP_PARAMS.bgeo_export_folder,
        //     UInt2Str4Digit(SimCount - RunLiquidNumber),
        //     particles->GetPosPtr(),
        //     particles->GetColPtr(),
        //     particles->GetRadiusPtr(),
        //     particles->GetLabelPtr(),
        //     particles->Size());

        AbcPosData->SubmitCurrentStatusFloat3(particles->GetPosPtr(),
                                              particles->Size());

        AbcScaleData->SubmitCurrentStatusFloat(particles->GetRadiusPtr(),
                                               particles->Size());

        AbcColorData->SubmitCurrentStatusFloat3(particles->GetColPtr(),
                                                particles->Size());
      }
    } else if (CUDA_SEEPAGEFLOW_APP_PARAMS.run) {
      CUDA_SEEPAGEFLOW_APP_PARAMS.run = false;

      KIRI_LOG_INFO("Average Per Frame={0}",
                    TotalFrameTime / (TotalFrameNumber + RunLiquidNumber));
    }
  }

  return;
}
