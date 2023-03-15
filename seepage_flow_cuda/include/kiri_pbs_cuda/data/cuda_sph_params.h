/***
 * @Author: Xu.WANG
 * @Date: 2021-02-10 15:29:35
 * @LastEditTime: 2021-07-18 19:01:34
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\data\cuda_sph_params.h
 */

#ifndef _CUDA_SPH_PARAMS_H_
#define _CUDA_SPH_PARAMS_H_

#pragma once

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>

namespace KIRI {
enum CudaSphSolverType { SPH_SOLVER, WCSPH_SOLVER, IISPH_SOLVER };

struct CudaSphParams {
  float rest_mass;
  float rest_density;
  float particle_radius;
  float kernel_radius;

  bool atf_visc;
  float stiff;
  float visc;
  float nu;
  float bnu;

  bool sta_akinci13;
  float st_gamma;
  float a_beta;

  float3 gravity;
  float dt;

  CudaSphSolverType solver_type;
};

enum CudaSphEmitterType { SQUARE, CIRCLE, RECTANGLE };

struct CudaSphEmitterParams {
  bool enable = false;
  bool run = false;

  float3 emit_pos;
  float3 emit_vel;
  float3 emit_col;

  CudaSphEmitterType emit_type = CudaSphEmitterType::SQUARE;

  float emit_radius = 0.0;
  float emit_width = 0.0;
  float emit_height = 0.0;
};

struct CudaSphAppParams {
  size_t max_num = 100000;
  bool run = false;
  bool run_offline = false;

  int scene_data_idx = 0;
  char bgeo_file_name[32] = "default";
  bool bgeo_export = false;
};

extern CudaSphParams CUDA_SPH_PARAMS;
extern CudaSphEmitterParams CUDA_SPH_EMITTER_PARAMS;
extern CudaSphAppParams CUDA_SPH_APP_PARAMS;

} // namespace KIRI

#endif