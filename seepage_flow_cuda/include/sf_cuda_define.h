/*** 
 * @Author: Xu.WANG
 * @Date: 2021-07-30 11:10:34
 * @LastEditTime: 2021-08-21 19:28:11
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \sph_seepage_flows\seepage_flows_cuda\include\sf_cuda_define.h
 */
#ifndef _SF_CUDA_DEFINE_H_
#define _SF_CUDA_DEFINE_H_

#include <kiri_pbs_cuda/data/cuda_seepageflow_params.h>

#include <kiri_pbs_cuda/solver/seepageflow/cuda_wcsph_sf_solver.cuh>

#include <kiri_pbs_cuda/emitter/cuda_volume_emitter.cuh>
#include <kiri_pbs_cuda/emitter/cuda_boundary_emitter.cuh>

#include <kiri_pbs_cuda/solver/seepageflow/cuda_sf_utils.cuh>

#include <kiri_pbs_cuda/emitter/cuda_emitter.cuh>
#include <kiri_pbs_cuda/searcher/cuda_neighbor_searcher.cuh>
#include <kiri_pbs_cuda/system/cuda_sf_system.cuh>

#endif // _SF_CUDA_DEFINE_H_
