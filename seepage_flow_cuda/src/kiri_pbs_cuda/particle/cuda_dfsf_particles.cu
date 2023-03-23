/*** 
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-03-23 15:57:38
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-03-23 15:57:51
 * @FilePath: \sph_seepage_flows\seepage_flow_cuda\src\kiri_pbs_cuda\particle\cuda_dfsf_particles.cu
 * @Description: 
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved. 
 */

#include <kiri_pbs_cuda/particle/cuda_dfsf_particles.cuh>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

namespace KIRI {

void CudaDFSFParticles::AdvectFluidVel(const float dt) {
 

  thrust::transform(
      thrust::device, mVel.Data(), mVel.Data() + Size(), mAcc.Data(),
      mVel.Data(), [dt] __host__ __device__(const float3 &lv, const float3 &a) {
        return lv + dt * a;
      });

  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();
}


} // namespace KIRI
