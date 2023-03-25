/*** 
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-03-22 16:03:20
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-03-25 23:15:33
 * @FilePath: \sph_seepage_flows\seepage_flow_cuda\include\kiri_pbs_cuda\particle\cuda_dfsf_particles.cuh
 * @Description: 
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved. 
 */
#ifndef _CUDA_DFSF_PARTICLES_CUH_
#define _CUDA_DFSF_PARTICLES_CUH_

#pragma once

#include <kiri_pbs_cuda/particle/cuda_sf_particles.cuh>
namespace KIRI {

class CudaDFSFParticles : public CudaSFParticles {
public:

  explicit CudaDFSFParticles::CudaDFSFParticles(const size_t numOfMaxParticles,
                                            const Vec_Float3 &p,
                                            const Vec_Float3 &col,
                                            const Vec_SizeT &label,
                                            const Vec_Float &mass,
                                             const Vec_Float &inertia,
                                            const Vec_Float &radius)
      : CudaSFParticles(numOfMaxParticles, p, col, label, mass,inertia, radius),
       mAlpha(numOfMaxParticles), mStiff(numOfMaxParticles),
        mWarmStiff(numOfMaxParticles), mVelMag(numOfMaxParticles),
        mDensityAdv(numOfMaxParticles), mDensityError(numOfMaxParticles) 
        { }

  explicit CudaDFSFParticles::CudaDFSFParticles(
      const size_t numOfMaxParticles, const Vec_Float3 &p,
      const Vec_Float3 &col, const Vec_SizeT &label, const Vec_Float &mass,const Vec_Float &inertia,
      const Vec_Float &radius, const Vec_Float3 &cda0asat,
      const Vec_Float2 &amcamcp)
      : CudaSFParticles(numOfMaxParticles, p,col,label,mass,inertia,radius,cda0asat,amcamcp),
       mAlpha(numOfMaxParticles), mStiff(numOfMaxParticles),
        mWarmStiff(numOfMaxParticles), mVelMag(numOfMaxParticles),
        mDensityAdv(numOfMaxParticles), mDensityError(numOfMaxParticles)  {

  }

  CudaDFSFParticles(const CudaDFSFParticles &) = delete;
  CudaDFSFParticles &operator=(const CudaDFSFParticles &) = delete;

  virtual ~CudaDFSFParticles() noexcept {}


    inline float *GetAlphaPtr() const { return mAlpha.Data(); }
  inline float *GetStiffPtr() const { return mStiff.Data(); }
  inline float *GetWarmStiffPtr() const { return mWarmStiff.Data(); }
  inline float *GetVelMagPtr() const { return mVelMag.Data(); }
  inline float *GetDensityAdvPtr() const { return mDensityAdv.Data(); }
  inline float *GetDensityErrorPtr() const { return mDensityError.Data(); }


  void AdvectFluidVel(const float dt);
  virtual void Advect(const float dt, const float damping)override;

protected:

  CudaArray<float> mAlpha;
  CudaArray<float> mStiff;
  CudaArray<float> mWarmStiff;
  CudaArray<float> mVelMag;
  CudaArray<float> mDensityAdv;
  CudaArray<float> mDensityError;

};

typedef SharedPtr<CudaDFSFParticles> CudaDFSFParticlesPtr;
} // namespace KIRI

#endif