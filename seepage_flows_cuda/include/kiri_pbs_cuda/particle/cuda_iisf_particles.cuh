/*
 * @Author: Xu.WANG
 * @Date: 2021-02-04 12:36:10
 * @LastEditTime: 2021-08-18 09:20:19
 * @LastEditors: Xu.WANG
 * @Description: Seepage flow particles
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\particle\cuda_iisf_particles.cuh
 */

#ifndef _CUDA_IISPH_SEEPAGEFLOW_PARTICLES_CUH_
#define _CUDA_IISPH_SEEPAGEFLOW_PARTICLES_CUH_

#pragma once

#include <kiri_pbs_cuda/particle/cuda_sf_particles.cuh>
namespace KIRI
{
	class CudaIISFParticles : public CudaSFParticles
	{
	public:
		explicit CudaIISFParticles::CudaIISFParticles(
			const size_t numOfMaxParticles)
			: CudaSFParticles(numOfMaxParticles),
			  mAii(numOfMaxParticles),
			  mDii(numOfMaxParticles),
			  mDijPj(numOfMaxParticles),
			  mDensityAdv(numOfMaxParticles),
			  mLastPressure(numOfMaxParticles),
			  mDensityError(numOfMaxParticles),
			  mPressureAcc(numOfMaxParticles)
		{
		}

		explicit CudaIISFParticles::CudaIISFParticles(
			const Vec_Float3 &p,
			const Vec_Float3 &col,
			const Vec_SizeT &label,
			const Vec_Float &mass,
			const Vec_Float &radius)
			: CudaSFParticles(p, col, label, mass, radius),
			  mAii(p.size()),
			  mDii(p.size()),
			  mDijPj(p.size()),
			  mDensityAdv(p.size()),
			  mLastPressure(p.size()),
			  mDensityError(p.size()),
			  mPressureAcc(p.size())
		{
		}

		explicit CudaIISFParticles::CudaIISFParticles(
			const size_t numOfMaxParticles,
			const Vec_Float3 &p,
			const Vec_Float3 &col,
			const Vec_SizeT &label,
			const Vec_Float &mass,
			const Vec_Float &radius)
			: CudaSFParticles(numOfMaxParticles, p, col, label, mass, radius),
			  mAii(numOfMaxParticles),
			  mDii(numOfMaxParticles),
			  mDijPj(numOfMaxParticles),
			  mDensityAdv(numOfMaxParticles),
			  mLastPressure(numOfMaxParticles),
			  mDensityError(numOfMaxParticles),
			  mPressureAcc(numOfMaxParticles)
		{
		}

		explicit CudaIISFParticles::CudaIISFParticles(
			const size_t numOfMaxParticles,
			const Vec_Float3 &p,
			const Vec_Float3 &col,
			const Vec_SizeT &label,
			const Vec_Float &mass,
			const Vec_Float &radius,
			const Vec_Float3 &cda0asat,
			const Vec_Float2 &amcamcp)
			: CudaSFParticles(numOfMaxParticles, p, col, label, mass, radius, cda0asat, amcamcp),
			  mAii(numOfMaxParticles),
			  mDii(numOfMaxParticles),
			  mDijPj(numOfMaxParticles),
			  mDensityAdv(numOfMaxParticles),
			  mLastPressure(numOfMaxParticles),
			  mDensityError(numOfMaxParticles),
			  mPressureAcc(numOfMaxParticles)
		{
		}

		CudaIISFParticles(const CudaIISFParticles &) = delete;
		CudaIISFParticles &operator=(const CudaIISFParticles &) = delete;

		virtual ~CudaIISFParticles() noexcept {}

		void PredictVelAdvect(const float dt);
		virtual void Advect(const float dt, const float damping) override;

		inline float *GetAiiPtr() const { return mAii.Data(); }
		inline float3 *GetDiiPtr() const { return mDii.Data(); }
		inline float3 *GetDijPjPtr() const { return mDijPj.Data(); }
		inline float *GetDensityAdvPtr() const { return mDensityAdv.Data(); }
		inline float *GetLastPressurePtr() const { return mLastPressure.Data(); }
		inline float *GetDensityErrorPtr() const { return mDensityError.Data(); }
		inline float3 *GetPressureAccPtr() const { return mPressureAcc.Data(); }

	protected:
		CudaArray<float> mAii;
		CudaArray<float3> mDii;
		CudaArray<float3> mDijPj;
		CudaArray<float> mDensityAdv;
		CudaArray<float> mLastPressure;
		CudaArray<float> mDensityError;
		CudaArray<float3> mPressureAcc;
	};

	typedef SharedPtr<CudaIISFParticles> CudaIISFParticlesPtr;
} // namespace KIRI

#endif