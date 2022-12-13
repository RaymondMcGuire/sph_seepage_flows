/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-04-17 15:08:41
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-12-13 22:34:12
 * @FilePath: \sph_seepage_flows\seepage_flows_cuda\include\kiri_pbs_cuda\particle\cuda_sf_particles.cuh
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#ifndef _CUDA_SEEPAGEFLOW_PARTICLES_CUH_
#define _CUDA_SEEPAGEFLOW_PARTICLES_CUH_

#pragma once

#include <kiri_pbs_cuda/particle/cuda_particles.cuh>
namespace KIRI
{
	typedef thrust::tuple<size_t, float3, float3> SFDataType;
	struct AccDampingForSand
	{
		float mDt;
		float mDamping;
		__host__ __device__ AccDampingForSand(
			const float dt,
			const float damping)
			: mDt(dt),
			  mDamping(damping) {}

		__host__ __device__ float3 operator()(const SFDataType &data) const
		{
			size_t label = data.get<0>();
			float3 acc = data.get<1>();
			float3 lv = data.get<2>();

			if (label == 1)
				return acc * (make_float3(1.f) - mDamping * sgn(acc * (lv + 0.5f * mDt * acc)));
			else
				return acc;
		}
	};

	class CudaSFParticles : public CudaParticles
	{
	public:
		explicit CudaSFParticles::CudaSFParticles(
			const size_t numOfMaxParticles)
			: CudaParticles(numOfMaxParticles),
			  mId(numOfMaxParticles),
			  mLabel(numOfMaxParticles),
			  mRadius(numOfMaxParticles),
			  mVel(numOfMaxParticles),
			  mAcc(numOfMaxParticles),
			  mCol(numOfMaxParticles),
			  mPressure(numOfMaxParticles),
			  mDensity(numOfMaxParticles),
			  mMass(numOfMaxParticles),
			  mVoidage(numOfMaxParticles),
			  mSaturation(numOfMaxParticles),
			  mMaxSaturation(numOfMaxParticles),
			  mAvgFlowVel(numOfMaxParticles),
			  mAvgDragForce(numOfMaxParticles),
			  mAdhesionForce(numOfMaxParticles),
			  mAvgAdhesionForce(numOfMaxParticles),
			  mCdA0Asat(numOfMaxParticles),
			  mAmcAmcp(numOfMaxParticles)
		{
		}

		explicit CudaSFParticles::CudaSFParticles(
			const Vec_SizeT &id,
			const Vec_Float3 &p,
			const Vec_Float3 &col,
			const Vec_SizeT &label,
			const Vec_Float &mass,
			const Vec_Float &radius)
			: CudaParticles(p),
			  mId(p.size()),
			  mLabel(p.size()),
			  mRadius(p.size()),
			  mVel(p.size()),
			  mAcc(p.size()),
			  mCol(p.size()),
			  mPressure(p.size()),
			  mDensity(p.size()),
			  mMass(p.size()),
			  mVoidage(p.size()),
			  mSaturation(p.size()),
			  mMaxSaturation(p.size()),
			  mAvgFlowVel(p.size()),
			  mAvgDragForce(p.size()),
			  mAdhesionForce(p.size()),
			  mAvgAdhesionForce(p.size()),
			  mCdA0Asat(p.size()),
			  mAmcAmcp(p.size())
		{
			KIRI_CUCALL(cudaMemcpy(mId.Data(), &id[0], sizeof(size_t) * id.size(), cudaMemcpyHostToDevice));
			KIRI_CUCALL(cudaMemcpy(mCol.Data(), &col[0], sizeof(float3) * col.size(), cudaMemcpyHostToDevice));
			KIRI_CUCALL(cudaMemcpy(mRadius.Data(), &radius[0], sizeof(float) * radius.size(), cudaMemcpyHostToDevice));
			KIRI_CUCALL(cudaMemcpy(mMass.Data(), &mass[0], sizeof(float) * mass.size(), cudaMemcpyHostToDevice));
			KIRI_CUCALL(cudaMemcpy(mLabel.Data(), &label[0], sizeof(size_t) * label.size(), cudaMemcpyHostToDevice));
		}

		explicit CudaSFParticles::CudaSFParticles(
			const size_t numOfMaxParticles,
			const Vec_SizeT &id,
			const Vec_Float3 &p,
			const Vec_Float3 &col,
			const Vec_SizeT &label,
			const Vec_Float &mass,
			const Vec_Float &radius)
			: CudaParticles(numOfMaxParticles, p),
			  mId(numOfMaxParticles),
			  mLabel(numOfMaxParticles),
			  mRadius(numOfMaxParticles),
			  mVel(numOfMaxParticles),
			  mAcc(numOfMaxParticles),
			  mCol(numOfMaxParticles),
			  mPressure(numOfMaxParticles),
			  mDensity(numOfMaxParticles),
			  mMass(numOfMaxParticles),
			  mVoidage(numOfMaxParticles),
			  mSaturation(numOfMaxParticles),
			  mMaxSaturation(numOfMaxParticles),
			  mAvgFlowVel(numOfMaxParticles),
			  mAvgDragForce(numOfMaxParticles),
			  mAdhesionForce(numOfMaxParticles),
			  mAvgAdhesionForce(numOfMaxParticles),
			  mCdA0Asat(numOfMaxParticles),
			  mAmcAmcp(numOfMaxParticles)
		{
			KIRI_CUCALL(cudaMemcpy(mId.Data(), &id[0], sizeof(size_t) * id.size(), cudaMemcpyHostToDevice));
			KIRI_CUCALL(cudaMemcpy(mCol.Data(), &col[0], sizeof(float3) * col.size(), cudaMemcpyHostToDevice));
			KIRI_CUCALL(cudaMemcpy(mRadius.Data(), &radius[0], sizeof(float) * radius.size(), cudaMemcpyHostToDevice));
			KIRI_CUCALL(cudaMemcpy(mMass.Data(), &mass[0], sizeof(float) * mass.size(), cudaMemcpyHostToDevice));
			KIRI_CUCALL(cudaMemcpy(mLabel.Data(), &label[0], sizeof(size_t) * label.size(), cudaMemcpyHostToDevice));
		}

		explicit CudaSFParticles::CudaSFParticles(
			const size_t numOfMaxParticles,
			const Vec_SizeT &id,
			const Vec_Float3 &p,
			const Vec_Float3 &col,
			const Vec_SizeT &label,
			const Vec_Float &mass,
			const Vec_Float &radius,
			const Vec_Float3 &cda0asat,
			const Vec_Float2 &amcamcp)
			: CudaParticles(numOfMaxParticles, p),
			  mId(numOfMaxParticles),
			  mLabel(numOfMaxParticles),
			  mRadius(numOfMaxParticles),
			  mVel(numOfMaxParticles),
			  mAcc(numOfMaxParticles),
			  mCol(numOfMaxParticles),
			  mPressure(numOfMaxParticles),
			  mDensity(numOfMaxParticles),
			  mMass(numOfMaxParticles),
			  mVoidage(numOfMaxParticles),
			  mSaturation(numOfMaxParticles),
			  mMaxSaturation(numOfMaxParticles),
			  mAvgFlowVel(numOfMaxParticles),
			  mAvgDragForce(numOfMaxParticles),
			  mAdhesionForce(numOfMaxParticles),
			  mAvgAdhesionForce(numOfMaxParticles),
			  mCdA0Asat(numOfMaxParticles),
			  mAmcAmcp(numOfMaxParticles)
		{
			KIRI_CUCALL(cudaMemcpy(mId.Data(), &id[0], sizeof(size_t) * id.size(), cudaMemcpyHostToDevice));
			KIRI_CUCALL(cudaMemcpy(mCol.Data(), &col[0], sizeof(float3) * col.size(), cudaMemcpyHostToDevice));
			KIRI_CUCALL(cudaMemcpy(mRadius.Data(), &radius[0], sizeof(float) * radius.size(), cudaMemcpyHostToDevice));
			KIRI_CUCALL(cudaMemcpy(mMass.Data(), &mass[0], sizeof(float) * mass.size(), cudaMemcpyHostToDevice));
			KIRI_CUCALL(cudaMemcpy(mLabel.Data(), &label[0], sizeof(size_t) * label.size(), cudaMemcpyHostToDevice));
			KIRI_CUCALL(cudaMemcpy(mCdA0Asat.Data(), &cda0asat[0], sizeof(float3) * cda0asat.size(), cudaMemcpyHostToDevice));
			KIRI_CUCALL(cudaMemcpy(mAmcAmcp.Data(), &amcamcp[0], sizeof(float2) * amcamcp.size(), cudaMemcpyHostToDevice));
		}

		CudaSFParticles(const CudaSFParticles &) = delete;
		CudaSFParticles &operator=(const CudaSFParticles &) = delete;

		inline size_t *GetIdPtr() const { return mId.Data(); }
		size_t *GetLabelPtr() const { return mLabel.Data(); }
		float *GetRadiusPtr() const { return mRadius.Data(); }

		float3 *GetVelPtr() const { return mVel.Data(); }
		float3 *GetAccPtr() const { return mAcc.Data(); }
		float3 *GetColPtr() const { return mCol.Data(); }
		float *GetPressurePtr() const { return mPressure.Data(); }
		float *GetDensityPtr() const { return mDensity.Data(); }
		float *GetMassPtr() const { return mMass.Data(); }

		float *GetVoidagePtr() const { return mVoidage.Data(); }
		float *GetSaturationPtr() const { return mSaturation.Data(); }
		float *GetMaxSaturationPtr() const { return mMaxSaturation.Data(); }
		float3 *GetAvgFlowVelPtr() const { return mAvgFlowVel.Data(); }
		float3 *GetAvgDragForcePtr() const { return mAvgDragForce.Data(); }

		float3 *GetAdhesionForcePtr() const { return mAdhesionForce.Data(); }
		float3 *GetAvgAdhesionForcePtr() const { return mAvgAdhesionForce.Data(); }

		float3 *GetCdA0AsatPtr() const { return mCdA0Asat.Data(); }
		float2 *GetAmcAmcpPtr() const { return mAmcAmcp.Data(); }

		virtual ~CudaSFParticles() noexcept {}

		virtual void Advect(const float dt, const float damping);

		void AddSphParticles(Vec_Float3 pos, float3 col, float3 vel, float mass, float radius);

	protected:
		CudaArray<size_t> mId;
		CudaArray<size_t> mLabel;
		CudaArray<float> mRadius;

		CudaArray<float3> mVel;
		CudaArray<float3> mAcc;
		CudaArray<float3> mCol;
		CudaArray<float> mMass;
		CudaArray<float> mDensity;
		CudaArray<float> mPressure;

		CudaArray<float> mVoidage;
		CudaArray<float> mSaturation;
		CudaArray<float> mMaxSaturation;
		CudaArray<float3> mAvgFlowVel;
		CudaArray<float3> mAvgDragForce;
		CudaArray<float3> mAdhesionForce;
		CudaArray<float3> mAvgAdhesionForce;

		CudaArray<float3> mCdA0Asat;
		CudaArray<float2> mAmcAmcp;
	};

	typedef SharedPtr<CudaSFParticles> CudaSFParticlesPtr;
} // namespace KIRI

#endif