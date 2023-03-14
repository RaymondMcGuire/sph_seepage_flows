/*
 * @Author: Xu.WANG
 * @Date: 2021-02-04 12:36:10
 * @LastEditTime: 2021-04-04 01:17:11
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\particle\cuda_boundary_particles.cuh
 */

#ifndef _CUDA_BOUNDARY_PARTICLES_CUH_
#define _CUDA_BOUNDARY_PARTICLES_CUH_

#pragma once

#include <kiri_pbs_cuda/particle/cuda_particles.cuh>

namespace KIRI
{
	class CudaBoundaryParticles final : public CudaParticles
	{
	public:
		explicit CudaBoundaryParticles::CudaBoundaryParticles(
			const Vec_Float3 &p)
			: CudaParticles(p),
			  mVolume(p.size()),
			  mLabel(p.size()) {}

		explicit CudaBoundaryParticles::CudaBoundaryParticles(
			const Vec_Float3 &p, const Vec_SizeT &label)
			: CudaParticles(p),
			  mVolume(p.size()),
			  mLabel(p.size())
		{
			KIRI_CUCALL(cudaMemcpy(mLabel.Data(), &label[0], sizeof(size_t) * label.size(), cudaMemcpyHostToDevice));
		}

		CudaBoundaryParticles(const CudaBoundaryParticles &) = delete;
		CudaBoundaryParticles &operator=(const CudaBoundaryParticles &) = delete;

		float *GetVolumePtr() const { return mVolume.Data(); }
		size_t *GetLabelPtr() const { return mLabel.Data(); }

		virtual ~CudaBoundaryParticles() noexcept {}

	protected:
		CudaArray<float> mVolume;
		CudaArray<size_t> mLabel;
	};

	typedef SharedPtr<CudaBoundaryParticles> CudaBoundaryParticlesPtr;
} // namespace KIRI

#endif