/*
 * @Author: Xu.WANG
 * @Date: 2021-02-04 12:36:10
 * @LastEditTime: 2021-08-18 08:59:13
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\emitter\cuda_volume_emitter.cuh
 */

#ifndef _CUDA_VOLUME_EMITTER_CUH_
#define _CUDA_VOLUME_EMITTER_CUH_

#pragma once

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>

namespace KIRI
{

    struct SphVolumeData
    {
        Vec_Float3 pos;
        Vec_Float3 col;
    };

    struct MultiSphRen14VolumeData
    {
        Vec_Float3 pos;
        Vec_Float3 col;
        Vec_Float mass;
        Vec_SizeT phaseLabel;
    };

    struct MultiSphYan16VolumeData
    {
        Vec_Float3 pos;
        Vec_Float3 col;
        Vec_Float mass;
        Vec_SizeT phaseLabel;
        Vec_SizeT phaseType;
    };

    struct DemVolumeData
    {
        Vec_Float3 pos;
        Vec_Float3 col;
        Vec_Float mass;
    };

    struct DemShapeVolumeData
    {
        float minRadius;
        Vec_Float3 pos;
        Vec_Float3 col;
        Vec_Float mass;
        Vec_Float radius;
    };

    struct SeepageflowVolumeData
    {
        float sandMinRadius;
        Vec_Float3 pos;
        Vec_Float3 col;
        Vec_Float mass;
        Vec_Float radius;
        Vec_SizeT label;
    };

    struct SeepageflowMultiVolumeData
    {
        float sandMinRadius;
        Vec_Float3 pos;
        Vec_Float3 col;
        Vec_Float mass;
        Vec_Float radius;
        Vec_SizeT label;
        Vec_Float2 amcamcp;
        Vec_Float3 cda0asat;
    };

    class CudaVolumeEmitter
    {
    public:
        explicit CudaVolumeEmitter(
            bool enable = true)
            : bEnable(enable)
        {
        }

        CudaVolumeEmitter(const CudaVolumeEmitter &) = delete;
        CudaVolumeEmitter &operator=(const CudaVolumeEmitter &) = delete;
        virtual ~CudaVolumeEmitter() noexcept {}

        void BuildSphVolume(SphVolumeData &data, float3 lowest, int3 vsize, float particleRadius, float3 color);
        void BuildUniDemVolume(DemVolumeData &data, float3 lowest, int3 vsize, float particleRadius, float3 color, float mass, float jitter = 0.001f);
        void BuildDemShapeVolume(DemShapeVolumeData &data, Vec_Float4 shape, float3 color, float density);

        void BuildSeepageflowBoxVolume(SeepageflowVolumeData &data, float3 lowest, int3 vsize, float particleRadius, float3 color, float mass, size_t label, float jitter = 0.001f);
        void BuildSeepageflowShapeVolume(SeepageflowVolumeData &data, Vec_Float4 shape, float3 color, float sandDensity, bool offsetY = false, float worldLowestY = 0.f);
        void BuildSeepageflowShapeMultiVolume(SeepageflowMultiVolumeData &data, Vec_Float4 shape, float3 color, float sandDensity, float3 cda0asat, float2 amcamcp, bool offsetY = false, float worldLowestY = 0.f);

        void BuildMultiSphRen14Volume(MultiSphRen14VolumeData &data, float3 lowest, int3 vsize, float particleRadius, float3 color, float mass, size_t phaseIdx);
        void BuildMultiSphYan16Volume(MultiSphYan16VolumeData &data, float3 lowest, int3 vsize, float particleRadius, float3 color, float mass, size_t phaseIdx, size_t phaseType);

        inline constexpr bool GetEmitterStatus() const { return bEnable; }

    private:
        bool bEnable;
    };

    typedef SharedPtr<CudaVolumeEmitter> CudaVolumeEmitterPtr;
} // namespace KIRI

#endif