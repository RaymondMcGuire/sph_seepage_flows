/*
 * @Author: Xu.WANG
 * @Date: 2021-02-04 12:36:10
 * @LastEditTime: 2021-03-20 00:26:31
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\emitter\cuda_emitter.cuh
 */

#ifndef _CUDA_EMITTER_CUH_
#define _CUDA_EMITTER_CUH_

#pragma once

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>

namespace KIRI
{

    class CudaEmitter
    {
    public:
        explicit CudaEmitter()
            : CudaEmitter(
                  make_float3(0.f),
                  make_float3(1.f, 0.f, 0.f),
                  false)

        {
        }

        explicit CudaEmitter(
            float3 emitPosition,
            float3 emitVelocity,
            bool enable)
            : mEmitPosition(emitPosition),
              mEmitVelocity(emitVelocity),
              mEmitAxis1(make_float3(1.f)),
              bBuild(false),
              bEnable(enable)
        {
            mSamples.clear();

            float3 axis = normalize(mEmitVelocity);

            if (abs(axis.x) == 1.f && abs(axis.y) == 0.f && abs(axis.z) == 0.f)
            {
                mEmitAxis1 = normalize(cross(axis, make_float3(0.f, 1.f, 0.f)));
            }
            else
            {
                mEmitAxis1 = normalize(cross(axis, make_float3(1.f, 0.f, 0.f)));
            }

            mEmitAxis2 = normalize(cross(axis, mEmitAxis1));
        }

        CudaEmitter(const CudaEmitter &) = delete;
        CudaEmitter &operator=(const CudaEmitter &) = delete;
        virtual ~CudaEmitter() noexcept {}

        Vec_Float3 Emit();

        inline void SetEmitterStatus(const bool enable) { bEnable = enable; }

        void BuildSquareEmitter(float particleRadius, float emitterRadius);
        void BuildCircleEmitter(float particleRadius, float emitterRadius);
        void BuildRectangleEmitter(float particleRadius, float emitterWidth, float emitterHeight);

        inline constexpr bool GetEmitterStatus() const { return bEnable; }
        inline size_t GetNumOfEmitterPoints() const { return mSamples.size(); }
        inline constexpr float3 GetEmitterPosition() const { return mEmitPosition; }
        inline constexpr float3 GetEmitterVelocity() const { return mEmitVelocity; }

    private:
        bool bEnable, bBuild;
        Vec_Float2 mSamples;
        float3 mEmitPosition, mEmitVelocity;
        float3 mEmitAxis1, mEmitAxis2;
    };

    typedef SharedPtr<CudaEmitter> CudaEmitterPtr;
} // namespace KIRI

#endif