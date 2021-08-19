/*** 
 * @Author: Xu.WANG
 * @Date: 2021-03-19 22:04:26
 * @LastEditTime: 2021-03-19 22:34:27
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\emitter\cuda_emitter.cpp
 */

#include <kiri_pbs_cuda/emitter/cuda_emitter.cuh>
namespace KIRI
{

    Vec_Float3 CudaEmitter::Emit()
    {
        KIRI_PBS_ASSERT(bBuild);

        Vec_Float3 emitPoints;
        for (size_t i = 0; i < mSamples.size(); i++)
        {
            float3 p = mEmitPosition + mSamples[i].x * mEmitAxis1 + mSamples[i].y * mEmitAxis2;
            emitPoints.emplace_back(p);
        }
        return emitPoints;
    }

    void CudaEmitter::BuildSquareEmitter(float particleRadius, float emitterRadius)
    {
        mSamples.clear();

        float offset = particleRadius * 2.f;

        for (float i = -emitterRadius; i < emitterRadius; i += offset)
        {
            for (float j = -emitterRadius; j < emitterRadius; j += offset)
            {
                mSamples.emplace_back(make_float2(i, j));
            }
        }

        if (!mSamples.empty())
            bBuild = true;
    }
    void CudaEmitter::BuildCircleEmitter(float particleRadius, float emitterRadius)
    {
        mSamples.clear();

        float offset = particleRadius * 2.f;

        for (float i = -emitterRadius; i < emitterRadius; i += offset)
        {
            for (float j = -emitterRadius; j < emitterRadius; j += offset)
            {
                float2 p = make_float2(i, j);
                if (length(p) <= emitterRadius)
                    mSamples.emplace_back(p);
            }
        }

        if (!mSamples.empty())
            bBuild = true;
    }
    void CudaEmitter::BuildRectangleEmitter(float particleRadius, float emitterWidth, float emitterHeight)
    {
        mSamples.clear();

        float offset = particleRadius * 2.f;

        for (float i = -emitterWidth; i < emitterWidth; i += offset)
        {
            for (float j = -emitterHeight; j < emitterHeight; j += offset)
            {
                mSamples.emplace_back(make_float2(i, j));
            }
        }

        if (!mSamples.empty())
            bBuild = true;
    }

} // namespace KIRI
