/***
 * @Author: Xu.WANG
 * @Date: 2022-04-17 15:08:41
 * @LastEditTime: 2022-05-12 22:16:02
 * @LastEditors: Xu.WANG
 * @Description:
 */
#include <kiri_pbs_cuda/emitter/cuda_emitter.cuh>
namespace KIRI {

Vec_Float3 CudaEmitter::Emit() {
  KIRI_PBS_ASSERT(bBuild);

  Vec_Float3 emitPoints;
  for (size_t i = 0; i < mSamples.size(); i++) {
    float3 p =
        mEmitPosition + mSamples[i].x * mEmitAxis1 + mSamples[i].y * mEmitAxis2;
    emitPoints.emplace_back(p);
  }
  return emitPoints;
}

void CudaEmitter::UpdateEmitterVelocity(float3 emitVelocity) {
  mEmitVelocity = emitVelocity;
  float3 axis = normalize(mEmitVelocity);

  if (abs(axis.x) == 1.f && abs(axis.y) == 0.f && abs(axis.z) == 0.f) {
    mEmitAxis1 = normalize(cross(axis, make_float3(0.f, 1.f, 0.f)));
  } else {
    mEmitAxis1 = normalize(cross(axis, make_float3(1.f, 0.f, 0.f)));
  }

  mEmitAxis2 = normalize(cross(axis, mEmitAxis1));
}

void CudaEmitter::BuildSquareEmitter(float particleRadius,
                                     float emitterRadius) {
  mSamples.clear();

  float offset = particleRadius * 2.f;

  for (float i = -emitterRadius; i < emitterRadius; i += offset) {
    for (float j = -emitterRadius; j < emitterRadius; j += offset) {
      mSamples.emplace_back(make_float2(i, j));
    }
  }

  if (!mSamples.empty())
    bBuild = true;
}

void CudaEmitter::BuildCircleEmitter(float particleRadius,
                                     float emitterRadius) {
  mSamples.clear();

  float offset = particleRadius * 2.f;

  for (float i = -emitterRadius; i < emitterRadius; i += offset) {
    for (float j = -emitterRadius; j < emitterRadius; j += offset) {
      float2 p = make_float2(i, j);
      if (length(p) <= emitterRadius)
        mSamples.emplace_back(p);
    }
  }

  if (!mSamples.empty())
    bBuild = true;
}
void CudaEmitter::BuildRectangleEmitter(float particleRadius,
                                        float emitterWidth,
                                        float emitterHeight) {
  mSamples.clear();

  float offset = particleRadius * 2.f;

  for (float i = -emitterWidth; i < emitterWidth; i += offset) {
    for (float j = -emitterHeight; j < emitterHeight; j += offset) {
      mSamples.emplace_back(make_float2(i, j));
    }
  }

  if (!mSamples.empty())
    bBuild = true;
}

} // namespace KIRI
