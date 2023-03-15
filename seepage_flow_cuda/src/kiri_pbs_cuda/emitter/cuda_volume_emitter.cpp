/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-01-12 14:57:38
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-01-12 14:59:11
 * @FilePath:
 * \sph_seepage_flows\seepage_flows_cuda\src\kiri_pbs_cuda\emitter\cuda_volume_emitter.cpp
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#include <kiri_pbs_cuda/emitter/cuda_volume_emitter.cuh>
#include <random>
namespace KIRI {

void CudaVolumeEmitter::BuildSphVolume(SphVolumeData &data, float3 lowest,
                                       int3 vsize, float particleRadius,
                                       float3 color) {
  if (!bEnable)
    return;

  float offset = 2.f * particleRadius;
  for (size_t i = 0; i < static_cast<size_t>(vsize.x); ++i) {
    for (size_t j = 0; j < static_cast<size_t>(vsize.y); ++j) {
      for (size_t k = 0; k < static_cast<size_t>(vsize.z); ++k) {
        float3 p = make_float3(lowest.x + i * offset, lowest.y + j * offset,
                               lowest.z + k * offset);

        data.pos.emplace_back(p);
        data.col.emplace_back(color);
      }
    }
  }
}

void CudaVolumeEmitter::BuildUniDemVolume(DemVolumeData &data, float3 lowest,
                                          int3 vsize, float particleRadius,
                                          float3 color, float mass,
                                          float jitter) {
  if (!bEnable)
    return;

  std::random_device seedGen;
  std::default_random_engine rndEngine(seedGen());
  std::uniform_real_distribution<> dist(-1.f, 1.f);

  float offset = 2.f * particleRadius;
  for (size_t i = 0; i < static_cast<size_t>(vsize.x); ++i) {
    for (size_t j = 0; j < static_cast<size_t>(vsize.y); ++j) {
      for (size_t k = 0; k < static_cast<size_t>(vsize.z); ++k) {
        float3 p = make_float3(lowest.x + i * offset, lowest.y + j * offset,
                               lowest.z + k * offset);

        data.pos.emplace_back(
            p + jitter * normalize(make_float3(dist(rndEngine), dist(rndEngine),
                                               dist(rndEngine))));
        data.col.emplace_back(color);
        data.mass.emplace_back(mass);
      }
    }
  }
}

void CudaVolumeEmitter::BuildDemShapeVolume(DemShapeVolumeData &data,
                                            Vec_Float4 shape, float3 color,
                                            float density) {
  if (!bEnable)
    return;

  data.minRadius = Huge<size_t>();
  for (size_t i = 0; i < shape.size(); i++) {
    float radius = shape[i].w;
    data.pos.emplace_back(make_float3(shape[i].x, shape[i].y, shape[i].z));
    data.col.emplace_back(color);
    data.radius.emplace_back(radius);
    data.mass.emplace_back(density *
                           ((4.f / 3.f) * KIRI_PI * std::powf(radius, 3.f)));
    data.minRadius = std::min(radius, data.minRadius);
  }
}

void CudaVolumeEmitter::BuildSeepageflowBoxVolume(SeepageflowVolumeData &data,
                                                  float3 lowest, int3 vsize,
                                                  float particleRadius,
                                                  float3 color, float mass,
                                                  size_t label, float jitter) {
  if (!bEnable)
    return;

  data.sandMinRadius = particleRadius;

  std::random_device seedGen;
  std::default_random_engine rndEngine(seedGen());
  std::uniform_real_distribution<> dist(-1.f, 1.f);

  float offset = 2.f * particleRadius;
  for (size_t i = 0; i < static_cast<size_t>(vsize.x); ++i) {
    for (size_t j = 0; j < static_cast<size_t>(vsize.y); ++j) {
      for (size_t k = 0; k < static_cast<size_t>(vsize.z); ++k) {
        float3 p = make_float3(lowest.x + i * offset, lowest.y + j * offset,
                               lowest.z + k * offset);

        data.pos.emplace_back(
            p + jitter * normalize(make_float3(dist(rndEngine), dist(rndEngine),
                                               dist(rndEngine))));
        data.col.emplace_back(color);
        data.label.emplace_back(label);
        data.radius.emplace_back(particleRadius);
        data.mass.emplace_back(mass);
      }
    }
  }
}

void CudaVolumeEmitter::BuildSeepageflowShapeVolume(
    SeepageflowVolumeData &data, Vec_Float4 shape, float3 color,
    float sandDensity, bool offsetY, float worldLowestY) {
  if (!bEnable)
    return;

  data.sandMinRadius = Huge<size_t>();
  float minY = Huge<size_t>();

  for (size_t i = 0; i < shape.size(); i++) {
    float radius = shape[i].w;
    data.pos.emplace_back(make_float3(shape[i].x, shape[i].y, shape[i].z));
    data.col.emplace_back(color);
    data.label.emplace_back(1);
    data.radius.emplace_back(radius);
    data.mass.emplace_back(sandDensity *
                           ((4.f / 3.f) * KIRI_PI * std::powf(radius, 3.f)));
    data.sandMinRadius = std::min(radius, data.sandMinRadius);
    minY = std::min(minY, shape[i].y);
  }

  if (offsetY) {
    float offsetYVal = minY - (worldLowestY + data.sandMinRadius * 5.f);
    for (size_t i = 0; i < data.pos.size(); i++)
      data.pos[i] -= make_float3(0.f, offsetYVal, 0.f);
  }
}

void CudaVolumeEmitter::BuildSeepageflowShapeMultiVolume(
    SeepageflowMultiVolumeData &data, Vec_Float4 shape, float3 color,
    float sandDensity, float3 cda0asat, float2 amcamcp, bool offsetY,
    float worldLowestY) {
  if (!bEnable)
    return;

  if (data.pos.size() == 0)
    data.sandMinRadius = Huge<size_t>();

  float minY = Huge<size_t>();

  for (size_t i = 0; i < shape.size(); i++) {
    float radius = shape[i].w;
    data.pos.emplace_back(make_float3(shape[i].x, shape[i].y, shape[i].z));
    data.col.emplace_back(color);
    data.label.emplace_back(1);
    data.radius.emplace_back(radius);
    data.mass.emplace_back(sandDensity *
                           ((4.f / 3.f) * KIRI_PI * std::powf(radius, 3.f)));
    data.sandMinRadius = std::min(radius, data.sandMinRadius);
    data.cda0asat.emplace_back(make_float3(cda0asat));
    data.amcamcp.emplace_back(make_float2(amcamcp));
    minY = std::min(minY, shape[i].y);
  }

  if (offsetY) {
    float offsetYVal = minY - (worldLowestY + data.sandMinRadius * 5.f);
    for (size_t i = 0; i < data.pos.size(); i++)
      data.pos[i] -= make_float3(0.f, offsetYVal, 0.f);
  }
}
} // namespace KIRI
