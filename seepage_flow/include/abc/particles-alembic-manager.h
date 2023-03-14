/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-02-19 15:45:28
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-02-19 23:43:33
 * @FilePath: \Kiri\KiriExamples\include\abc\particles-alembic-manager.h
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */
#ifndef PBF_PARTICLES_ALEMBIC_MANAGER_HPP
#define PBF_PARTICLES_ALEMBIC_MANAGER_HPP

#include <abc/alembic-manager-base.h>
#include <kiri_pbs_cuda/cuda_helper/helper_math.h>

class ParticlesAlembicManager : public AlembicManagerBase {
public:
  ParticlesAlembicManager(const std::string &output_file_path,
                          const double delta_time,
                          const std::string &object_name)
      : AlembicManagerBase(output_file_path, delta_time, object_name) {}

  void SubmitCurrentStatusFloat3(float3 *position_array, const uint num) {
    using namespace Alembic::Abc;
    using namespace Alembic::AbcGeom;

    uint f3_bytes = num * sizeof(float3);
    float3 *cpu_positions = (float3 *)malloc(f3_bytes);

    cudaMemcpy(cpu_positions, position_array, f3_bytes, cudaMemcpyDeviceToHost);

    if (m_is_first) {
      SubmitCurrentStatusFirstTimeFloat3(cpu_positions, num);

      m_is_first = false;

      return;
    }

    const V3fArraySample position_array_sample(
        reinterpret_cast<const V3f *>(cpu_positions), num);

    OPointsSchema::Sample sample;
    sample.setPositions(position_array_sample);
    // sample.setVelocities(velocity_array_sample);

    // GeometryScope radius_scope = kVaryingScope;
    // OFloatGeomParam::Sample radius_sample;
    // radius_sample.setVals(radius_array_sample);
    // radius_sample.setScope(radius_scope);
    // sample.setWidths(radius_sample);

    m_points.getSchema().set(sample);
  }

  void SubmitCurrentStatusFloat(float *radius_array, const uint num) {
    using namespace Alembic::Abc;
    using namespace Alembic::AbcGeom;

    uint f_bytes = num * sizeof(float);
    float *cpu_radius = (float *)malloc(f_bytes);

    cudaMemcpy(cpu_radius, radius_array, f_bytes, cudaMemcpyDeviceToHost);

    if (m_is_first) {
      SubmitCurrentStatusFirstTimeFloat(cpu_radius, num);

      m_is_first = false;

      return;
    }

    std::vector<float> position;
    for (auto i = 0; i < num; i++) {
      position.emplace_back(cpu_radius[i]);
      position.emplace_back(cpu_radius[i]);
      position.emplace_back(cpu_radius[i]);
    }

    OPointsSchema::Sample sample;
    sample.setPositions(P3fArraySample((const V3f *)&position.front(), num));
    m_points.getSchema().set(sample);
  }

private:
  void SubmitCurrentStatusFirstTimeFloat(float *radius_array, const uint num) {
    using namespace Alembic::Abc;
    using namespace Alembic::AbcGeom;

    const std::vector<std::int32_t> counts(num, 1);

    std::vector<uint64_t> index_buffer(num);
    for (std::size_t elem_index = 0; elem_index < num; ++elem_index) {
      index_buffer[elem_index] = elem_index;
    }

    const UInt64ArraySample index_array_sample(index_buffer.data(), num);

    std::vector<float> position;
    for (auto i = 0; i < num; i++) {
      position.emplace_back(radius_array[i]);
      position.emplace_back(radius_array[i]);
      position.emplace_back(radius_array[i]);
    }

    OPointsSchema::Sample sample;
    sample.setIds(index_array_sample);
    sample.setPositions(P3fArraySample((const V3f *)&position.front(), num));

    m_points.getSchema().set(sample);
  }

  void SubmitCurrentStatusFirstTimeFloat3(const float3 *position_array,
                                          const uint num) {
    using namespace Alembic::Abc;
    using namespace Alembic::AbcGeom;

    const std::vector<std::int32_t> counts(num, 1);

    std::vector<uint64_t> index_buffer(num);
    for (std::size_t elem_index = 0; elem_index < num; ++elem_index) {
      index_buffer[elem_index] = elem_index;
    }

    const V3fArraySample position_array_sample(
        reinterpret_cast<const V3f *>(position_array), num);
    const UInt64ArraySample index_array_sample(index_buffer.data(), num);

    OPointsSchema::Sample sample;
    sample.setIds(index_array_sample);
    sample.setPositions(position_array_sample);

    m_points.getSchema().set(sample);
  }
};

typedef std::shared_ptr<ParticlesAlembicManager> ParticlesAlembicManagerPtr;

#endif
