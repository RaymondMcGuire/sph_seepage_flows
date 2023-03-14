/*** 
 * @Author: Xu.WANG
 * @Date: 2021-03-19 22:04:26
 * @LastEditTime: 2021-04-04 02:20:24
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\emitter\cuda_boundary_emitter.cpp
 */
#include <kiri_pbs_cuda/emitter/cuda_boundary_emitter.cuh>
namespace KIRI
{
    void CudaBoundaryEmitter::BuildWorldBoundary(BoundaryData &data, const float3 &lowest, const float3 &highest, const float particleRadius)
    {
        if (!bEnable)
            return;

        size_t epsilon = 0;
        float spacing = particleRadius * 2.f;
        float3 sides = (highest - lowest) / spacing;

        //ZX plane - bottom
        for (size_t i = -epsilon; i <= sides.x + epsilon; ++i)
        {
            for (size_t j = -epsilon; j <= sides.z + epsilon; ++j)
            {
                data.pos.emplace_back(make_float3(lowest.x + i * spacing, lowest.y, lowest.z + j * spacing));
                data.label.emplace_back(0);
            }
        }

        //ZX plane - top
        for (size_t i = -epsilon; i <= sides.x + epsilon; ++i)
        {
            for (size_t j = -epsilon; j <= sides.z + epsilon; ++j)
            {
                data.pos.emplace_back(make_float3(lowest.x + i * spacing, highest.y, lowest.z + j * spacing));
                data.label.emplace_back(0);
            }
        }

        //XY plane - back
        for (size_t i = -epsilon; i <= sides.x + epsilon; ++i)
        {
            for (size_t j = -epsilon; j <= sides.y + epsilon; ++j)
            {
                data.pos.emplace_back(make_float3(lowest.x + i * spacing, lowest.y + j * spacing, lowest.z));
                data.label.emplace_back(0);
            }
        }

        //XY plane - front
        for (size_t i = -epsilon; i <= sides.x + epsilon; ++i)
        {
            for (size_t j = -epsilon; j <= sides.y - epsilon; ++j)
            {
                data.pos.emplace_back(make_float3(lowest.x + i * spacing, lowest.y + j * spacing, highest.z));
                data.label.emplace_back(0);
            }
        }

        //YZ plane - left
        for (size_t i = -epsilon; i <= sides.y + epsilon; ++i)
        {
            for (size_t j = -epsilon; j <= sides.z + epsilon; ++j)
            {
                data.pos.emplace_back(make_float3(lowest.x, lowest.y + i * spacing, lowest.z + j * spacing));
                data.label.emplace_back(0);
            }
        }

        //YZ plane - right
        for (size_t i = -epsilon; i <= sides.y + epsilon; ++i)
        {
            for (size_t j = -epsilon; j <= sides.z + epsilon; ++j)
            {
                data.pos.emplace_back(make_float3(highest.x, lowest.y + i * spacing, lowest.z + j * spacing));
                data.label.emplace_back(0);
            }
        }
    }

    void CudaBoundaryEmitter::BuildBoundaryShapeVolume(BoundaryData &data, Vec_Float4 shape)
    {
        if (!bEnable)
            return;

        for (size_t i = 0; i < shape.size(); i++)
        {
            data.pos.emplace_back(make_float3(shape[i].x, shape[i].y, shape[i].z));
            data.label.emplace_back(1);
        }
    }

} // namespace KIRI
