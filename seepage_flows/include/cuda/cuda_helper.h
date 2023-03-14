/*** 
 * @Author: Xu.WANG
 * @Date: 2020-11-04 03:24:07
 * @LastEditTime: 2021-03-29 17:22:23
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriExamples\include\cuda\cuda_helper.h
 */
#include <kiri_pch.h>
#include <kiri_pbs_cuda/cuda_helper/helper_math.h>

namespace KIRI
{

    inline float3 KiriToCUDA(const Vector3F vec)
    {
        return make_float3(vec.x, vec.y, vec.z);
    }

    inline std::vector<float3> KiriToCUDA(const Array1Vec3F arr)
    {
        std::vector<float3> data;
        for (size_t i = 0; i < arr.size(); i++)
        {
            data.emplace_back(make_float3(arr[i].x, arr[i].y, arr[i].z));
        }
        return data;
    }

    inline std::vector<float3> KiriToCUDA(const Vec_Vec3F arr)
    {
        std::vector<float3> data;
        for (size_t i = 0; i < arr.size(); i++)
        {
            data.emplace_back(make_float3(arr[i].x, arr[i].y, arr[i].z));
        }
        return data;
    }

    inline Vec_Float4 KiriToCUDA(const Array1Vec4F arr)
    {
        Vec_Float4 data;
        for (size_t i = 0; i < arr.size(); i++)
        {
            data.emplace_back(make_float4(arr[i].x, arr[i].y, arr[i].z, arr[i].w));
        }
        return data;
    }

    inline rect3 KiriToCUDA(const Rect3 &rect)
    {
        rect3 cuRect;
        cuRect.origin = make_float3(rect.original.x, rect.original.y, rect.original.z);
        cuRect.size = make_float3(rect.size.x, rect.size.y, rect.size.z);
        return cuRect;
    }

    inline std::vector<rect3> KiriToCUDA(const std::vector<Rect3> &rects)
    {
        std::vector<rect3> data;
        for (size_t i = 0; i < rects.size(); i++)
        {
            data.emplace_back(KiriToCUDA(rects[i]));
        }
        return data;
    }

    inline Vector3F CUDAToKiri(const float3 vec)
    {
        return Vector3F(vec.x, vec.y, vec.z);
    }

    inline Rect3 CUDAToKiri(const rect3 rect)
    {
        return Rect3(
            Vector3F(rect.origin.x, rect.origin.y, rect.origin.z),
            Vector3F(rect.size.x, rect.size.y, rect.size.z));
    }

    inline std::vector<float3> KiriArrVec4FToVecFloat3(const Array1Vec4F arr)
    {
        std::vector<float3> data;
        for (size_t i = 0; i < arr.size(); i++)
        {

            data.emplace_back(make_float3(arr[i].x, arr[i].y, arr[i].z));
        }
        return data;
    }

    inline Array1Vec4F CUDAVecF4ToKiriVec4F(const Vec_Float4 arr)
    {
        Array1Vec4F data;
        for (size_t i = 0; i < arr.size(); i++)
        {
            data.append(Vector4F(arr[i].x, arr[i].y, arr[i].z, arr[i].w));
        }
        return data;
    }

    inline Array1Vec4F CUDAFloat3ToKiriVector4F(const std::vector<float3> arr)
    {
        Array1Vec4F data;
        for (size_t i = 0; i < arr.size(); i++)
        {
            data.append(Vector4F(arr[i].x, arr[i].y, arr[i].z, 0.1f));
        }
        return data;
    }

    inline std::vector<float3> KiriVertexToVecFloat3(const Array1<VertexFull> arr)
    {
        std::vector<float3> data;
        for (size_t i = 0; i < arr.size(); i++)
        {

            data.emplace_back(make_float3(arr[i].Position[0], arr[i].Position[1], arr[i].Position[2]));
        }
        return data;
    }

    inline std::vector<uint3> KiriIndicesToFaces(const Array1<UInt> arr)
    {
        std::vector<uint3> data;
        for (size_t i = 0; i < arr.size(); i += 3)
        {
            data.emplace_back(make_uint3(arr[i], arr[i + 1], arr[i + 2]));
        }
        return data;
    }

} // namespace KIRI