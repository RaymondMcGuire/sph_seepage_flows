/*** 
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-05-14 20:01:11
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-05-17 11:26:49
 * @FilePath: \sph_seepage_flows\seepage_flow\include\cuda\cuda_helper.h
 * @Description: 
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved. 
 */
// clang-format off
#include <kiri_pch.h>
#include <kiri_pbs_cuda/cuda_helper/helper_math.h>
// clang-format on
namespace KIRI {

inline std::vector<size_t> TransferGPUData2CPU(const size_t* gpu_data, const uint size )
{
     uint bytes = size * sizeof(size_t);
    size_t *cpu_data = (size_t *)malloc(bytes);
    cudaMemcpy(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost);
    return std::vector<size_t>(cpu_data, cpu_data + size);
}


inline std::vector<int> TransferGPUData2CPU(const int* gpu_data, const uint size )
{
     uint bytes = size * sizeof(int);
    int *cpu_data = (int *)malloc(bytes);
    cudaMemcpy(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost);
    return std::vector<int>(cpu_data, cpu_data + size);
}

inline std::vector<uint> TransferGPUData2CPU(const uint* gpu_data, const uint size )
{
     uint bytes = size * sizeof(uint);
    uint *cpu_data = (uint *)malloc(bytes);
    cudaMemcpy(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost);
    return std::vector<uint>(cpu_data, cpu_data + size);
}

inline std::vector<int2> TransferGPUData2CPU(const int2* gpu_data, const uint size )
{
     uint bytes = size * sizeof(int2);
    int2 *cpu_data = (int2 *)malloc(bytes);
    cudaMemcpy(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost);
    return std::vector<int2>(cpu_data, cpu_data + size);
}

inline std::vector<int3> TransferGPUData2CPU(const int3* gpu_data, const uint size )
{
     uint bytes = size * sizeof(int3);
    int3 *cpu_data = (int3 *)malloc(bytes);
    cudaMemcpy(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost);
    return std::vector<int3>(cpu_data, cpu_data + size);
}

inline std::vector<int4> TransferGPUData2CPU(const int4* gpu_data, const uint size )
{
     uint bytes = size * sizeof(int4);
    int4 *cpu_data = (int4 *)malloc(bytes);
    cudaMemcpy(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost);
    return std::vector<int4>(cpu_data, cpu_data + size);
}

inline std::vector<float> TransferGPUData2CPU(const float* gpu_data, const uint size )
{
     uint bytes = size * sizeof(float);
    float *cpu_data = (float *)malloc(bytes);
    cudaMemcpy(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost);
    return std::vector<float>(cpu_data, cpu_data + size);
}

inline std::vector<float2> TransferGPUData2CPU(const float2* gpu_data, const uint size )
{
     uint bytes = size * sizeof(float2);
    float2 *cpu_data = (float2 *)malloc(bytes);
    cudaMemcpy(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost);
    return std::vector<float2>(cpu_data, cpu_data + size);
}

inline std::vector<float3> TransferGPUData2CPU(const float3* gpu_data, const uint size )
{
     uint bytes = size * sizeof(float3);
    float3 *cpu_data = (float3 *)malloc(bytes);
    cudaMemcpy(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost);
    return std::vector<float3>(cpu_data, cpu_data + size);
}

inline std::vector<float4> TransferGPUData2CPU(const float4* gpu_data, const uint size )
{
     uint bytes = size * sizeof(float4);
    float4 *cpu_data = (float4 *)malloc(bytes);
    cudaMemcpy(cpu_data, gpu_data, bytes, cudaMemcpyDeviceToHost);
    return std::vector<float4>(cpu_data, cpu_data + size);
}




inline float3 KiriToCUDA(const Vector3F vec) {
  return make_float3(vec.x, vec.y, vec.z);
}

inline std::vector<float3> KiriToCUDA(const Array1Vec3F arr) {
  std::vector<float3> data;
  for (size_t i = 0; i < arr.size(); i++) {
    data.emplace_back(make_float3(arr[i].x, arr[i].y, arr[i].z));
  }
  return data;
}

inline std::vector<float3> KiriToCUDA(const Vec_Vec3F arr) {
  std::vector<float3> data;
  for (size_t i = 0; i < arr.size(); i++) {
    data.emplace_back(make_float3(arr[i].x, arr[i].y, arr[i].z));
  }
  return data;
}

inline Vec_Float4 KiriToCUDA(const Array1Vec4F arr) {
  Vec_Float4 data;
  for (size_t i = 0; i < arr.size(); i++) {
    data.emplace_back(make_float4(arr[i].x, arr[i].y, arr[i].z, arr[i].w));
  }
  return data;
}

inline Vector3F CUDAToKiri(const float3 vec) {
  return Vector3F(vec.x, vec.y, vec.z);
}

inline std::vector<float3> KiriArrVec4FToVecFloat3(const Array1Vec4F arr) {
  std::vector<float3> data;
  for (size_t i = 0; i < arr.size(); i++) {

    data.emplace_back(make_float3(arr[i].x, arr[i].y, arr[i].z));
  }
  return data;
}

inline Array1Vec4F CUDAVecF4ToKiriVec4F(const Vec_Float4 arr) {
  Array1Vec4F data;
  for (size_t i = 0; i < arr.size(); i++) {
    data.append(Vector4F(arr[i].x, arr[i].y, arr[i].z, arr[i].w));
  }
  return data;
}

inline Array1Vec4F CUDAFloat3ToKiriVector4F(const std::vector<float3> arr) {
  Array1Vec4F data;
  for (size_t i = 0; i < arr.size(); i++) {
    data.append(Vector4F(arr[i].x, arr[i].y, arr[i].z, 0.1f));
  }
  return data;
}


} // namespace KIRI