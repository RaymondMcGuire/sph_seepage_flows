/*
 * @Author: Xu.WANG
 * @Date: 2020-07-21 16:37:22
 * @LastEditTime: 2021-02-25 02:48:03
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\cuda_searcher_struct.cuh
 */

#ifndef _CUDA_SEARCHER_STRUCT_CUH_
#define _CUDA_SEARCHER_STRUCT_CUH_

#pragma once

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>

struct ComputeGridXYZByPos
{
    float3 LowestPoint;
    float cellWidth;
    int3 GridSize;
    __host__ __device__ ComputeGridXYZByPos(const float3 &mLowestPoint,
                                            const float mCellSize, const int3 &mGridSize)
        : LowestPoint(mLowestPoint), cellWidth(mCellSize), GridSize(mGridSize) {}

    __host__ __device__ int3 operator()(float3 pos)
    {
        float3 diff = pos - LowestPoint;
        int x = min(max(int(diff.x / cellWidth), 0), GridSize.x - 1),
            y = min(max(int(diff.y / cellWidth), 0), GridSize.y - 1),
            z = min(max(int(diff.z / cellWidth), 0), GridSize.z - 1);
        return make_int3(x, y, z);
    }
};

struct ComputePolyGridXYZByPos
{
    float3 LowestPoint;
    float cellWidth;
    int3 GridSize;
    __host__ __device__ ComputePolyGridXYZByPos(const float3 &mLowestPoint,
                                                const float mCellSize, const int3 &mGridSize)
        : LowestPoint(mLowestPoint), cellWidth(mCellSize), GridSize(mGridSize) {}

    __host__ __device__ int3 operator()(float3 pos)
    {
        float3 diff = pos - LowestPoint;
        int x = min(int(diff.x / cellWidth), GridSize.x - 1),
            y = min(int(diff.y / cellWidth), GridSize.y - 1),
            z = min(int(diff.z / cellWidth), GridSize.z - 1);
        return make_int3(x, y, z);
    }
};

struct ComputeGridHashByPos
{
    float3 LowestPoint;
    float cellWidth;
    int3 GridSize;
    __host__ __device__ ComputeGridHashByPos(const float3 &mLowestPoint,
                                             const float mCellSize, const int3 &mGridSize)
        : LowestPoint(mLowestPoint), cellWidth(mCellSize), GridSize(mGridSize) {}

    __host__ __device__ int operator()(float3 pos)
    {
        float3 diff = pos - LowestPoint;
        int x = min(max((int)(diff.x / cellWidth), 0), GridSize.x - 1),
            y = min(max((int)(diff.y / cellWidth), 0), GridSize.y - 1),
            z = min(max((int)(diff.z / cellWidth), 0), GridSize.z - 1);
        return x * GridSize.y * GridSize.z + y * GridSize.z + z;
    }
};

struct ComputeGridHashByPos4
{
    float3 LowestPoint;
    float cellWidth;
    int3 GridSize;
    __host__ __device__ ComputeGridHashByPos4(const float3 &mLowestPoint,
                                              const float mCellSize, const int3 &mGridSize)
        : LowestPoint(mLowestPoint), cellWidth(mCellSize), GridSize(mGridSize) {}

    __host__ __device__ int operator()(float4 pos)
    {
        float3 pos3 = make_float3(pos.x, pos.y, pos.z);
        float3 diff = pos3 - LowestPoint;
        int x = min(max((int)(diff.x / cellWidth), 0), GridSize.x - 1),
            y = min(max((int)(diff.y / cellWidth), 0), GridSize.y - 1),
            z = min(max((int)(diff.z / cellWidth), 0), GridSize.z - 1);
        return x * GridSize.y * GridSize.z + y * GridSize.z + z;
    }
};

struct GridXYZ2GridHash
{
    int3 GridSize;
    __host__ __device__ GridXYZ2GridHash(const int3 &mGridSize)
        : GridSize(mGridSize) {}

    template <typename T>
    __host__ __device__ uint operator()(T x, T y, T z)
    {
        return x * GridSize.y * GridSize.z + y * GridSize.z + z;
    }
};

#endif