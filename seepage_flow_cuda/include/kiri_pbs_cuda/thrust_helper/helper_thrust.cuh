/*
 * @Author: Xu.WANG
 * @Date: 2021-02-03 20:17:45
 * @LastEditTime: 2021-03-14 19:40:21
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\thrust_helper\helper_thrust.cuh
 */

#ifndef _THRUST_HELPER_CUH_
#define _THRUST_HELPER_CUH_

#pragma once

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>

namespace ThrustHelper
{
    template <typename T>
    struct Plus
    {
        T B;
        Plus(const T b) : B(b) {}
        __host__ __device__
            T
            operator()(const T &a) const
        {
            return a + B;
        }
    };

    template <typename T>
    struct AbsPlus
    {
        __host__ __device__
            T
            operator()(const T &a, const T &b) const
        {
            return abs(a) + abs(b);
        }
    };

    template <typename T>
    struct CompareLengthCuda
    {
        static_assert(
            KIRI::IsSame_Float2<T>::value || KIRI::IsSame_Float3<T>::value || KIRI::IsSame_Float4<T>::value,
            "data type is not correct");

        __host__ __device__ bool operator()(T f1, T f2)
        {
            return length(f1) < length(f2);
        }
    };

    static inline __host__ __device__ int3 ComputeGridXYZByPos3(const float3 &pos, const float cellSize, const int3 &gridSize)
    {
        int x = min(max((int)(pos.x / cellSize), 0), gridSize.x - 1),
            y = min(max((int)(pos.y / cellSize), 0), gridSize.y - 1),
            z = min(max((int)(pos.z / cellSize), 0), gridSize.z - 1);

        return make_int3(x, y, z);
    }

    template <typename T>
    struct Pos2GridHash
    {
        static_assert(
            KIRI::IsSame_Float3<T>::value || KIRI::IsSame_Float4<T>::value,
            "position data structure must be float3 or float4");

        float3 mLowestPoint;
        float mCellSize;
        int3 mGridSize;
        __host__ __device__ Pos2GridHash(
            const float3 lowestPoint,
            const float cellSize,
            const int3 &gridSize)
            : mLowestPoint(lowestPoint),
              mCellSize(cellSize),
              mGridSize(gridSize) {}

        __host__ __device__ uint operator()(const T &pos)
        {
            float3 relPos = make_float3(pos.x, pos.y, pos.z) - mLowestPoint;
            int3 grid_xyz = ComputeGridXYZByPos3(relPos, mCellSize, mGridSize);
            return grid_xyz.x * mGridSize.y * mGridSize.z + grid_xyz.y * mGridSize.z + grid_xyz.z;
        }
    };

    template <typename T>
    struct Pos2GridXYZ
    {
        static_assert(
            KIRI::IsSame_Float3<T>::value || KIRI::IsSame_Float4<T>::value,
            "position data structure must be float3 or float4");

        float3 mLowestPoint;
        float mCellSize;
        int3 mGridSize;
        __host__ __device__ Pos2GridXYZ(
            const float3 lowestPoint,
            const float cellSize,
            const int3 &gridSize)
            : mLowestPoint(lowestPoint),
              mCellSize(cellSize),
              mGridSize(gridSize) {}

        __host__ __device__ int3 operator()(const T &pos)
        {
            float3 relPos = make_float3(pos.x, pos.y, pos.z) - mLowestPoint;
            return ComputeGridXYZByPos3(relPos, mCellSize, mGridSize);
        }
    };

    struct GridXYZ2GridHash
    {
        int3 mGridSize;
        __host__ __device__ GridXYZ2GridHash(const int3 &gridSize)
            : mGridSize(gridSize) {}

        template <typename T>
        __host__ __device__ uint operator()(T x, T y, T z)
        {
            return (x >= 0 && x < mGridSize.x && y >= 0 && y < mGridSize.y && z >= 0 && z < mGridSize.z)
                       ? (((x * mGridSize.y) + y) * mGridSize.z + z)
                       : (mGridSize.x * mGridSize.y * mGridSize.z);
        }
    };

} // namespace ThrustHelper

#endif