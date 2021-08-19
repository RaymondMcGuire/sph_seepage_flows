/*
 * @Author: Makoto Fujisawa
 * @Date: 2009-08, 2011-06
 * @Last Modified by: Xu.Wang
 * @Last Modified time: 2020-07-25 11:20:11
 */

#ifndef _CUDA_COMMON_CUH_
#define _CUDA_COMMON_CUH_

#pragma once

// basic library
#include <iostream>

// cuda library
#include <vector_functions.h>
#include <vector_types.h>
#include <kiri_pbs_cuda/cuda_helper/helper_cuda.h>

#define PBS_CUCHECK checkCudaErrors
#define PBS_CUERROR getLastCudaError

// thrust library
#include <thrust/copy.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>

// math library
#include <kiri_pbs_cuda/math/cuda_pbs_math.h>

// constants
#define CUDA_BLOCK_SIZE 512

#define MAX_NUM_OF_PHASES 3
#define MAX_NUM_OF_PARTICLES 2000000

// math
#define M_PI 3.141592653589793238462643383279502884197f
#define M_2PI 6.283185307179586476925286766559005768394f     // 2*PI
#define M_SQRTPI 1.772453850905516027298167483341145182797f  // sqrt(PI)
#define M_SQRT2PI 2.506628274631000502415765284811045253006f // sqrt(2*PI)

#define LIM_EPS 1e-3

inline float frand()
{
    return rand() / (float)RAND_MAX;
}

// physics
#define CUDA_GRAVITY 9.81f

// helper func
__host__ __device__ inline uint CeilDiv(uint a, uint b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

#define expand(p) p.x, p.y, p.z
#define expandq(q) q.s, q.v.x, q.v.y, q.v.z
#define expandt3x3(t3x3) t3x3.e1.x, t3x3.e1.y, t3x3.e1.z, t3x3.e2.x, t3x3.e2.y, t3x3.e2.z, t3x3.e3.x, t3x3.e3.y, t3x3.e3.z

__host__ __device__ inline void printTensor3x3(const char *name, tensor3x3 t3)
{
    printf(name);
    printf("=\n (%.6f,%.6f,%.6f)\n (%.6f,%.6f,%.6f)\n (%.6f,%.6f,%.6f)\n",
           expandt3x3(t3));
}

__host__ __device__ inline void printFloat3(const char *name, float3 f3)
{
    printf(name);
    printf("=(%.6f,%.6f,%.6f)\n",
           expand(f3));
}

__host__ __device__ inline void printFloat(const char *name, float f)
{
    printf(name);
    printf("=%.6f\n", f);
}

// merge array helper func
__host__ __device__ inline float *mergeFloatArrayInHost(int mergedBeforeNum, int mergeNum, float *mergedBeforeData, float *newData)
{
    int mergedNum = mergedBeforeNum + mergeNum;

    float *mergedArray = (float *)malloc(mergedNum * sizeof(float));
    float *mergedBeforeHostData = (float *)malloc(mergedBeforeNum * sizeof(float));

    PBS_CUCHECK(cudaMemcpy(mergedBeforeHostData, mergedBeforeData, mergedBeforeNum * sizeof(float), cudaMemcpyDeviceToHost));

    // mergedArray = mergedBeforeHostData + newData
    thrust::copy(mergedBeforeHostData, mergedBeforeHostData + mergedBeforeNum, mergedArray);
    thrust::copy(newData, newData + mergeNum, mergedArray + mergedBeforeNum);

    free(mergedBeforeHostData);

    return mergedArray;
}

__host__ __device__ inline float3 *mergeVector3ArrayInHost(int mergedBeforeNum, int mergeNum, float3 *mergedBeforeData, float3 *newData)
{
    int mergedNum = mergedBeforeNum + mergeNum;

    float3 *mergedArray = (float3 *)malloc(mergedNum * sizeof(float3));
    float3 *mergedBeforeHostData = (float3 *)malloc(mergedBeforeNum * sizeof(float3));

    PBS_CUCHECK(cudaMemcpy(mergedBeforeHostData, mergedBeforeData, mergedBeforeNum * sizeof(float3), cudaMemcpyDeviceToHost));

    // mergedArray = mergedBeforeHostData + newData
    thrust::copy(mergedBeforeHostData, mergedBeforeHostData + mergedBeforeNum, mergedArray);
    thrust::copy(newData, newData + mergeNum, mergedArray + mergedBeforeNum);

    free(mergedBeforeHostData);

    return mergedArray;
}

__host__ __device__ inline float4 *mergeVector4ArrayInHost(int mergedBeforeNum, int mergeNum, float4 *mergedBeforeData, float4 *newData)
{
    int mergedNum = mergedBeforeNum + mergeNum;

    float4 *mergedArray = (float4 *)malloc(mergedNum * sizeof(float4));
    float4 *mergedBeforeHostData = (float4 *)malloc(mergedBeforeNum * sizeof(float4));

    PBS_CUCHECK(cudaMemcpy(mergedBeforeHostData, mergedBeforeData, mergedBeforeNum * sizeof(float4), cudaMemcpyDeviceToHost));

    // mergedArray = mergedBeforeHostData + newData
    thrust::copy(mergedBeforeHostData, mergedBeforeHostData + mergedBeforeNum, mergedArray);
    thrust::copy(newData, newData + mergeNum, mergedArray + mergedBeforeNum);

    free(mergedBeforeHostData);

    return mergedArray;
}

__host__ __device__ inline void mergeUIntArrayInDevice(int mergedBeforeNum, int mergeNum, uint *mergedBeforeData, uint *newHostData)
{
    // alloc device data
    uint *newDeviceData;
    cudaMalloc((void **)&newDeviceData, sizeof(uint) * mergeNum);
    cudaMemcpy(newDeviceData, newHostData, sizeof(uint) * mergeNum, cudaMemcpyHostToDevice);

    thrust::device_ptr<uint> newDeviceDataPtr(newDeviceData);
    thrust::device_ptr<uint> mergedBeforeDataPtr(mergedBeforeData);

    thrust::copy(newDeviceDataPtr, newDeviceDataPtr + mergeNum, mergedBeforeDataPtr + mergedBeforeNum);
}
#endif /* _CUDA_COMMON_CUH_ */