/*** 
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-05-14 20:01:11
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-06-26 11:20:24
 * @FilePath: \sph_seepage_flows\seepage_flow_cuda\include\kiri_pbs_cuda\kiri_pbs_pch.cuh
 * @Description: 
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved. 
 */

#ifndef _KIRI_PBS_PCH_CUH_
#define _KIRI_PBS_PCH_CUH_

#pragma once

// clang-format off
// Standard Libraries
#include <iostream>
#include <memory>
#include <vector>

// CUDA Libraries
#include <kiri_pbs_cuda/cuda_helper/helper_cuda.h>
#include <kiri_pbs_cuda/math/cuda_pbs_math.h>

// Thrust Libraries
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
// clang-format on

// CUDA Global Params & Macros
#define KIRI_CUBLOCKSIZE 256
#define KIRI_CUCHECK checkCudaErrors
#define KIRI_CUERROR getLastCudaError
#define KIRI_CUCALL(x)                                                         \
  do {                                                                         \
    if ((x) != cudaSuccess) {                                                  \
      printf("CUDA Error at %s:%d\t Error code = %d\n", __FILE__, __LINE__,    \
             x);                                                               \
    }                                                                          \
  } while (0)

#define KIRI_CUKERNAL()                                                        \
  ;                                                                            \
  {                                                                            \
    cudaError_t err = cudaGetLastError();                                      \
    if (err)                                                                   \
      printf("CUDA Error at %s:%d:\t%s\n", __FILE__, __LINE__,                 \
             cudaGetErrorString(err));                                         \
  }

#define KIRI_EPSILON 1e-6f
#define KIRI_RANDOM_SEEDS 1000
#define KIRI_PI 3.141592653589793238462643383279502884197f
#define KIRI_2PI 6.283185307179586476925286766559005768394f
#define KIRI_SQRTPI 1.772453850905516027298167483341145182797f
#define KIRI_SQRT2PI 2.506628274631000502415765284811045253006f

#define KIRI_EXPANDF3(p) p.x, p.y, p.z

#define KIRI_PBS_ASSERT(condition)                                             \
  {                                                                            \
    if (!(condition)) {                                                        \
      std::cerr << "KIRI_PBS_ASSERT FAILED: " << #condition << " @ "           \
                << __FILE__ << " (" << __LINE__ << ")" << std::endl;           \
    }                                                                          \
  }

// MultiSPH Max Phase Num
#define MULTISPH_MAX_PHASE_NUM 2

namespace KIRI {
// Helper Function
static inline __host__ __device__ uint CuCeilDiv(const uint a, const uint b) {
  return (a % b != 0) ? (a / b + 1) : (a / b);
}

template <class Type, class UType> using IsSame = std::is_same<Type, UType>;

template <class Type> using IsSame_Float = std::is_same<Type, float>;
template <class Type> using IsSame_Float2 = std::is_same<Type, float2>;
template <class Type> using IsSame_Float3 = std::is_same<Type, float3>;
template <class Type> using IsSame_Float4 = std::is_same<Type, float4>;
template <class Type> using IsSame_Double = std::is_same<Type, double>;
template <class Type> using IsSame_Int = std::is_same<Type, int>;
template <class Type> using IsSame_UInt = std::is_same<Type, uint>;
template <class Type> using IsSame_SizeT = std::is_same<Type, size_t>;
template <class Type> using IsSame_Bool = std::is_same<Type, bool>;

template <class Type> using Vector = std::vector<Type>;

using Vec_Float = Vector<float>;
using Vec_Float2 = Vector<float2>;
using Vec_Float3 = Vector<float3>;
using Vec_Float4 = Vector<float4>;

using Vec_SizeT = Vector<size_t>;

template <class Type> using Vec_Vec = Vector<Vector<Type>>;

template <class Type> using SharedPtr = std::shared_ptr<Type>;
template <class Type> using UniquePtr = std::unique_ptr<Type>;

template <class T> constexpr auto MEpsilon() {
  return std::numeric_limits<T>::epsilon();
}
template <class T> constexpr auto Tiny() {
  return std::numeric_limits<T>::min();
}
template <class T> constexpr auto Huge() {
  return std::numeric_limits<T>::max();
}

} // namespace KIRI
#endif