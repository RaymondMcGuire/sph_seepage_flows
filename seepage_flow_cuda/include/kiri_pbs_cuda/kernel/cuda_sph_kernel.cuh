/*
 * @Author: Xu.WANG
 * @Date: 2021-02-07 17:48:08
 * @LastEditTime: 2021-03-02 18:24:27
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\kernel\cuda_sph_kernel.cuh
 */

#ifndef _CUDA_SPH_KERNEL_CUH_
#define _CUDA_SPH_KERNEL_CUH_

#pragma once

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>

namespace KIRI {
struct CubicKernel {
  float h, coef;
  __host__ __device__ CubicKernel(const float radius) : h(radius) {
    const float h3 = h * h * h;
    coef = 1.f / (h3 * KIRI_PI);
  }

  __device__ float operator()(const float r) {
    float res = 0.f;
    const float q = fabsf(r) / h;
    if (q <= 1.f && q > KIRI_EPSILON) {
      if (q <= 0.5f) {
        const float q2 = q * q;
        const float q3 = q2 * q;
        res = coef * (6.f * q3 - 6.f * q2 + 1.f);
      } else {
        res = coef * (2.f * powf(1.f - q, 3.f));
      }
    }
    return res;
  }
};

struct CubicKernelGrad {
  float h, coef;
  __host__ __device__ CubicKernelGrad(const float radius) : h(radius) {
    const float h3 = h * h * h;
    coef = 6.f / (h3 * KIRI_PI);
  }

  __device__ float3 operator()(const float3 r) {
    float3 res = make_float3(0.f);
    const float rl = length(r);
    const float q = rl / h;
    if ((rl > KIRI_EPSILON) && (q <= 1.f)) {
      const float3 gradq = r / (rl * h);
      if (q <= 0.5f) {
        res = coef * q * (3.f * q - 2.f) * gradq;
      } else {
        res = coef * (2.f * q - q * q - 1.f) * gradq;
      }
    }
    return res;
  }
};

struct Poly6Kernel {
  float h, coef, h2;
  __host__ __device__ Poly6Kernel(const float radius) : h(radius) {
    h2 = h * h;
    float ih = 1.f / h;
    float ih3 = ih * ih * ih;
    float ih9 = ih3 * ih3 * ih3;
    coef = 315.f * ih9 / (64.f * KIRI_PI);
  }
  __host__ __device__ float operator()(float r) {
    float r2 = r * r;
    if (r2 >= h2)
      return 0;
    float d = h2 - r2;
    return coef * d * d * d;
  }
};

struct SpikyKernelGrad {
  float h, coef;
  __host__ __device__ SpikyKernelGrad(const float radius) : h(radius) {
    float h2 = h * h;
    float h6 = h2 * h2 * h2;
    coef = -45.f / (KIRI_PI * h6);
  }

  __device__ float3 operator()(float3 r) {
    float rlen = length(r);
    if (rlen >= h || rlen < KIRI_EPSILON)
      return make_float3(0, 0, 0);
    float d = h - rlen;
    return coef * d * d / rlen * r;
  }
};

struct SpikyKernelLaplacian {
  float h, coef;
  __host__ __device__ SpikyKernelLaplacian(const float radius) : h(radius) {
    float h2 = h * h;
    float h3 = h * h2;
    float h5 = h2 * h3;
    coef = 90.f / (KIRI_PI * h5);
  }

  __device__ float operator()(float r) {
    if (r >= h)
      return 0.f;

    float d = 1.f - r / h;
    return coef * d;
  }
};

struct ViscosityKernelLaplacian {
  float h, coef;
  __host__ __device__ ViscosityKernelLaplacian(const float radius) : h(radius) {
    const float h6 = h * h * h * h * h * h;
    coef = 45.f / (h6 * KIRI_PI);
  }

  __device__ float operator()(const float r) {
    float res = 0.f;
    const float q = fabsf(r) / h;
    if (q <= 1.f) {
      res = coef * (h - r);
    }
    return res;
  }
};

/**
 * @description: Akinci2013, "Versatile surface tension and adhesion for SPH
 * fluids", Surface Tension Spline Function
 * @param h: kernel radius
 * @return C: surface tension coefficient
 */
struct SurfaceTensionAkinci13 {
  float coef, h, h6, h9;
  __host__ __device__ SurfaceTensionAkinci13(const float radius) : h(radius) {
    float h3 = h * h * h;
    h6 = h3 * h3;
    h9 = h3 * h6;
    coef = 32.f / (KIRI_PI * h9);
  }
  __host__ __device__ float operator()(float r) {
    float rh = h - r;
    if ((r > rh) && (rh >= 0.f))
      return coef * powf(rh, 3.f) * r * r * r;
    else if ((r > 0.f) && (rh >= r))
      return coef * (2.f * powf(rh, 3.f) * r * r * r - h6 / 64.f);
    else
      return 0.f;
  }
};

struct AdhesionAkinci13 {
  float coef, h;
  __host__ __device__ AdhesionAkinci13(const float radius) : h(radius) {
    coef = 0.007f / powf(h, 3.25f);
  }
  __host__ __device__ float operator()(float r) {
    if (2.f * r > h && r <= h)
      return coef * powf(-4.f * r * r / h + 6.f * r - 2.f * h, 0.25f);
    else
      return 0.f;
  }
};

} // namespace KIRI

#endif /* _CUDA_SPH_KERNEL_CUH_ */