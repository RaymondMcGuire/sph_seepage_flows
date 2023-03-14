/*
 * @Author: Xu.WANG
 * @Date: 2021-02-07 17:48:08
 * @LastEditTime: 2021-08-19 00:26:02
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\solver\seepageflow\cuda_sf_utils.cuh
 */

#ifndef _CUDA_SF_UTILS_CUH_
#define _CUDA_SF_UTILS_CUH_

#pragma once

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>

namespace KIRI
{
    struct LinearAttenCoeff
    {
        float c0, csat;
        __host__ __device__ LinearAttenCoeff(
            const float c0,
            const float csat)
            : c0(c0),
              csat(csat)
        {
        }

        __device__ float operator()(const float sr)
        {
            return csat + (1.f - sr) * (c0 - csat);
        }
    };

    struct QuadraticBezierCoeff
    {
        float a, b, c, d, e;
        float py0, py1, px1, py2;
        __host__ __device__ QuadraticBezierCoeff(
            const float py0,
            const float py1,
            const float px1,
            const float py2)
            : py0(py0),
              py1(py1),
              px1(px1),
              py2(py2)
        {
            a = (px1 + 1.f) / 4.f;
            b = -2.f * a;
            c = b * b;
            d = -4.f * (1.f + b - px1);
            e = 2.f * (1.f + b - px1);
        }

        __device__ float rx2t(const float sr)
        {
            return (b + std::sqrt(c + d * (px1 - sr))) / e;
        }

        __device__ float operator()(const float sr)
        {
            if (sr < 0.f)
                return py0;

            if (sr >= 1.f)
                return py2;

            if (sr <= px1)
            {
                const float t = sr / px1;
                const float omt = 1.f - t;
                return omt * omt * py0 + 2 * t * omt * py1 + t * t * py1;
            }
            else
            {
                const float t = rx2t(sr);
                const float omt = 1.f - t;
                return omt * omt * py1 + 2 * t * omt * py1 + t * t * py2;
            }
        }
    };

} // namespace KIRI

#endif /* _CUDA_SPH_KERNEL_CUH_ */