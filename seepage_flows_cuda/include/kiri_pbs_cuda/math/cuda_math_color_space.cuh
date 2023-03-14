/*
 * @Author: Xu.WANG
 * @Date: 2020-11-24 20:10:34
 * @LastEditTime: 2020-11-24 20:22:51
 * @LastEditors: Xu.WANG
 * @Description: 
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\math\cuda_math_color_space.cuh
 */

#ifndef _CUDA_MATH_COLOR_SPACE_CUH_
#define _CUDA_MATH_COLOR_SPACE_CUH_

#pragma once

#include <kiri_pbs_cuda/cuda_helper/helper_math.h>

inline __host__ __device__ float3 rgb2hsv(float3 rgb)
{
    float3 out;
    float min, max, delta;

    min = rgb.x < rgb.y ? rgb.x : rgb.y;
    min = min < rgb.z ? min : rgb.z;

    max = rgb.x > rgb.y ? rgb.x : rgb.y;
    max = max > rgb.z ? max : rgb.z;

    out.z = max; // v
    delta = max - min;
    if (delta < 0.00001)
    {
        out.y = 0;
        out.x = 0; // undefined, maybe nan?
        return out;
    }
    if (max > 0.0)
    {                          // NOTE: if Max is == 0, this divide would cause a crash
        out.y = (delta / max); // s
    }
    else
    {
        // if max is 0, then r = g = b = 0
        // s = 0, h is undefined
        out.y = 0.0;
        out.x = NAN; // its now undefined
        return out;
    }
    if (rgb.x >= max)                    // > is bogus, just keeps compilor happy
        out.x = (rgb.y - rgb.z) / delta; // between yellow & magenta
    else if (rgb.y >= max)
        out.x = 2.0 + (rgb.z - rgb.x) / delta; // between cyan & yellow
    else
        out.x = 4.0 + (rgb.x - rgb.y) / delta; // between magenta & cyan

    out.x *= 60.0; // degrees

    if (out.x < 0.0)
        out.x += 360.0;

    return out;
}

inline __host__ __device__ float3 hsv2rgb(float3 hsv)
{
    float hh, p, q, t, ff;
    long i;
    float3 out;

    if (hsv.y <= 0.0)
    { // < is bogus, just shuts up warnings
        out.x = hsv.z;
        out.y = hsv.z;
        out.z = hsv.z;
        return out;
    }
    hh = hsv.x;
    if (hh >= 360.0)
        hh = 0.0;
    hh /= 60.0;
    i = (long)hh;
    ff = hh - i;
    p = hsv.z * (1.0 - hsv.y);
    q = hsv.z * (1.0 - (hsv.y * ff));
    t = hsv.z * (1.0 - (hsv.y * (1.0 - ff)));

    switch (i)
    {
    case 0:
        out.x = hsv.z;
        out.y = t;
        out.z = p;
        break;
    case 1:
        out.x = q;
        out.y = hsv.z;
        out.z = p;
        break;
    case 2:
        out.x = p;
        out.y = hsv.z;
        out.z = t;
        break;

    case 3:
        out.x = p;
        out.y = q;
        out.z = hsv.z;
        break;
    case 4:
        out.x = t;
        out.y = p;
        out.z = hsv.z;
        break;
    case 5:
    default:
        out.x = hsv.z;
        out.y = p;
        out.z = q;
        break;
    }
    return out;
}

inline __host__ __device__ float3 linear_ramp(float3 rgb1, float3 rgb2, float t)
{
    float3 hsv1 = rgb2hsv(rgb1);
    float3 hsv2 = rgb2hsv(rgb2);

    float3 linear_hsv = (hsv2 - hsv1) * t + hsv1;

    return hsv2rgb(linear_hsv);
}

#endif /* _CUDA_MATH_COLOR_SPACE_CUH_ */
