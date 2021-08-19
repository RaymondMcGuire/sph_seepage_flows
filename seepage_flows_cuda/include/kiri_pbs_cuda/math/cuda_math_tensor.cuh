/*
 * @Author: Xu.Wang 
 * @Date: 2020-07-25 11:16:57 
 * @ Modified by: Xu Wang
 * @ Modified time: 2020-09-01 12:58:39
 */

#ifndef _CUDA_MATH_TENSOR_CUH_
#define _CUDA_MATH_TENSOR_CUH_

#pragma once

#include <kiri_pbs_cuda/cuda_helper/helper_math.h>

// tensor3x3
struct tensor3x3
{
    float3 e1;
    float3 e2;
    float3 e3;
};

inline __host__ __device__ tensor3x3 make_tensor3x3(float e)
{
    tensor3x3 t3x3;
    t3x3.e1 = make_float3(e);
    t3x3.e2 = make_float3(e);
    t3x3.e3 = make_float3(e);

    return t3x3;
}

inline __host__ __device__ tensor3x3 make_tensor3x3(float3 e)
{
    tensor3x3 t3x3;
    t3x3.e1 = e;
    t3x3.e2 = e;
    t3x3.e3 = e;

    return t3x3;
}

inline __host__ __device__ tensor3x3 make_tensor3x3(float3 v1, float3 v2)
{
    tensor3x3 t3x3;
    t3x3.e1 = v1.x * v2;
    t3x3.e2 = v1.y * v2;
    t3x3.e3 = v1.z * v2;

    return t3x3;
}

inline __host__ __device__ tensor3x3 make_tensor3x3(float3 e1, float3 e2, float3 e3)
{
    tensor3x3 t3x3;
    t3x3.e1 = e1;
    t3x3.e2 = e2;
    t3x3.e3 = e3;

    return t3x3;
}

inline __host__ __device__ tensor3x3 make_transpose(tensor3x3 t3x3)
{
    tensor3x3 t3x3t;
    t3x3t.e1 = make_float3(t3x3.e1.x, t3x3.e2.x, t3x3.e3.x);
    t3x3t.e2 = make_float3(t3x3.e1.y, t3x3.e2.y, t3x3.e3.y);
    t3x3t.e3 = make_float3(t3x3.e1.z, t3x3.e2.z, t3x3.e3.z);

    return t3x3t;
}

inline __host__ __device__ tensor3x3 operator+(tensor3x3 a, tensor3x3 b)
{
    return make_tensor3x3(a.e1 + b.e1, a.e2 + b.e2, a.e3 + b.e3);
}

inline __host__ __device__ tensor3x3 operator+(tensor3x3 a, float b)
{
    return make_tensor3x3(a.e1 + b, a.e2 + b, a.e3 + b);
}

inline __host__ __device__ void operator+=(tensor3x3 &a, float b)
{
    a.e1 += b;
    a.e2 += b;
    a.e3 += b;
}

inline __host__ __device__ void operator+=(tensor3x3 &a, tensor3x3 b)
{
    a.e1 += b.e1;
    a.e2 += b.e2;
    a.e3 += b.e3;
}

inline __host__ __device__ tensor3x3 operator-(tensor3x3 a, float b)
{
    return make_tensor3x3(a.e1 - b, a.e2 - b, a.e3 - b);
}

inline __host__ __device__ tensor3x3 operator-(tensor3x3 a, tensor3x3 b)
{
    return make_tensor3x3(a.e1 - b.e1, a.e2 - b.e2, a.e3 - b.e3);
}

inline __host__ __device__ tensor3x3 operator/(tensor3x3 t3x3, float s)
{
    return make_tensor3x3(t3x3.e1 / s, t3x3.e2 / s, t3x3.e3 / s);
}

inline __host__ __device__ tensor3x3 operator/(float s, tensor3x3 t3x3)
{
    return make_tensor3x3(s / t3x3.e1, s / t3x3.e2, s / t3x3.e3);
}

inline __host__ __device__ tensor3x3 operator/(tensor3x3 a, tensor3x3 b)
{
    return make_tensor3x3(a.e1 / b.e1, a.e2 / b.e2, a.e3 / b.e3);
}

inline __host__ __device__ void operator/=(tensor3x3 &a, float b)
{
    a.e1 /= b;
    a.e2 /= b;
    a.e3 /= b;
}

inline __host__ __device__ tensor3x3 operator*(tensor3x3 t3x3, float s)
{
    return make_tensor3x3(t3x3.e1 * s, t3x3.e2 * s, t3x3.e3 * s);
}

inline __host__ __device__ tensor3x3 operator*(float s, tensor3x3 t3x3)
{
    return make_tensor3x3(s * t3x3.e1, s * t3x3.e2, s * t3x3.e3);
}

inline __host__ __device__ void operator*=(tensor3x3 &a, float b)
{
    a.e1 *= b;
    a.e2 *= b;
    a.e3 *= b;
}

inline __host__ __device__ tensor3x3 make_diagonal(float3 v)
{
    float3 e1 = make_float3(v.x, 0.f, 0.f);
    float3 e2 = make_float3(0.f, v.y, 0.f);
    float3 e3 = make_float3(0.f, 0.f, v.z);
    return make_tensor3x3(e1, e2, e3);
}

inline __host__ __device__ tensor3x3 make_symmetrize(tensor3x3 a)
{
    float a11 = a.e1.x, a12 = a.e1.y, a13 = a.e1.z;
    float a22 = a.e2.y, a23 = a.e2.z;
    float a33 = a.e3.z;

    float3 e1 = make_float3(a11, a12, a13);
    float3 e2 = make_float3(a12, a22, a23);
    float3 e3 = make_float3(a13, a23, a33);
    return make_tensor3x3(e1, e2, e3);
}

inline __host__ __device__ tensor3x3 make_identity()
{
    float3 e1 = make_float3(1.f, 0.f, 0.f);
    float3 e2 = make_float3(0.f, 1.f, 0.f);
    float3 e3 = make_float3(0.f, 0.f, 1.f);
    return make_tensor3x3(e1, e2, e3);
}

inline __host__ __device__ tensor3x3 dot(tensor3x3 a, tensor3x3 b)
{
    float a11 = a.e1.x, a12 = a.e1.y, a13 = a.e1.z;
    float a21 = a.e2.x, a22 = a.e2.y, a23 = a.e2.z;
    float a31 = a.e3.x, a32 = a.e3.y, a33 = a.e3.z;

    float b11 = b.e1.x, b12 = b.e1.y, b13 = b.e1.z;
    float b21 = b.e2.x, b22 = b.e2.y, b23 = b.e2.z;
    float b31 = b.e3.x, b32 = b.e3.y, b33 = b.e3.z;

    tensor3x3 result;
    result.e1 = make_float3(a11 * b11 + a12 * b21 + a13 * b31, a11 * b12 + a12 * b22 + a13 * b32, a11 * b13 + a12 * b23 + a13 * b33);
    result.e2 = make_float3(a21 * b11 + a22 * b21 + a23 * b31, a21 * b12 + a22 * b22 + a23 * b32, a21 * b13 + a22 * b23 + a23 * b33);
    result.e3 = make_float3(a31 * b11 + a32 * b21 + a33 * b31, a31 * b12 + a32 * b22 + a33 * b32, a31 * b13 + a32 * b23 + a33 * b33);

    return result;
}

inline __host__ __device__ float ddot(tensor3x3 a, tensor3x3 b)
{
    float a11 = a.e1.x, a12 = a.e1.y, a13 = a.e1.z;
    float a21 = a.e2.x, a22 = a.e2.y, a23 = a.e2.z;
    float a31 = a.e3.x, a32 = a.e3.y, a33 = a.e3.z;

    float b11 = b.e1.x, b12 = b.e1.y, b13 = b.e1.z;
    float b21 = b.e2.x, b22 = b.e2.y, b23 = b.e2.z;
    float b31 = b.e3.x, b32 = b.e3.y, b33 = b.e3.z;

    float result = a11 * b11 + a12 * b12 + a13 * b13 + a21 * b21 + a22 * b22 + a23 * b23 + a31 * b31 + a32 * b32 + a33 * b33;

    return result;
}

inline __host__ __device__ float det(tensor3x3 a)
{
    float a11 = a.e1.x, a12 = a.e1.y, a13 = a.e1.z;
    float a21 = a.e2.x, a22 = a.e2.y, a23 = a.e2.z;
    float a31 = a.e3.x, a32 = a.e3.y, a33 = a.e3.z;

    float result = a11 * a22 * a33 + a12 * a23 * a31 + a13 * a21 * a32 - a13 * a22 * a31 - a23 * a32 * a11 - a12 * a21 * a33;
    return result;
}

inline __host__ __device__ tensor3x3 adj(tensor3x3 a)
{
    float a11 = a.e1.x, a12 = a.e1.y, a13 = a.e1.z;
    float a21 = a.e2.x, a22 = a.e2.y, a23 = a.e2.z;
    float a31 = a.e3.x, a32 = a.e3.y, a33 = a.e3.z;

    tensor3x3 result;
    result.e1 = make_float3(a22 * a33 - a32 * a23, (a21 * a33 - a31 * a23) * -1.f, a21 * a32 - a22 * a31);
    result.e2 = make_float3((a12 * a33 - a13 * a32) * -1.f, a11 * a33 - a13 * a31, (a11 * a32 - a12 * a31) * -1.f);
    result.e3 = make_float3(a12 * a23 - a22 * a13, (a11 * a23 - a21 * a13) * -1.f, a11 * a22 - a21 * a12);

    return result;
}

inline __host__ __device__ tensor3x3 inverse(tensor3x3 a)
{
    tensor3x3 b = make_transpose(a);
    float det_b = det(b);
    if (det_b == 0.f)
    {
        return make_tensor3x3(make_float3(0.f));
    }

    tensor3x3 adj_b = adj(b);
    tensor3x3 result = 1.f / det_b * adj_b;

    return result;
}

inline __host__ __device__ float3 dot(tensor3x3 a, float3 b)
{
    return make_float3(dot(a.e1, b), dot(a.e2, b), dot(a.e3, b));
}

inline __host__ __device__ tensor3x3 decompose_symmetric(tensor3x3 t3x3)
{
    tensor3x3 trans = make_transpose(t3x3);
    tensor3x3 s = 0.5f * (t3x3 + trans);
    return s;
}

inline __host__ __device__ tensor3x3 decompose_antisymmetric(tensor3x3 t3x3)
{
    tensor3x3 trans = make_transpose(t3x3);
    tensor3x3 s = 0.5f * (t3x3 - trans);
    return s;
}

inline __host__ __device__ float first_stress_invariants(tensor3x3 t3x3)
{
    float I1 = t3x3.e1.x + t3x3.e2.y + t3x3.e3.z;
    return I1;
}

inline __host__ __device__ float second_stress_invariants(tensor3x3 t3x3)
{
    float e11 = t3x3.e1.x, e12 = t3x3.e1.y, e13 = t3x3.e1.z, e22 = t3x3.e2.y, e23 = t3x3.e2.z, e33 = t3x3.e3.z;
    float I2 = e11 * e22 + e22 * e33 + e11 * e33 - e12 * e12 - e23 * e23 - e13 * e13;
    return I2;
}

inline __host__ __device__ float third_stress_invariants(tensor3x3 t3x3)
{
    float e11 = t3x3.e1.x, e12 = t3x3.e1.y, e13 = t3x3.e1.z, e22 = t3x3.e2.y, e23 = t3x3.e2.z, e31 = t3x3.e3.x, e33 = t3x3.e3.z;
    float I3 = e11 * e22 * e33 + 2 * e12 * e23 * e31 - e12 * e12 * e33 - e23 * e23 * e11 - e13 * e13 * e22;
    return I3;
}

inline __host__ __device__ float first_deviatoric_stress_invariants(tensor3x3 t3x3)
{
    float J1 = 0.f;
    return J1;
}

inline __host__ __device__ float second_deviatoric_stress_invariants(tensor3x3 t3x3)
{
    float I1 = first_stress_invariants(t3x3);
    float I2 = second_stress_invariants(t3x3);
    float J2 = I1 * I1 / 3.f - I2;
    return J2;
}

inline __host__ __device__ float second_deviatoric_stress_invariants_by_deviatoric_tensor(tensor3x3 t3x3)
{
    float e11 = t3x3.e1.x, e12 = t3x3.e1.y, e13 = t3x3.e1.z, e22 = t3x3.e2.y, e23 = t3x3.e2.z, e33 = t3x3.e3.z;
    float J2 = 0.5f * (e11 * e11 + e22 * e22 + e33 * e33 + 2 * e12 * e12 + 2 * e23 * e23 + 2 * e13 * e13);
    return J2;
}

inline __host__ __device__ float third_deviatoric_stress_invariants(tensor3x3 t3x3)
{
    float I1 = first_stress_invariants(t3x3);
    float I2 = second_stress_invariants(t3x3);
    float I3 = third_stress_invariants(t3x3);
    float J3 = I1 * I1 * I1 * 2.f / 27.f - I1 * I2 / 3.f + I3;
    return J3;
}

inline __host__ __device__ tensor3x3 hydrostatic_stress_tensor(tensor3x3 t3x3)
{
    float p = first_stress_invariants(t3x3) / 3.f;
    float3 e1 = make_float3(p, 0.f, 0.f);
    float3 e2 = make_float3(0.f, p, 0.f);
    float3 e3 = make_float3(0.f, 0.f, p);

    return make_tensor3x3(e1, e2, e3);
}

inline __host__ __device__ tensor3x3 deviatoric_tensor(tensor3x3 t3x3)
{
    tensor3x3 hydro = hydrostatic_stress_tensor(t3x3);
    tensor3x3 deviatoric = t3x3 - hydro;
    return deviatoric;
}

//------------------unit test
// float3 e1 = make_float3(1.f, 2.f, 3.f);
// float3 e2 = make_float3(3.f, 2.f, 1.f);
// float3 e3 = make_float3(1.f, 2.f, 3.f);
// tensor3x3 t3x3 = make_tensor3x3(e1, e2);
// printTensor3x3("t3x3", t3x3);
// tensor3x3 epsilon3x3 = decompose_symmetric(t3x3);
// printTensor3x3("epsilon3x3", epsilon3x3);
// tensor3x3 omega3x3 = decompose_antisymmetric(t3x3);
// printTensor3x3("omega3x3", omega3x3);

// float3 e1 = make_float3(0.5f, 0.3f, 0.2f);
// float3 e2 = make_float3(0.3f, -0.2f, -0.1f);
// float3 e3 = make_float3(0.2f, -0.1f, 0.1f);
// tensor3x3 t3x3 = make_tensor3x3(e1, e2, e3);
// tensor3x3 dev = deviatoric_tensor(t3x3);
// printTensor3x3("dev", dev);

// float3 e1 = make_float3(5.f, 3.f, 1.f);
// float3 e2 = make_float3(3.f, 2.f, 0.f);
// float3 e3 = make_float3(1.f, 0.f, 4.f);
// tensor3x3 t3x3 = make_tensor3x3(e1, e2, e3);
// float I1 = first_stress_invariants(t3x3);
// printFloat("I1", I1);
// float I2 = second_stress_invariants(t3x3);
// printFloat("I2", I2);
// float I3 = third_stress_invariants(t3x3);
// printFloat("I3", I3);

// float3 e1 = make_float3(5.f, 3.f, 1.f);
// float3 e2 = make_float3(3.f, 2.f, 0.f);
// float3 e3 = make_float3(1.f, 0.f, 4.f);
// tensor3x3 t3x3 = make_tensor3x3(e1, e2, e3);
// printTensor3x3("t3x3", t3x3);
// t3x3 *= 2.f;
// printTensor3x3("t3x3 *= 2", t3x3);
// t3x3 /= 2.f;
// printTensor3x3("t3x3 /= 2", t3x3);
// t3x3 += 1.f;
// printTensor3x3("t3x3 += 1", t3x3);
// t3x3 += t3x3;
// printTensor3x3("t3x3 += t3x3", t3x3);

// float3 e1 = make_float3(5.f, 3.f, 1.f);
// float3 e2 = make_float3(4.f, 2.f, 0.f);
// float3 e3 = make_float3(3.f, 0.f, 4.f);
// tensor3x3 t3x3 = make_tensor3x3(e1, e2, e3);
// printTensor3x3("t3x3", t3x3);
// float tdotVal = dot(t3x3, make_float3(1.f, 2.f, 3.f));
// printFloat("tdotVal", tdotVal);
// float vdotT = dot(make_float3(1.f, 2.f, 3.f), t3x3);
// printFloat("vdotT", vdotT);

// float3 e1 = make_float3(5.f, 3.f, 1.f);
// float3 e2 = make_float3(4.f, 2.f, 0.f);
// float3 e3 = make_float3(3.f, 0.f, 4.f);
// tensor3x3 t3x3 = make_tensor3x3(e1, e2, e3);
// tensor3x3 trans = make_transpose(t3x3);
// tensor3x3 dotRes = dot(t3x3, trans);
// printTensor3x3("dotRes", dotRes);

//float3 e1 = make_float3(1.f, -3.f, 4.f);
//float3 e2 = make_float3(-3.f, -6.f, 1.f);
//float3 e3 = make_float3(4.f, 1.f, 5.f);
//tensor3x3 t3x3 = make_tensor3x3(e1, e2, e3);
//float sec = second_deviatoric_stress_invariants_by_deviatoric_tensor(t3x3);
//printFloat("sec", sec);

// float3 e1 = make_float3(3.f, 0.f, 2.f);
// float3 e2 = make_float3(2.f, 0.f, -2.f);
// float3 e3 = make_float3(0.f, 1.f, 1.f);
// tensor3x3 t3x3 = make_tensor3x3(e1, e2, e3);
// float deter = det(t3x3);
// printFloat("det", deter);

// tensor3x3 adjt = adj(make_transpose(t3x3));
// printTensor3x3("adj", adjt);

// tensor3x3 inv = inverse(t3x3);
// printTensor3x3("inv", inv);

// float3 a1 = make_float3(1.f, 2.f, 3.f);
// float3 a2 = make_float3(4.f, 2.f, 2.f);
// float3 a3 = make_float3(2.f, 3.f, 4.f);
// tensor3x3 a = make_tensor3x3(a1, a2, a3);
// float3 b1 = make_float3(1.f, 4.f, 7.f);
// float3 b2 = make_float3(2.f, 5.f, 8.f);
// float3 b3 = make_float3(3.f, 6.f, 9.f);
// tensor3x3 b = make_tensor3x3(b1, b2, b3);
// float doubleDot = ddot(a, b);
//printFloat("doubleDot", doubleDot);

#endif /* _CUDA_MATH_TENSOR_CUH_ */
