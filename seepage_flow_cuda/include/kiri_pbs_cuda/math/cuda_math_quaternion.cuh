/**
 * @ Author: Xu Wang
 * @ Create Time: 2020-08-13 11:32:40
 * @ Modified by: Xu Wang
 * @ Modified time: 2020-09-12 13:06:08
 * @ Description:
 */

#ifndef _CUDA_MATH_QUATERNION_CUH_
#define _CUDA_MATH_QUATERNION_CUH_

#pragma once

#include <kiri_pbs_cuda/cuda_helper/helper_math.h>
#include <kiri_pbs_cuda/math/cuda_math_tensor.cuh>

// quaternion(s,vx,vy,vz)
struct quaternion {
  float s;
  float3 v;
};

inline __host__ __device__ quaternion make_quaternion(float s, float3 v) {
  quaternion q;
  q.s = s;
  q.v = v;

  return q;
}

inline __host__ __device__ quaternion make_quaternion(float4 v4) {
  quaternion q;
  q.s = v4.x;
  q.v = make_float3(v4.y, v4.z, v4.w);

  return q;
}

inline __host__ __device__ quaternion make_quaternion(quaternion q0) {
  quaternion q;
  q.s = q0.s;
  q.v = q0.v;

  return q;
}

inline __host__ __device__ quaternion make_quaternion(float s, float vx,
                                                      float vy, float vz) {
  quaternion q;
  q.s = s;
  q.v = make_float3(vx, vy, vz);

  return q;
}

inline __host__ __device__ float4 coeffs(quaternion q) {
  return make_float4(q.s, q.v.x, q.v.y, q.v.z);
}

inline __host__ __device__ quaternion cross(quaternion q0, quaternion q1) {
  quaternion q;
  q.s = q0.s * q1.s - dot(q0.v, q1.v);
  q.v = q0.s * q1.v + q1.s * q0.v + cross(q0.v, q1.v);

  return q;
}

inline __host__ __device__ quaternion conjugate(quaternion q0) {
  quaternion q;
  q.s = q0.s;
  q.v = -q0.v;

  return q;
}

inline __host__ __device__ quaternion normalize(quaternion q) {
  quaternion nq;
  float invLen = rsqrtf(dot(q.v, q.v) + q.s * q.s);
  nq.s = q.s * invLen;
  nq.v = q.v * invLen;
  return nq;
}

inline __host__ __device__ tensor3x3
rotation_matrix_by_quaternion(quaternion q) {
  tensor3x3 t3x3;
  float e11 = 1.f - 2.f * q.v.y * q.v.y - 2.f * q.v.z * q.v.z;
  float e22 = 1.f - 2.f * q.v.x * q.v.x - 2.f * q.v.z * q.v.z;
  float e33 = 1.f - 2.f * q.v.x * q.v.x - 2.f * q.v.y * q.v.y;

  float e12 = 2.f * q.v.x * q.v.y - 2.f * q.s * q.v.z;
  float e13 = 2.f * q.v.x * q.v.z + 2.f * q.s * q.v.y;

  float e21 = 2.f * q.v.x * q.v.y + 2.f * q.s * q.v.z;
  float e23 = 2.f * q.v.y * q.v.z - 2.f * q.s * q.v.x;

  float e31 = 2.f * q.v.x * q.v.z - 2.f * q.s * q.v.y;
  float e32 = 2.f * q.v.y * q.v.z + 2.f * q.s * q.v.x;

  t3x3.e1 = make_float3(e11, e12, e13);
  t3x3.e2 = make_float3(e21, e22, e23);
  t3x3.e3 = make_float3(e31, e32, e33);

  return t3x3;
}

// https://gamedev.stackexchange.com/questions/28395/rotating-vector3-by-a-quaternion
inline __host__ __device__ float3 rotate_vector_by_quaternion(float3 v,
                                                              quaternion q) {
  float3 u = q.v;
  float s = q.s;

  float3 v_prime =
      2.f * dot(u, v) * u + (s * s - dot(u, u)) * v + 2.f * s * cross(u, v);

  return v_prime;
}

// http://www.euclideanspace.com/physics/kinematics/angularvelocity/QuaternionDifferentiation2.pdf
inline __host__ __device__ quaternion dot(float3 angVel, quaternion Q) {
  quaternion nq;

  float qw = Q.s;
  float3 qv = Q.v;
  nq.s = (-qv.x * angVel.x - qv.y * angVel.y - qv.z * angVel.z) / 2.f;

  float x = (qw * angVel.x - qv.z * angVel.y + qv.y * angVel.z) / 2.f;
  float y = (qv.z * angVel.x + qw * angVel.y - qv.x * angVel.z) / 2.f;
  float z = (-qv.y * angVel.x + qv.x * angVel.y + qw * angVel.z) / 2.f;

  nq.v = make_float3(x, y, z);
  return nq;
}

// // rotation matrix: MM^-1 = MM^T = I
// quaternion q = make_quaternion(0.f, 0.f, 1.f, 0.f);
// tensor3x3 rm = rotation_matrix_by_quaternion(q);
// tensor3x3 trm = make_transpose(rm);
// tensor3x3 I = dot(rm, trm);
// printTensor3x3("rm", rm);
// printTensor3x3("trm", trm);
// printTensor3x3("I", I);

#endif /* _CUDA_MATH_QUATERNION_CUH_ */
