/*
 * @Author: Xu.WANG
 * @Date: 2021-02-04 12:36:10
 * @LastEditTime: 2021-04-19 00:26:19
 * @LastEditors: Xu.WANG
 * @Description:
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\data\cuda_array.cuh
 */

#ifndef _CUDA_ARRAY_CUH_
#define _CUDA_ARRAY_CUH_

#pragma once

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>

namespace KIRI {
template <typename T> class CudaArray {
  // static_assert(
  //     IsSame_Float<T>::value || IsSame_Float2<T>::value ||
  //     IsSame_Float3<T>::value || IsSame_Float4<T>::value ||
  //         IsSame_Int<T>::value || IsSame_UInt<T>::value
  //         ||IsSame_SizeT<T>::value ||IsSame_Bool<T>::value,
  //     "data type is not correct");

public:
  explicit CudaArray(const size_t len, const bool unified = false)
      : mLen(len), mArray([len, unified]() {
          T *ptr;
          if (unified)
            KIRI_CUCALL(cudaMallocManaged((void **)&ptr, sizeof(T) * len));
          else
            KIRI_CUCALL(cudaMalloc((void **)&ptr, sizeof(T) * len));
          SharedPtr<T> t(new (ptr) T[len],
                         [](T *ptr) { KIRI_CUCALL(cudaFree(ptr)); });
          return t;
        }()) {
    this->Clear();
  }

  CudaArray(const CudaArray &) = delete;
  CudaArray &operator=(const CudaArray &) = delete;
  T *Data(const int offset = 0) const { return mArray.get() + offset; }

  size_t Length() const { return mLen; }
  void Clear() {
    KIRI_CUCALL(cudaMemset(this->Data(), 0, sizeof(T) * this->Length()));
  }

  ~CudaArray() noexcept {}

private:
  const size_t mLen;
  const SharedPtr<T> mArray;
};
} // namespace KIRI

#endif