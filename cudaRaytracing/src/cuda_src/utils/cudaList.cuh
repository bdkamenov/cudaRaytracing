#ifndef __CUDA_LIST_CUH__
#define __CUDA_LIST_CUH__

#include "cuda_runtime_api.h"

namespace CUDA_RAY
{


template<typename T>
class CudaList
{
public:

    __device__ CudaList() : size_(0), cap_(2), list_(new T[cap_]) {
    }

    __device__ ~CudaList() 
    {
        delete[] list_;
    }

    __device__ int getSize() const { return size_; }

    __device__ T* operator[](int i) const { return list_ + i; }

    __device__ void addElem(const T& elem)
    {
        if (size_ >= cap_) {
            T* old = list_;
            list_ = new T[cap_ = cap_ * 2];
            // memcpy doesnt work for some reason ???? 
           // memcpy(list_, old, size_ * sizeof(T));
            printf("%i \n", size_ * sizeof(T));
            for (int i = 0; i < size_; ++i)
            {
                list_[i] = old[i];
            }


            delete[] old;
        }

        list_[size_] = elem;
        ++size_;
        printf("%i  %i \n", size_, cap_);
    }


    int size_;
    int cap_;
    T* list_;
};

}
#endif // __CUDA_LIST_CUH__