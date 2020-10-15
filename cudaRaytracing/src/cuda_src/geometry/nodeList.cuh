#ifndef __GEOMETRY_LIST_CUH__
#define __GEOMETRY_LIST_CUH__

#include "geometry/geometry.cuh"
#include "shading/shading.cuh"

namespace CUDA_RAY {

struct Node
{
    Geometry* geom_;
    Shading* shader_;
};

class NodeList 
{
public:
    __device__ int getSize() const { return size_; }

    __device__ Node*& operator[](int i) const { return list_[i]; }


    Node** list_;
    int size_;
};

}
#endif //__GEOMETRY_LIST_CUH__