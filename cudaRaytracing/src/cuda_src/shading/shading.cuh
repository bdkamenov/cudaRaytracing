#ifndef __SHADING_CUH__
#define __SHADING_CUH__

#include "color.cuh"
#include "geometry/geometry.cuh"
#include "shading/texturing.cuh"
#include "utils/cudaList.cuh"


namespace CUDA_RAY {

class Shading;

struct Node
{
	Geometry* geom_;
	Shading* shader_;
};

struct Light
{
	Vector pos_;
	float intensity_;
};

class Shading
{
public:
	__device__ virtual Color shade(const Ray& ray, const IntersectInfo& info, const CudaList<Light>* lights, const CudaList<Node>* nodeList) = 0;
	__device__ virtual ~Shading() {}
};


class Lambert : public Shading
{
public:
	__device__ Lambert(Texture* texture = nullptr, const Color& color = Color());
	__device__ ~Lambert() {}
	__device__ virtual Color shade(const Ray& ray, const IntersectInfo& info, const CudaList<Light>* lights, const CudaList<Node>* nodeList) override;

	Texture* texture_;
	Color color_;
};

}

#endif // __SHADING_H__