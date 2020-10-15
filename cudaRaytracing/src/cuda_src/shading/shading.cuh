#ifndef __SHADING_CUH__
#define __SHADING_CUH__

#include "color.cuh"
#include "geometry/geometry.cuh"
#include "utils/cudaList.cuh"


namespace CUDA_RAY {


struct Light
{
	Vector pos_;
	float intensity_;
};

class Shading
{
public:
	__device__ virtual Color shade(const Ray& ray, const IntersectInfo& info, const CudaList<Light>* lights) = 0;
	__device__ virtual ~Shading() {}
};


class CheckerShader : public Shading
{
public:
	__device__ CheckerShader(const Color& a, const Color& b, float scale = 5.f);
	__device__ ~CheckerShader() {}
	__device__ virtual Color shade(const Ray& ray, const IntersectInfo& info, const CudaList<Light>* lights) override;

	Color a_, b_;
	float scale_;
};

}

#endif // __SHADING_H__