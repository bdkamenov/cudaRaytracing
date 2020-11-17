#ifndef __TEXTURING_CUH__
#define __TEXTURING_CUH__

#include "color.cuh"
#include "geometry/geometry.cuh"
#include "utils/cudaList.cuh"

namespace CUDA_RAY {


class Texture
{
public:
	__device__ virtual Color sample(const IntersectInfo& info) = 0;
	__device__ virtual ~Texture() {}
};

class CheckerTexture : public Texture
{
public:
	__device__ CheckerTexture(const Color& a, const Color& b, float scale = 0.2f) : a_(a), b_(b), scale_(scale) {}
	__device__ ~CheckerTexture() {}
	
	__device__ virtual Color sample(const IntersectInfo& info) override;

	Color a_, b_;
	float scale_;
};

}

#endif // __SHADING_H__