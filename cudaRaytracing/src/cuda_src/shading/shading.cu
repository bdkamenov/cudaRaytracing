#include "shading/shading.cuh"
#include "utils/cudaList.cuh"

namespace CUDA_RAY {

CheckerShader::CheckerShader(const Color& a, const Color& b, float scale) : a_(a), b_(b), scale_(scale)
{

}

__device__ Color CheckerShader::shade(const Ray& ray, const IntersectInfo& info, const CUDA_RAY::CudaList<Light>* lights)
{
	int x = int(floor(info.u_ / scale_));
	int y = int(floor(info.v_ / scale_));

	Color checkerColor = ((x + y) % 2 == 0) ? a_ : b_;

	Color result = checkerColor;

	//printf("bpabpa %i \n", lights->size_);

	for(int i = 0; i < lights->size_; ++i)
	{
		Vector v1 = info.normal_;
		Vector v2 = lights->list_[i].pos_ - info.ip_;
		double distanceToLightSqr = v2.lengthSqr();
		v2.normalize();
		double lambertCoeff = dot(v1, v2);
		double attenuationCoeff = 1.0 / distanceToLightSqr;

		result = result * lambertCoeff * attenuationCoeff * lights->list_[i].intensity_;
	}

	return result;
}

}