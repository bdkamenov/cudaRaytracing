#include "shading/shading.cuh"
#include "utils/cudaList.cuh"

namespace CUDA_RAY {

__device__ bool visibilityCheck(const CudaList<Node>* nodeList, const Vector& start, const Vector& end)
{
	Ray ray;
	ray.start_ = start;
	ray.dir_ = end - start;
	ray.dir_.normalize();

	double targetDist = (end - start).length();

	for (int i = 0; i < nodeList->size_; ++i)
	{
		IntersectInfo info;
		if (!nodeList->list_[i].geom_->intersect(ray, info)) 
			continue;

		if (info.distance_ < targetDist) 
		{
			return false;
		}
	}
	return true;
}


Lambert::Lambert(Texture* texture, const Color& color) : color_(color), texture_(texture) {}

__device__ Color Lambert::shade(const Ray& ray, const IntersectInfo& info, const CUDA_RAY::CudaList<Light>* lights, const CudaList<Node>* nodeList)
{
	Color textureColor = texture_ ? texture_->sample(info) : color_;
	Color result = Color(0.f, 0.f, 0.f);

	//printf("bpabpa %i \n", lights->size_);

	for(int i = 0; i < lights->size_; ++i)
	{
		if (!visibilityCheck(nodeList, info.ip_ + info.normal_ * 1e-3, lights->list_[i].pos_))
		{
			textureColor = Color(0.f, 0.f, 0.f);
		}

		Vector v1 = info.normal_;
		Vector v2 = lights->list_[i].pos_ - info.ip_;
		double distanceToLightSqr = v2.lengthSqr();
		v2.normalize();
		double lambertCoeff = dot(v1, v2);
		double attenuationCoeff = 1.0 / distanceToLightSqr;

		result += textureColor * lambertCoeff * attenuationCoeff * lights->list_[i].intensity_;
	}

	return result;
}

}