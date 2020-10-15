#include "geometry/geometry.cuh"

namespace CUDA_RAY {

__device__ bool Plane::intersect(const Ray& ray, IntersectInfo& info)
{
	if (ray.start_.y_ > y_ && ray.dir_.y_ >= 0)
		return false;
	if (ray.start_.y_ < y_ && ray.dir_.y_ <= 0)
		return false;

	float scaleFactor = (y_ - ray.start_.y_) / ray.dir_.y_;
	info.ip_ = ray.start_ + ray.dir_ * scaleFactor;
	info.distance_ = scaleFactor;
	info.normal_ = Vector(0, ray.start_.y_ > y_ ? 1 : -1, 0);
	info.u_ = info.ip_.x_;
	info.v_ = info.ip_.z_;
	return true;
}



__device__ bool Sphere::intersect(const Ray& ray, IntersectInfo& info)
{

	return true;
}


}