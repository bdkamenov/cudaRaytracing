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
	// H = ray.start - O
	// p^2 * dir.length()^2 + p * 2 * dot(H, dir) + H.length()^2 - R^2 == 0
	Vector H = ray.start_ - pos_;
	float A = 1; // ray.lengthSqr()
	float B = 2 * dot(H, ray.dir_);
	float C = H.lengthSqr() - rad_ * rad_;

	float discr = B * B - 4 * A * C;

	if (discr < 0) return false; // no intersection

	float p1, p2;
	p1 = (-B - sqrt(discr)) / (2 * A);
	p2 = (-B + sqrt(discr)) / (2 * A);
	float p;
	// p1 <= p2
	bool backNormal = false;
	if (p1 > 0) p = p1;
	else if (p2 > 0) {
		p = p2;
		backNormal = true;
	}
	else return false;

	info.distance_ = p;
	info.ip_ = ray.start_ + ray.dir_ * p;
	info.normal_ = info.ip_ - pos_;
	info.normal_.normalize();
	if (backNormal) info.normal_ = -info.normal_;
	info.u_ = info.v_ = 0;
	Vector posRelative = info.ip_ - pos_;
	info.u_ = atan2(posRelative.z_, posRelative.x_);
	info.v_ = asin(posRelative.y_ / rad_);
	// remap [(-PI..PI)x(-PI/2..PI/2)] -> [(0..1)x(0..1)]
	info.u_ = (info.u_ + PI) / (2 * PI);
	info.v_ = (info.v_ + PI / 2) / (PI);
	//info. = this;
	return true;
}


}