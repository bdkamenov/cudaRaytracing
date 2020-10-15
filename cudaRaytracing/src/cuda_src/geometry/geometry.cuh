#ifndef __GEOMETRY_CUH__
#define __GEOMETRY_CUH__

#include "utils/utils.cuh"
#include "maths/vector.cuh"

namespace CUDA_RAY {


struct IntersectInfo
{
	CUDA_RAY::Vector ip_; // intersection point
	CUDA_RAY::Vector normal_;
	float distance_;
	float u_, v_;
};


class Geometry
{

public:
	__device__ virtual bool intersect(const Ray& ray, IntersectInfo& info) = 0;
	__device__ virtual ~Geometry() {};
};


class Plane: public Geometry
{
public:
	__device__ Plane() : y_(0) {}
	__device__ Plane(float y) : y_(y) {}
	__device__ ~Plane() {};

	__device__ virtual bool intersect(const Ray& ray, IntersectInfo& info) override;

private:
	float y_;
};

class Sphere: public Geometry
{
public:
	__device__ Sphere(Vector pos): pos_(pos) {}
	__device__ ~Sphere() {}

	__device__ virtual bool intersect(const Ray& ray, IntersectInfo& info) override;

	Vector pos_;
	float rad_;
};

}

#endif // __GEOMETRY_CUH__