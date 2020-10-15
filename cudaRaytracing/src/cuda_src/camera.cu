#include "camera.cuh"
#include "maths/matrix.cuh"
#include "utils/util.h"
#include "utils/utils.cuh"

namespace CUDA_RAY {

__device__ Vector Camera::frameBegin()
{
	float x2d = aspectRatio_, y2d = +1;

	float wantedAngle = toRadians(fov_ / 2);
	float wantedLength = tan(wantedAngle);
	float hypotLength = sqrt(sqr(aspectRatio_) + sqr(1.0));
	float scaleFactor = wantedLength / hypotLength;

	x2d *= scaleFactor * 1.5;
	y2d *= scaleFactor * 1.5;

	topLeft_ = Vector(-x2d, y2d, 1);
	topRight_ = Vector(x2d, y2d, 1);
	bottomLeft_ = Vector(-x2d, -y2d, 1);

	rotation_ =
		rotationAroundZ(toRadians(roll_)) *
		rotationAroundX(toRadians(pitch_)) *
		rotationAroundY(toRadians(yaw_));

	topLeft_ *= rotation_;
	topRight_ *= rotation_;
	bottomLeft_ *= rotation_;

	topLeft_ += position_;
	topRight_ += position_;
	bottomLeft_ += position_;
}


__device__ Ray Camera::getScreenRay(float xScreen, float yScreen) const
{
	Vector throughPoint =
		topLeft_ + (topRight_ - topLeft_) * (xScreen / RESX)
		+ (bottomLeft_ - topLeft_) * (yScreen / RESY);

	Ray ray;
	ray.dir_ = throughPoint - position_;
	ray.dir_.normalize();
	ray.start_ = position_;
	return ray;
}

}