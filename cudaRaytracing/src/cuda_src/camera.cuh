#ifndef __CAMERA_H__
#define __CAMERA_H__


#include "maths/vector.cuh"
#include "maths/matrix.cuh"

namespace CUDA_RAY {

struct Camera
{
public:
	Vector position_;

	float yaw_, pitch_, roll_; // in degrees
	float aspectRatio_;
	float fov_; // in degrees
	Matrix rotation_;

	__device__ Vector frameBegin();
	__device__ Ray getScreenRay(float xScreen, float yScreen) const;

private:
	Vector topLeft_, topRight_, bottomLeft_;

};
}
#endif // !__CAMERA_H__