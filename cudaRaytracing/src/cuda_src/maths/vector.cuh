#ifndef __VECTOR3D_H__
#define __VECTOR3D_H__

#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace CUDA_RAY {

class Vector {
public:
    __host__ __device__ Vector() = default;
    __host__ __device__ Vector(float x, float y, float z):x_(x), y_(y), z_(z) { }


    __host__ __device__ void set(float x, float y, float z)
    {
        x_ = x;
        y_ = y;
        z_ = z;
    }

    __host__ __device__ void makeZero()
    {
        x_ = y_ = z_ = 0.0;
    }

    __host__ __device__ inline float length() const
    {
        return sqrt(x_ * x_ + y_ * y_ + z_ * z_);
    }

    __host__ __device__ inline constexpr float lengthSqr() const
    {
        return (x_ * x_ + y_ * y_ + z_ * z_);
    }

    __host__ __device__ void scale(float multiplier)
    {
        x_ *= multiplier;
        y_ *= multiplier;
        z_ *= multiplier;
    }

    __host__ __device__ void operator *= (float multiplier)
    {
        scale(multiplier);
    }

    __host__ __device__ void operator += (const Vector& other)
    {
        x_ += other.x_;
        y_ += other.y_;
        z_ += other.z_;
    }

    __host__ __device__ void operator /= (float divider)
    {
        scale(1.0 / divider);
    }

    __host__ __device__ void normalize()
    {
        float multiplier = 1.0 / length();

        scale(multiplier);
    }

    __host__ __device__ void setLength(float newLength)
    {
        scale(newLength / length());
    }

    float x_, y_, z_;

};

__host__ __device__ inline Vector operator+ (const Vector& a, const Vector& b)
{
    return Vector(a.x_ + b.x_, a.y_ + b.y_, a.z_ + b.z_);
}

__host__ __device__ inline Vector operator- (const Vector& a, const Vector& b)
{
    return Vector(a.x_ - b.x_, a.y_ - b.y_, a.z_ - b.z_);
}

__host__ __device__ inline Vector operator- (const Vector& a)
{
    return Vector(-a.x_, -a.y_, -a.z_);
}

/// dot product
__host__ __device__ inline float operator * (const Vector& a, const Vector& b)
{
    return a.x_ * b.x_ + a.y_ * b.y_ + a.z_ * b.z_;
}

/// dot product (functional form, to make it more explicit):
__host__ __device__ inline float dot(const Vector& a, const Vector& b)
{
    return a.x_ * b.x_ + a.y_ * b.y_ + a.z_ * b.z_;
}

/// cross product
__host__ __device__ inline Vector operator^ (const Vector& a, const Vector& b)
{
    return Vector(
            a.y_ * b.z_ - a.z_ * b.y_,
            a.z_ * b.x_ - a.x_ * b.z_,
            a.x_ * b.y_ - a.y_ * b.x_
    );
}

__host__ __device__ inline Vector operator* (const Vector& a, float multiplier)
{
    return Vector(a.x_ * multiplier, a.y_ * multiplier, a.z_ * multiplier);
}

__host__ __device__ inline Vector operator* (float multiplier, const Vector& a)
{
    return Vector(a.x_ * multiplier, a.y_ * multiplier, a.z_ * multiplier);
}

__host__ __device__ inline Vector operator/ (const Vector& a, float divider)
{
    float multiplier = 1.0 / divider;
    return Vector(a.x_ * multiplier, a.y_ * multiplier, a.z_ * multiplier);
}

__host__ __device__ inline Vector reflect(Vector in, const Vector& norm)
{
    in.normalize();
    in += 2 * norm * dot(norm, -in);
    in.normalize();
    return in;
}

__host__ __device__ inline Vector faceforward(const Vector& ray, const Vector& norm)
{
    if (dot(ray, norm) < 0) return norm;
    else return -norm;
}

struct Ray {
    Vector start_;
    Vector dir_; // normed!
};

}
#endif // __VECTOR3D_H__
