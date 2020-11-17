#ifndef __CUDA_COLOR_H__
#define __CUDA_COLOR_H__

#include "utils/util.h"
#include "utils/utils.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace CUDA_RAY{


__host__ __device__ inline unsigned convertTo8bit(float x)
{
    if (x < 0) x = 0;
    if (x > 1) x = 1;
    return nearestInt(x * 255.0f);
}

/// Represents a color, using floatingpoint components in [0..1]
struct Color {

    // a union, that allows us to refer to the channels by name (::r, ::g, ::b),
    // or by index (::components[0] ...). See operator [].
    union {
        struct { float r_, g_, b_; };
        float components_[3];
    };

    __host__ __device__ Color() : r_(0.f), g_(0.f), b_(0.f) {}
    __host__ __device__ Color(float r, float g, float b) : r_(r), g_(g), b_(b) { }  //!< Construct a color from floatingpoint values

    __host__ __device__ Color(unsigned rgbcolor) : //!< Construct a color from R8G8B8 value like "0xffce08"
                        b_((rgbcolor & 0xff) / 255.0f),
                        g_(((rgbcolor >> 8) & 0xff) / 255.0f),
                        r_(((rgbcolor >> 16) & 0xff) / 255.0f) { }

    /// convert to RGB32, with channel shift specifications. The default values are for
    /// the blue channel occupying the least-significant byte
    __host__ __device__ unsigned toRGB32(int redShift = 16, int greenShift = 8, int blueShift = 0)
    {
        unsigned ir = convertTo8bit(r_);
        unsigned ig = convertTo8bit(g_);
        unsigned ib = convertTo8bit(b_);
        return (ib << blueShift) | (ig << greenShift) | (ir << redShift);
    }

    /// make black
    __host__ __device__ void makeZero(void)
    {
        r_ = g_ = b_ = 0;
    }

    /// set the color explicitly
    __host__ __device__ void setColor(float r, float g, float b)
    {
        r_ = r;
        g_ = g;
        b_ = b;
    }

    /// get the intensity of the color (direct)
    __host__ __device__ constexpr float intensity()
    {
        return (r_ + g_ + b_) / 3;
    }

    /// get the perceptual intensity of the color
    __host__ __device__ constexpr float intensityPerceptual()
    {
        return (r_ * 0.299 + g_ * 0.587 + b_ * 0.114);
    }

    /// Accumulates some color to the current
    __host__ __device__ void operator += (const Color& rhs)
    {
        r_ += rhs.r_;
        g_ += rhs.g_;
        b_ += rhs.b_;
    }

    /// multiplies the color
    __host__ __device__ void operator *= (float multiplier)
    {
        r_ *= multiplier;
        g_ *= multiplier;
        b_ *= multiplier;
    }

    /// divides the color
    __host__ __device__ void operator /= (float divider)
    {
        r_ /= divider;
        g_ /= divider;
        b_ /= divider;
    }

    __host__ __device__ inline const float& operator[] (int index) const
    {
        return components_[index];
    }

    __host__ __device__ inline float& operator[] (int index)
    {
        return components_[index];
    }

};

/// adds two colors
__host__ __device__ inline Color operator+ (const Color& a, const Color& _b)
{
    return Color(a.r_ + _b.r_, a.g_ + _b.g_, a.b_ + _b.b_);
}

/// subtracts two colors
__host__ __device__ inline Color operator- (const Color& a, const Color& _b)
{
    return Color(a.r_ - _b.r_, a.g_ - _b.g_, a.b_ - _b.b_);
}

/// multiplies two colors
__host__ __device__ inline Color operator* (const Color& a, const Color& _b)
{
    return Color(a.r_ * _b.r_, a.g_ * _b.g_, a.b_ * _b.b_);
}

/// multiplies a color by some multiplier
__host__ __device__ inline  Color operator* (const Color& a, float multiplier)
{
    return Color(a.r_ * multiplier, a.g_ * multiplier, a.b_ * multiplier);
}

/// multiplies a color by some multiplier
__host__ __device__ inline Color operator* (float multiplier, const Color& a)
{
    return Color(a.r_ * multiplier, a.g_ * multiplier, a.b_ * multiplier);
}

/// divides some color
__host__ __device__ inline Color operator/ (const Color& a, float divider)
{
    return Color(a.r_ / divider, a.g_ / divider, a.b_ / divider);
}
}

#endif // __CUDA_COLOR_H__
