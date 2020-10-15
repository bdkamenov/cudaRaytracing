#ifndef __COLOR_H__
#define __COLOR_H__

#include "utils/util.h"

namespace CXX
{

    inline unsigned convertTo8bit(float x)
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

        Color() = default;
        constexpr Color(float r, float g, float b) : r_(r), g_(g), b_(b) { }  //!< Construct a color from floatingpoint values

        explicit Color(unsigned rgbcolor) : //!< Construct a color from R8G8B8 value like "0xffce08"
            r_(((rgbcolor >> 16) & 0xff) / 255.0f),
            g_(((rgbcolor >> 8) & 0xff) / 255.0f),
            b_((rgbcolor & 0xff) / 255.0f)
        { }

        /// convert to RGB32, with channel shift specifications. The default values are for
        /// the blue channel occupying the least-significant byte
        unsigned toRGB32(int redShift = 16, int greenShift = 8, int blueShift = 0)
        {
            unsigned ir = convertTo8bit(r_);
            unsigned ig = convertTo8bit(g_);
            unsigned ib = convertTo8bit(b_);
            return (ib << blueShift) | (ig << greenShift) | (ir << redShift);
        }

        /// make black
        void makeZero(void)
        {
            r_ = g_ = b_ = 0;
        }

        /// set the color explicitly
        void setColor(float r, float g, float b)
        {
            r_ = r;
            g_ = g;
            b_ = b;
        }

        /// get the intensity of the color (direct)
        constexpr float intensity()
        {
            return (r_ + g_ + b_) / 3;
        }

        /// get the perceptual intensity of the color
        constexpr float intensityPerceptual()
        {
            return (r_ * 0.299 + g_ * 0.587 + b_ * 0.114);
        }

        /// Accumulates some color to the current
        void operator += (const Color& rhs)
        {
            r_ += rhs.r_;
            g_ += rhs.g_;
            b_ += rhs.b_;
        }

        /// multiplies the color
        void operator *= (float multiplier)
        {
            r_ *= multiplier;
            g_ *= multiplier;
            b_ *= multiplier;
        }

        /// divides the color
        void operator /= (float divider)
        {
            r_ /= divider;
            g_ /= divider;
            b_ /= divider;
        }

        inline const float& operator[] (int index) const
        {
            return components_[index];
        }

        inline float& operator[] (int index)
        {
            return components_[index];
        }

    };

    /// adds two colors
    inline constexpr Color operator+ (const Color& a, const Color& _b)
    {
        return Color(a.r_ + _b.r_, a.g_ + _b.g_, a.b_ + _b.b_);
    }

    /// subtracts two colors
    inline constexpr Color operator- (const Color& a, const Color& _b)
    {
        return Color(a.r_ - _b.r_, a.g_ - _b.g_, a.b_ - _b.b_);
    }

    /// multiplies two colors
    inline constexpr Color operator* (const Color& a, const Color& _b)
    {
        return Color(a.r_ * _b.r_, a.g_ * _b.g_, a.b_ * _b.b_);
    }

    /// multiplies a color by some multiplier
    inline constexpr Color operator* (const Color& a, float multiplier)
    {
        return Color(a.r_ * multiplier, a.g_ * multiplier, a.b_ * multiplier);
    }

    /// multiplies a color by some multiplier
    inline constexpr Color operator* (float multiplier, const Color& a)
    {
        return Color(a.r_ * multiplier, a.g_ * multiplier, a.b_ * multiplier);
    }

    /// divides some color
    inline constexpr Color operator/ (const Color& a, float divider)
    {
        return Color(a.r_ / divider, a.g_ / divider, a.b_ / divider);
    }
}

#endif // __COLOR_H__
