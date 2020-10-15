#ifndef __MATRIX_H__
#define __MATRIX_H__

#include "device_launch_parameters.h"
#include "maths/vector.cuh"

namespace CUDA_RAY {

    struct Matrix {
        float m_[3][3];
        __device__ Matrix() = default;

        __device__ Matrix(float diagonalElement)
        {
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    if (i == j) m_[i][j] = diagonalElement;
                    else m_[i][j] = 0.0;
        }
    };

    __device__ inline Vector operator * (const Vector& v, const Matrix& m)
    {
        return Vector(
            v.x_ * m.m_[0][0] + v.y_ * m.m_[1][0] + v.z_ * m.m_[2][0],
            v.x_ * m.m_[0][1] + v.y_ * m.m_[1][1] + v.z_ * m.m_[2][1],
            v.x_ * m.m_[0][2] + v.y_ * m.m_[1][2] + v.z_ * m.m_[2][2]
        );
    }

    __device__ inline void operator *= (Vector& v, const Matrix& a) { v = v * a; }

    __device__ Matrix operator * (const Matrix& a, const Matrix& b); //!< matrix multiplication; result = a*b
    __device__ Matrix inverseMatrix(const Matrix& a); //!< finds the inverse of a matrix (assuming it exists)
    __device__ float determinant(const Matrix& a); //!< finds the determinant of a matrix

    __device__ Matrix rotationAroundX(float angle); //!< returns a rotation matrix around the X axis; the angle is in radians
    __device__ Matrix rotationAroundY(float angle); //!< same as above, but rotate around Y
    __device__ Matrix rotationAroundZ(float angle); //!< same as above, but rotate around Z
}
#endif // __MATRIX_H__
