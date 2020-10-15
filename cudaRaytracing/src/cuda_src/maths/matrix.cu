#include <math.h>
#include "matrix.cuh"

namespace CUDA_RAY {

    __device__ Matrix rotationAroundX(float angle)
    {
        float S = sin(angle);
        float C = cos(angle);
        Matrix a(1.0);
        a.m_[1][1] = C;
        a.m_[2][1] = S;
        a.m_[1][2] = -S;
        a.m_[2][2] = C;
        return a;
    }

    __device__ Matrix rotationAroundY(float angle)
    {
        float S = sin(angle);
        float C = cos(angle);
        Matrix a(1.0);
        a.m_[0][0] = C;
        a.m_[2][0] = -S;
        a.m_[0][2] = S;
        a.m_[2][2] = C;
        return a;
    }

    __device__ Matrix rotationAroundZ(float angle)
    {
        float S = sin(angle);
        float C = cos(angle);
        Matrix a(1.0);
        a.m_[0][0] = C;
        a.m_[1][0] = S;
        a.m_[0][1] = -S;
        a.m_[1][1] = C;
        return a;
    }

    __device__ Matrix operator * (const Matrix& a, const Matrix& b)
    {
        Matrix c(0.0);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 3; k++)
                    c.m_[i][j] += a.m_[i][k] * b.m_[k][j];
        return c;
    }

    __device__ float determinant(const Matrix& a)
    {
        return a.m_[0][0] * a.m_[1][1] * a.m_[2][2]
            - a.m_[0][0] * a.m_[1][2] * a.m_[2][1]
            - a.m_[0][1] * a.m_[1][0] * a.m_[2][2]
            + a.m_[0][1] * a.m_[1][2] * a.m_[2][0]
            + a.m_[0][2] * a.m_[1][0] * a.m_[2][1]
            - a.m_[0][2] * a.m_[1][1] * a.m_[2][0];
    }

    __device__ float cofactor(const Matrix& m, int ii, int jj)
    {
        int rows[2], rc = 0, cols[2], cc = 0;
        for (int i = 0; i < 3; i++)
            if (i != ii) rows[rc++] = i;
        for (int j = 0; j < 3; j++)
            if (j != jj) cols[cc++] = j;
        float t = m.m_[rows[0]][cols[0]] * m.m_[rows[1]][cols[1]] - m.m_[rows[1]][cols[0]] * m.m_[rows[0]][cols[1]];
        if ((ii + jj) % 2) t = -t;
        return t;
    }

    __device__ Matrix inverseMatrix(const Matrix& m)
    {
        float D = determinant(m);
        if (fabs(D) < 1e-12) return m; // an error; matrix is not invertible
        float rD = 1.0 / D;
        Matrix result;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                result.m_[i][j] = rD * cofactor(m, j, i);
        return result;
    }

}