#ifndef __RAY_H__
#define __RAY_H__

#include <maths/vector.cuh>

struct Ray {
    Vector start_;
    Vector dir_; // normed!
};


#endif //RAY_H
