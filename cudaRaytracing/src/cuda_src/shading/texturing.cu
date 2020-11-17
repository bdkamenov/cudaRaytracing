#include "shading/texturing.cuh"
#include "utils/cudaList.cuh"

namespace CUDA_RAY {

Color CheckerTexture::sample(const IntersectInfo& info)
{
	int x = int(floor(info.u_ * scale_));
	int y = int(floor(info.v_ * scale_));

	return ((x + y) % 2 == 0) ? a_ : b_;
}
}