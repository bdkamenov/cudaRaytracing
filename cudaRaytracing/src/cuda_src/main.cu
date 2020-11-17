#include <iostream>
#include <time.h>
#include <assert.h>

#include "sdl_wrapper/sdl.h"

#include "geometry/nodeList.cuh"
#include "utils/cudaList.cuh"
#include "geometry/geometry.cuh"
#include "color.cuh"
#include "camera.cuh"
#include "utils/utils.cuh"

using CUDA_RAY::Camera;
using CUDA_RAY::Color;
using CUDA_RAY::Node;
using CUDA_RAY::NodeList;
using CUDA_RAY::Ray;
using CUDA_RAY::Light;
using CUDA_RAY::Vector;
using CUDA_RAY::CudaList;
using CUDA_RAY::Lambert;
using CUDA_RAY::CheckerTexture;

const int LIGHTS_COUNT = 2;
constexpr const int NODES_COUNT = 2;

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__device__ Color raytrace(const CudaList<Node>* list,  const Ray& ray, const CudaList<Light>* lights)
{
    Node* closestNode = nullptr;
    CUDA_RAY::IntersectInfo closestInfo;
    float closestDist = INF;

    //printf(" %i \n", list->getSize());
    //printf(" lapdplsadl \n");

    for (int i = 0; i < list->getSize(); ++i)
    {
        CUDA_RAY::IntersectInfo info;
        Node* node = (*list)[i];

        if (!node->geom_->intersect(ray, info)) continue;

        if (info.distance_ < closestDist)
        {
            closestDist = info.distance_;
            closestNode = node;
            closestInfo = info;
        }
    }

    // check if we hit the sky:
    if (closestNode == nullptr)
        return Color(0, 0, 0); // TODO: return background color
    else
       return closestNode->shader_->shade(ray, closestInfo, lights, list);
}


__global__ void render(Camera** cam, CudaList<Node>* list, CudaList<Light>* lights, Color* vfb, int x, int y)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
 
    // printf("threadIdx.x %i, blockIdx.x %i, blockDim.x %i \n", threadIdx.x, blockIdx.x, blockDim.x);

    if ((i >= x) || (j >= y))
    {
        printf("OUTSIDE!");
        printf("i = %d, j = %d", i, j);
        return;
    }

    CUDA_RAY::Ray ray = (*cam)->getScreenRay(j, i);

    int index = i * x + j;
    vfb[index] = raytrace(list, ray, lights);
}

__global__ void setupCamera(Camera** cam)
{
    *cam = new CUDA_RAY::Camera;

    (*cam)->position_ = CUDA_RAY::Vector(35, 60, 20);

    (*cam)->yaw_ = 0;
    (*cam)->pitch_ = 0;
    (*cam)->roll_ = 0;

    (*cam)->fov_ = 90;
    (*cam)->aspectRatio_ = float(RESX) / float(RESY);

    (*cam)->frameBegin();
}


__global__ void setupNodes(CudaList<Node>* list)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {

        list->list_ = new Node[NODES_COUNT];
        list->cap_ = NODES_COUNT;

        Node a;
        a.geom_ = new CUDA_RAY::Plane(1.f);
        a.shader_ = new CUDA_RAY::Lambert(new CheckerTexture(Color(1.f, 0.f, 0.f), Color(0.f, 1.f, 0.f)));

        list->addElem(a);

        Node b;
        b.geom_ = new CUDA_RAY::Sphere(Vector(20, 50, 60), 10);
        b.shader_ = new CUDA_RAY::Lambert(new CheckerTexture(Color(0.5f, 0.f, 0.f), Color(0.1f, 0.7f, 0.3f), 5.f));
        list->addElem(b);


        /*list->list_[0].geom_ = new CUDA_RAY::Plane(1.f);
        list->list_[0].shader_ = new CUDA_RAY::CheckerShader(Color(1.f, 0.f, 0.f),
            Color(0.f, 1.f, 0.f));
        list->size_ = NODES_COUNT;*/
    }
}

__global__ void setupLights(CudaList<Light>* lights)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {

        lights->list_ = new Light[LIGHTS_COUNT];
        lights->cap_ = LIGHTS_COUNT;

        Light a;
        a.pos_ = Vector(0, 100, 10);
        a.intensity_ = 11000.f;

        Light b;
        b.pos_ = Vector(40, 100, 10);
        b.intensity_ = 11000.f;
        lights->addElem(b);
        lights->addElem(a);

        /*a.pos_ = Vector(35, 5, 200);
        lights->addElem(a);*/

        /*a.pos_ = Vector(35, 5, 300);
        lights->addElem(a);*/
    }
}

__global__ void freeWorld(Camera** cam, CudaList<Node>* list, CudaList<Light>* lights)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        delete *cam;

        /*delete lights[0];
        delete lights[1];
        delete[] lights;*/

        delete list->list_[0].shader_;
        delete list->list_[0].geom_;
        delete list->list_[1].shader_;
        delete list->list_[1].geom_;
        // delete[] list->list_;
    }
}

int main(int argc, char* args[])
{
    CXX::SdlObject& sdl = CXX::SdlObject::instance();

    int num_pixels = sdl.frameWidth() * sdl.frameHeight();
    size_t vfb_size = num_pixels * sizeof(CUDA_RAY::Color);

    CUDA_RAY::Color* vfb = nullptr;
    checkCudaErrors(cudaMallocManaged((void**)&vfb, vfb_size));


    // Camera setup
    Camera** camera;
    checkCudaErrors(cudaMalloc((void**)&camera, sizeof(CUDA_RAY::Camera*)));
    setupCamera<<<1, 1>>>(camera);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    // Geometries and shaders setup
    CudaList<Node>* objects;
    checkCudaErrors(cudaMalloc(&objects, sizeof(CudaList<Node>)));
    setupNodes<<<1, 1>>> (objects);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());


    // Lightning setup
    CudaList<Light>* lights;
    checkCudaErrors(cudaMalloc(&lights, sizeof(CudaList<Light>)));
    setupLights<<<1, 1>>>(lights);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    int tx = 8;
    int ty = 8;
    dim3 blocks((sdl.frameHeight() / tx), (sdl.frameWidth() / ty));
    dim3 threads(tx, ty);

    // Render our buffer
    clock_t start, stop;
    start = clock();

    render<<<blocks, threads>>>(camera, objects, lights, vfb, sdl.frameWidth(), sdl.frameWidth());

    // Wait for GPU to finish before accessing on host
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;

    std::cout << "took " << timer_seconds << " seconds.\n";


    sdl.displayVFB<CUDA_RAY::Color>(vfb);
    sdl.waitForUserExit();

    freeWorld<<<1, 1>>>(camera, objects, lights);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaFree(vfb));
    checkCudaErrors(cudaFree(objects));
    checkCudaErrors(cudaFree(camera));
    checkCudaErrors(cudaFree(lights));
    cudaDeviceReset();

	return 0;
}