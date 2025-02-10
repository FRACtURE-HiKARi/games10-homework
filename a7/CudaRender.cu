#include "Scene.hpp"
#include "CudaRender.h"
#include "CudaHelper.hpp"
#include <thread>
#include <chrono>

#define MAX_DEPTH 64
#define MAX_TRIANGLES 1024
#define P_RR 0.8
#define COMP(X) X.x, X.y, X.z
__device__ float spp;
__device__ float scale_d;
__device__ float imageAspectRatio_d;
__device__ int width_d;
__device__ int height_d;
__device__ Vector3f* eye_pos_d;
Vector3f* framebuffer;
Triangle_d* triangles;
Triangle* triangles_h;
__device__ int num_triangles = 0;
int num_pixels;

inline void printLastErr()
{
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        printf("Last Error: %s\n", cudaGetErrorString(launchErr));
    }
    else {
        printf("Success. \n");
    }
}

void init_objects(const Scene& scene)
{
    triangles_h = (Triangle*)malloc(sizeof(Triangle) * MAX_TRIANGLES);
    int i = 0;
    for (auto o: scene.objects)
    {
        for (auto t: ((MeshTriangle*)o)->triangles)
        {
            if (i < MAX_TRIANGLES)
            {
                triangles_h[i++] = t;
            }
        }
    }
    cudaMalloc(&triangles, i * sizeof(Triangle_d));
    cudaMemcpyToSymbol(num_triangles, &i, sizeof(int));
    printf("Num Triangles: %d\n", i);
    for (int j = 0; j < i; j++)
    {
        triangleToDevice(&triangles[j], triangles_h[j]);
    }
}

void init_memory(const Scene& scene, int spp)
{
    float scale = tan(deg2rad(scene.fov * 0.5));
    float imageAspectRatio = scene.width / (float)scene.height;
    cudaMemcpyToSymbol(scale_d, &scale, sizeof(float));
    cudaMemcpyToSymbol(imageAspectRatio_d, &imageAspectRatio, sizeof(float));
    cudaMemcpyToSymbol(width_d, &scene.width, sizeof(int));
    cudaMemcpyToSymbol(height_d, &scene.height, sizeof(int));
    
    cudaMalloc(&framebuffer, num_pixels * sizeof(Vector3f));
    cudaMemset(framebuffer, 0, num_pixels * sizeof(Vector3f));
    init_objects(scene);
}

void clean_up()
{
    free(triangles_h);
    cudaFree(framebuffer);
    cudaFree(triangles);
    Vector3f* tmp;
    cudaMemcpyFromSymbol(&tmp, eye_pos_d, sizeof(Vector3f*));
    cudaFree(tmp);
}

__device__ Intersection_d getClosest(const Ray& ray, Triangle_d* ts, int depth)
{
    Intersection_d result;
    for (int i = 0; i < num_triangles; i++)
    {
        auto t = ts[i];
        Intersection_d inter = t.getIntersection(ray);
        if (inter.happened && inter.distance < result.distance)
        {
            result = inter;
        }
    }
    return result;
}

__device__ Intersection_d getClosest(const Ray &ray, Triangle_d* ts)
{
    return getClosest(ray, ts, 0);
}

// Implementation of PathTracing in CUDA
__device__ Vector3f trace(const Ray& ray, Triangle_d* ts, curandState* state)
{
    int depth = 0;
    Vector3f stack_dir[MAX_DEPTH];
    Vector3f stack_multiplier[MAX_DEPTH];

    Ray currentRay = ray;
    while (depth < MAX_DEPTH)
    {
        stack_dir[depth] = Vector3f(0);
        stack_multiplier[depth] = Vector3f(0);
        Intersection_d inter = getClosest(currentRay, ts);
        if (!inter.happened){
            // return Vector3f(0);
            break;
        }
        if (inter.m->hasEmission()){
            stack_dir[depth] = inter.m->getEmission();
            break;
        }
        Vector3f p = inter.coords;
        Material* m = inter.m;
        float pdf_light;
        Intersection inter_light;
        sampleLight(ts, num_triangles, inter_light, pdf_light, state);
        // test if blocked
        Vector3f x = inter_light.coords;
        Vector3f ws = (x - p).normalized();
        Vector3f wo = currentRay.direction;
        Vector3f N = inter.normal;
        Ray dir_ray(p, ws);
        Intersection_d block_test = getClosest(dir_ray, ts);
        if (block_test.distance - (x - p).norm() > -0.005)
        {
            Vector3f NN = inter_light.normal;
            stack_dir[depth] = inter_light.emit * m->eval(wo, ws, N) * dotProduct(ws, N) * dotProduct(-ws, NN) / dotProduct(x-p, x-p) / pdf_light;
        }
        if (curand_uniform(state) > P_RR){
            break;
        }
        
        //Vector3f wi = m->sample(wo, N);
        Vector3f wi = sampleMaterial(m, -wo, N, state);
        Ray indir_ray(p, wi);
        Intersection_d nonemit_inter = getClosest(indir_ray, ts);
        if (nonemit_inter.happened && !nonemit_inter.m->hasEmission())
        {
            //L_indir = trace(indir_ray, ts, depth + 1, state) * m->eval(wo, wi, N) * dotProduct(wi, N) / m->pdf(wo, wi, N) / P_RR;
            stack_multiplier[depth] = m->eval(wo, wi, N) * dotProduct(wi, N) / m->pdf(wo, wi, N) / P_RR;
            currentRay = indir_ray;
            depth += 1;
            continue;
        }
    }
    // accumulate all lights
    Vector3f result = stack_dir[depth--];
    while (depth >= 0)
    {
        result = result * stack_multiplier[depth] + stack_dir[depth];
        depth -= 1;
    }
    return result;
}

// kernel thread
__global__
void CUDA_PT(Vector3f* fb, Triangle_d* ts, int spp, curandState* states)
{
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 & blockIdx.y == 0)
    {
        eye_pos_d = new Vector3f(278.0f, 273.0f, -800.0f);
    }
    int xi = threadIdx.x + blockIdx.x * blockDim.x;
    int yi = threadIdx.y + blockIdx.y * blockDim.y;
    int tid = xi + width_d * yi;
    float x = (2 * (xi + 0.5) / (float)width_d - 1) * imageAspectRatio_d * scale_d;
    float y = (1 - 2 * (yi + 0.5) / (float)height_d) * scale_d;
    curand_init(1234, tid, 0, &states[tid]);
    Vector3f result;
    Vector3f dir = normalize(Vector3f(-x, y, 1));
    Ray ray(*eye_pos_d, dir);
    for (int i = 0; i < spp; i++)
    {
        result += trace(ray, ts, &states[tid]) / (float)spp;
    }
    fb[tid] = result;
}

void gpuWaitThread() {cudaDeviceSynchronize();}

#define BLOCK_DIM 32
void cudaRender(Vector3f* fb_h, const Scene& scene, int spp)
{
    dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
    dim3 gridDim(
        (scene.width + BLOCK_DIM - 1) / BLOCK_DIM,
        (scene.height + BLOCK_DIM - 1) / BLOCK_DIM
    );
    num_pixels = scene.width * scene.height;
    printf("Init memory...\n");
    init_memory(scene, spp);
    printLastErr();

    curandState* states_d;
    cudaMalloc(&states_d, num_pixels * sizeof(curandState));

    printf("Starting Kernel\n");
    CUDA_PT<<<gridDim, blockDim>>>(framebuffer, triangles, spp, states_d);
    printLastErr();
    printf("Waiting for sync\n");
    cudaDeviceSynchronize();
    printLastErr();
    cudaMemcpy(fb_h, framebuffer, sizeof(Vector3f) * num_pixels, cudaMemcpyDeviceToHost);
}