#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "Triangle.hpp"
#include "Vector.hpp"

class Triangle_d
{
public:
    Vector3f v0, v1, v2; // vertices A, B ,C , counter-clockwise order
    Vector3f e1, e2;     // 2 edges v1-v0, v2-v0;
    Vector3f t0, t1, t2; // texture coords
    Vector3f normal;
    float area;
    Material* m;

    __host__ __device__ Triangle_d(Vector3f _v0, Vector3f _v1, Vector3f _v2, Material* _m = nullptr)
        : v0(_v0), v1(_v1), v2(_v2), m(_m)
    {
        e1 = v1 - v0;
        e2 = v2 - v0;
        normal = normalize(crossProduct(e1, e2));
        area = crossProduct(e1, e2).norm()*0.5f;
    }
    void getSurfaceProperties(const Vector3f& P, const Vector3f& I,
                              const uint32_t& index, const Vector2f& uv,
                              Vector3f& N, Vector2f& st) const 
    {
        N = normal;
        //        throw std::runtime_error("triangle::getSurfaceProperties not
        //        implemented.");
    }
    __host__ __device__ float getArea(){
        return area;
    }
    __host__ __device__ bool hasEmit(){
        return m->hasEmission();
    }
};

struct Intersection_d
{
    HOST_DEVICE Intersection_d(){
        happened=false;
        coords=Vector3f();
        normal=Vector3f();
        distance= FLT_MAX;
        obj =nullptr;
        m=nullptr;
    }
    bool happened;
    Vector3f coords;
    Vector3f tcoords;
    Vector3f normal;
    Vector3f emit;
    float distance;
    Triangle_d* obj;
    Material* m;
};

__device__ inline
void sampleTriangle(Triangle_d& t, Intersection& inter, float& pdf, curandState* state)
{
    float x = sqrt(curand_uniform(state)), y = curand_uniform(state);
    inter.coords = t.v0 * (1.f - x) + t.v1 * (x * (1.f - y)) + t.v2 * (x * y);
    inter.normal = t.normal;
    pdf = 1.f / t.area;
}

__device__ inline
Vector3f sampleMaterial(Material* m, const Vector3f &wi, const Vector3f &N, curandState* state)
{
    switch(m->m_type){
        case DIFFUSE:
        {
            // uniform sample on the hemisphere
            float x_1 = curand_uniform(state), x_2 = curand_uniform(state);
            float z = abs(1.0f - 2.0f * x_1);
            float r = sqrt(1.0f - z * z), phi = 2 * M_PI * x_2;
            Vector3f localRay(r*cos(phi), r*sin(phi), z);
            return m->toWorld(localRay, N);
            
            break;
        }
    }
}

__device__
inline void sampleLight(Triangle_d* ts, int num_triangles, Intersection& inter, float& pdf, curandState* state)
{
    float emit_area_sum = 0;
    for (uint k = 0; k < num_triangles; k++)
    {
        if (ts[k].hasEmit())
            emit_area_sum += ts[k].getArea();
    }
    float p = curand_uniform(state) * emit_area_sum;
    emit_area_sum = 0;
    for (uint k = 0; k < num_triangles; k++)
    {
        if (ts[k].hasEmit())
        {
            emit_area_sum += ts[k].getArea();
            if (p <= emit_area_sum)
            {
                sampleTriangle(ts[k], inter, pdf, state);
                break;
            }
        }
    }
}

inline void triangleToDevice(Triangle_d* dst, Triangle& src)
{
    cudaMemcpy(&dst->v0, &src.v0, sizeof(Vector3f), cudaMemcpyHostToDevice);
    cudaMemcpy(&dst->v1, &src.v1, sizeof(Vector3f), cudaMemcpyHostToDevice);
    cudaMemcpy(&dst->v2, &src.v2, sizeof(Vector3f), cudaMemcpyHostToDevice);
    cudaMemcpy(&dst->e1, &src.e1, sizeof(Vector3f), cudaMemcpyHostToDevice);
    cudaMemcpy(&dst->e2, &src.e2, sizeof(Vector3f), cudaMemcpyHostToDevice);
    cudaMemcpy(&dst->t0, &src.t0, sizeof(Vector3f), cudaMemcpyHostToDevice);
    cudaMemcpy(&dst->t1, &src.t1, sizeof(Vector3f), cudaMemcpyHostToDevice);
    cudaMemcpy(&dst->t2, &src.t2, sizeof(Vector3f), cudaMemcpyHostToDevice);
    cudaMemcpy(&dst->area, &src.area, sizeof(float), cudaMemcpyHostToDevice);
    Material* tmp;
    cudaMalloc(&tmp, sizeof(Material));
    cudaMemcpy(tmp, src.m, sizeof(Material), cudaMemcpyHostToDevice);
    cudaMemcpy(&dst->m, &tmp, sizeof(void*), cudaMemcpyHostToDevice);
}