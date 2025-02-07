#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "Triangle.hpp"
#include "Vector.hpp"

class Triangle_d;

struct Intersection_d
{
    __host__ __device__ Intersection_d(){
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
    __host__ Triangle_d (const Triangle& src)
        : v0(src.v0), v1(src.v1), v2(src.v2), e1(src.e1), e2(src.e2), 
        t0(src.t0), t1(src.t1), t2(src.t2), normal(src.normal), area(src.area)
    {
        cudaMalloc(&m, sizeof(Material));
        cudaMemcpy(m, src.m, sizeof(Material), cudaMemcpyHostToDevice);
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
    __host__ __device__ inline 
    Intersection_d getIntersection(const Ray& ray)
    {
        Intersection_d inter;
        if (dotProduct(ray.direction, normal) > 0)
            return inter;
        double u, v, t_tmp = 0;
        Vector3f pvec = crossProduct(ray.direction, e2);
        double det = dotProduct(e1, pvec);
        if (abs(det) < EPSILON)
            return inter;

        double det_inv = 1. / det;
        Vector3f tvec = ray.origin - v0;
        u = dotProduct(tvec, pvec) * det_inv;
        if (u < 0 || u > 1)
            return inter;
        Vector3f qvec = crossProduct(tvec, e1);
        v = dotProduct(ray.direction, qvec) * det_inv;
        if (v < 0 || u + v > 1)
            return inter;
        t_tmp = dotProduct(e2, qvec) * det_inv;
        // TODO find ray triangle intersection
        /*
            bool happened;
            Vector3f coords;
            Vector3f normal;
            double distance;
            Object* obj;
            Material* m;
        */
        //if(t_tmp < 0) return inter;
        inter.happened = t_tmp > 0 && u>0 && v>0 && (1-u-v>0);
        inter.coords = ray(t_tmp);//
        inter.emit = m->getEmission();
        inter.normal = normal;
        inter.distance = t_tmp;//
        inter.obj = this;
        inter.m = m;

        return inter;
    }
};

__device__ inline
void sampleTriangle(Triangle_d& t, Intersection& inter, float& pdf, curandState* state)
{
    float x = sqrt(curand_uniform(state)), y = curand_uniform(state);
    inter.coords = t.v0 * (1.f - x) + t.v1 * (x * (1.f - y)) + t.v2 * (x * y);
    inter.normal = t.normal;
    pdf = 1.f / t.area;
    inter.emit = t.m->m_emission;
}

__device__ inline
Vector3f toWorld(const Vector3f &a, const Vector3f &N){
    Vector3f B, C;
    if (abs(N.x) > abs(N.y)){
        float invLen = 1.0f / sqrt(N.x * N.x + N.z * N.z);
        C = Vector3f(N.z * invLen, 0.0f, -N.x *invLen);
    }
    else {
        float invLen = 1.0f / sqrt(N.y * N.y + N.z * N.z);
        C = Vector3f(0.0f, N.z * invLen, -N.y *invLen);
    }
    B = crossProduct(C, N);
    return a.x * B + a.y * C + a.z * N;
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
            return toWorld(localRay, N);
            
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
                pdf *= (ts[k].getArea() / emit_area_sum);
                break;
            }
        }
    }
}

inline void triangleToDevice(Triangle_d* dst, Triangle& src)
{
    Triangle_d host_copy(src);
    cudaMemcpy(dst, &host_copy, sizeof(Triangle_d), cudaMemcpyHostToDevice);
}