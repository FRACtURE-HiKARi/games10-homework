//
// Created by LEI XU on 5/16/19.
//

#ifndef RAYTRACING_INTERSECTION_H
#define RAYTRACING_INTERSECTION_H
#include "Vector.hpp"
#include "Material.hpp"
class Object;
class Sphere;

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <float.h>
#define HOST_DEVICE __host__ __device__
#define __MAX DBL_MAX
#else
#define HOST_DEVICE
#define __MAX std::numeric_limits<double>::max()
#endif

struct Intersection
{
    HOST_DEVICE Intersection(){
        happened=false;
        coords=Vector3f();
        normal=Vector3f();
        distance= __MAX;
        obj =nullptr;
        m=nullptr;
    }
    bool happened;
    Vector3f coords;
    Vector3f tcoords;
    Vector3f normal;
    Vector3f emit;
    double distance;
    Object* obj;
    Material* m;
};
#undef __MAX
#endif //RAYTRACING_INTERSECTION_H
