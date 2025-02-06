//
// Created by LEI XU on 5/13/19.
//
#pragma once
#ifndef RAYTRACING_OBJECT_H
#define RAYTRACING_OBJECT_H

#include "Vector.hpp"
#include "global.hpp"
#include "Bounds3.hpp"
#include "Ray.hpp"
#include "Intersection.hpp"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <float.h>
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

class Object
{
public:
    Object() {}
    //virtual ~Object() {}
    virtual bool intersect(const Ray& ray) = 0;
    virtual bool intersect(const Ray& ray, float &, uint32_t &) const = 0;
    virtual Intersection getIntersection(Ray _ray) = 0;
    virtual void getSurfaceProperties(const Vector3f &, const Vector3f &, const uint32_t &, const Vector2f &, Vector3f &, Vector2f &) const = 0;
    virtual Vector3f evalDiffuseColor(const Vector2f &) const =0;
    virtual Bounds3 getBounds()=0;
    HOST_DEVICE virtual float getArea()=0;
    virtual void Sample(Intersection &pos, float &pdf)=0;
    HOST_DEVICE virtual bool hasEmit()=0;
};



#endif //RAYTRACING_OBJECT_H
