//
// Created by LEI XU on 5/13/19.
//
#pragma once
#ifndef RAYTRACING_VECTOR_H
#define RAYTRACING_VECTOR_H

#include <iostream>
#include <cmath>
#include <algorithm>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define HOST_DEVICE __host__ __device__
#define __MIN(X, Y) (X < Y ? X : Y)
#define __MAX(X, Y) (X > Y ? X : Y)
#else
#define HOST_DEVICE
#define __MIN(X, Y) std::min(X, Y)
#define __MAX(X, Y) std::max(X, Y)
#endif

class Vector3f {
public:
    float x, y, z;
    HOST_DEVICE inline Vector3f() : x(0), y(0), z(0) {}
    HOST_DEVICE inline Vector3f(float xx) : x(xx), y(xx), z(xx) {}
    HOST_DEVICE inline Vector3f(float xx, float yy, float zz) : x(xx), y(yy), z(zz) {}
    HOST_DEVICE inline Vector3f operator * (const float &r) const { return Vector3f(x * r, y * r, z * r); }
    HOST_DEVICE inline Vector3f operator / (const float &r) const { return Vector3f(x / r, y / r, z / r); }

    HOST_DEVICE inline float norm() {return std::sqrt(x * x + y * y + z * z);}
    HOST_DEVICE inline Vector3f normalized() {
        float n = std::sqrt(x * x + y * y + z * z);
        return Vector3f(x / n, y / n, z / n);
    }

    HOST_DEVICE inline Vector3f operator * (const Vector3f &v) const { return Vector3f(x * v.x, y * v.y, z * v.z); }
    HOST_DEVICE inline Vector3f operator - (const Vector3f &v) const { return Vector3f(x - v.x, y - v.y, z - v.z); }
    HOST_DEVICE inline Vector3f operator + (const Vector3f &v) const { return Vector3f(x + v.x, y + v.y, z + v.z); }
    HOST_DEVICE inline Vector3f operator - () const { return Vector3f(-x, -y, -z); }
    HOST_DEVICE inline Vector3f& operator += (const Vector3f &v) { x += v.x, y += v.y, z += v.z; return *this; }
    HOST_DEVICE inline friend Vector3f operator * (const float &r, const Vector3f &v)
    { return Vector3f(v.x * r, v.y * r, v.z * r); }
    inline friend std::ostream & operator << (std::ostream &os, const Vector3f &v)
    { return os << v.x << ", " << v.y << ", " << v.z; }


    HOST_DEVICE inline static Vector3f Min(const Vector3f &p1, const Vector3f &p2) {
        return Vector3f(__MIN(p1.x, p2.x), __MIN(p1.y, p2.y),
                       __MIN(p1.z, p2.z));
    }

    HOST_DEVICE inline static Vector3f Max(const Vector3f &p1, const Vector3f &p2) {
        return Vector3f(__MAX(p1.x, p2.x), __MAX(p1.y, p2.y),
                       __MAX(p1.z, p2.z));
    }
    HOST_DEVICE inline float operator[](int index) const {
    return (&x)[index];
}
};
//HOST_DEVICE inline double Vector3f::operator[]


class Vector2f
{
public:
    Vector2f() : x(0), y(0) {}
    Vector2f(float xx) : x(xx), y(xx) {}
    Vector2f(float xx, float yy) : x(xx), y(yy) {}
    Vector2f operator * (const float &r) const { return Vector2f(x * r, y * r); }
    Vector2f operator + (const Vector2f &v) const { return Vector2f(x + v.x, y + v.y); }
    float x, y;
};

HOST_DEVICE inline Vector3f lerp(const Vector3f &a, const Vector3f& b, const float &t)
{ return a * (1 - t) + b * t; }

HOST_DEVICE inline Vector3f normalize(const Vector3f &v)
{
    float mag2 = v.x * v.x + v.y * v.y + v.z * v.z;
    if (mag2 > 0) {
        float invMag = 1 / sqrtf(mag2);
        return Vector3f(v.x * invMag, v.y * invMag, v.z * invMag);
    }

    return v;
}

HOST_DEVICE inline float dotProduct(const Vector3f &a, const Vector3f &b)
{ return a.x * b.x + a.y * b.y + a.z * b.z; }

HOST_DEVICE inline Vector3f crossProduct(const Vector3f &a, const Vector3f &b)
{
    return Vector3f(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
    );
}

#undef __MAX
#undef __MIN

#endif //RAYTRACING_VECTOR_H
