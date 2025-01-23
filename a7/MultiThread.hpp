#pragma once

#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <memory>
#include "Scene.hpp"

static const int MAX_THREADS = 20;
std::atomic<int> num_threads(0);
std::mutex mtx;
std::condition_variable cv;

void thread_func(const Scene& scene, const Ray& ray, float spp, Vector3f& dest)
{
    {
        std::lock_guard<std::mutex> lock(mtx);
        num_threads += 1;
    }
    dest += scene.castRay(ray, 0) / spp;
    {
        std::lock_guard<std::mutex> lock(mtx);
        num_threads -= 1;
    }
    cv.notify_one();
}

float multi_cast(const Scene& scene, const Ray& ray, float spp, Vector3f& dest)
{
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [](){ return num_threads < MAX_THREADS; });
    std::thread t(thread_func, std::cref(scene), std::cref(ray), spp, std::ref(dest));
    t.detach();
}

void thread_sync()
{
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [](){ return num_threads == 0; });
}