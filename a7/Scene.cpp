//
// Created by Göksu Güvendiren on 2019-05-14.
//

#include "Scene.hpp"


void Scene::buildBVH() {
    printf(" - Generating BVH...\n\n");
    this->bvh = new BVHAccel(objects, 1, BVHAccel::SplitMethod::NAIVE);
}

Intersection Scene::intersect(const Ray &ray) const
{
    return this->bvh->Intersect(ray);
}

void Scene::sampleLight(Intersection &pos, float &pdf) const
{
    float emit_area_sum = 0;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        if (objects[k]->hasEmit()){
            emit_area_sum += objects[k]->getArea();
        }
    }
    float p = get_random_float() * emit_area_sum;
    emit_area_sum = 0;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        if (objects[k]->hasEmit()){
            emit_area_sum += objects[k]->getArea();
            if (p <= emit_area_sum){
                objects[k]->Sample(pos, pdf);
                break;
            }
        }
    }
}

bool Scene::trace(
        const Ray &ray,
        const std::vector<Object*> &objects,
        float &tNear, uint32_t &index, Object **hitObject)
{
    *hitObject = nullptr;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        float tNearK = kInfinity;
        uint32_t indexK;
        Vector2f uvK;
        if (objects[k]->intersect(ray, tNearK, indexK) && tNearK < tNear) {
            *hitObject = objects[k];
            tNear = tNearK;
            index = indexK;
        }
    }


    return (*hitObject != nullptr);
}

// Implementation of Path Tracing
Vector3f Scene::castRay(const Ray &ray, int depth) const
{
    // TO DO Implement Path Tracing Algorithm here
    Vector3f L_dir(0);
    Vector3f L_indir(0);
    // get intersection of the ray
    Intersection inter = intersect(ray);
    if (!inter.happened)
        return Vector3f(0);
    if (inter.m->hasEmission())
        return inter.m->getEmission();

    Vector3f p = inter.coords;
    Material* m = inter.m;
    // calculate direct light
    Intersection inter_light;
    float pdf_light;
    sampleLight(inter_light, pdf_light);
    // test if blocked
    Vector3f x = inter_light.coords;
    Vector3f ws = (x - p).normalized();
    Vector3f wo = ray.direction;
    Vector3f N = inter.normal;
    Ray dir_ray(p, ws);
    Intersection block_test = intersect(dir_ray);
    if (block_test.distance - (x - p).norm() > -0.005)
    {
        Vector3f NN = inter_light.normal;
        L_dir = inter_light.emit * m->eval(wo, ws, N) * dotProduct(ws, N) * dotProduct(-ws, NN) / dotProduct(x-p, x-p) / pdf_light;
    }

    // calculate indirect light
    if (get_random_float() > RussianRoulette)
        return L_dir;
    Vector3f wi = m->sample(wo, N);
    Ray indir_ray(p, wi);
    Intersection nonemit_inter = intersect(indir_ray);
    if (nonemit_inter.happened && !nonemit_inter.m->hasEmission())
    {
        L_indir = castRay(indir_ray, depth + 1) * m->eval(wo, wi, N) * dotProduct(wi, N) / m->pdf(wo, wi, N) / RussianRoulette;
    }
return L_dir + L_indir;
}
