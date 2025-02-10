// clang-format off
//
// Created by goksu on 4/6/19.
//

#include <algorithm>
#include <vector>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>
#include <queue>


rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions)
{
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices)
{
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols)
{
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}

auto to_vec4(const Eigen::Vector3f& v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}

bool insideTriangle(float x, float y, const Vector3f* _v) {
    Vector3f P;
    P << x , y , 0;
    float z1 = (P - _v[0]).cross(_v[0] - _v[1])(2);
    float z2 = (P - _v[1]).cross(_v[1] - _v[2])(2);
    float z3 = (P - _v[2]).cross(_v[2] - _v[0])(2);
    if ((z1<0 && z2<0 && z3<0) || (z1>0 && z2>0 && z3>0))
        return true;
    return false;
}

static bool insideTriangle(int x, int y, const Vector3f* _v)
{   
    // TODO : Implement this function to check if the point (x, y) is inside the triangle represented by _v[0], _v[1], _v[2]
    return insideTriangle((float)x, (float)y, _v);
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector3f* v)
{
    float c1 = (x*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*y + v[1].x()*v[2].y() - v[2].x()*v[1].y()) / (v[0].x()*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*v[0].y() + v[1].x()*v[2].y() - v[2].x()*v[1].y());
    float c2 = (x*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*y + v[2].x()*v[0].y() - v[0].x()*v[2].y()) / (v[1].x()*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*v[1].y() + v[2].x()*v[0].y() - v[0].x()*v[2].y());
    float c3 = (x*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*y + v[0].x()*v[1].y() - v[1].x()*v[0].y()) / (v[2].x()*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*v[2].y() + v[0].x()*v[1].y() - v[1].x()*v[0].y());
    return {c1,c2,c3};
}

void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type)
{
    auto& buf = pos_buf[pos_buffer.pos_id];
    auto& ind = ind_buf[ind_buffer.ind_id];
    auto& col = col_buf[col_buffer.col_id];

    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;
    for (auto& i : ind)
    {
        Triangle t;
        Eigen::Vector4f v[] = {
                mvp * to_vec4(buf[i[0]], 1.0f),
                mvp * to_vec4(buf[i[1]], 1.0f),
                mvp * to_vec4(buf[i[2]], 1.0f)
        };
        //Homogeneous division
        for (auto& vec : v) {
            vec /= vec.w();
        }
        //Viewport transformation
        for (auto & vert : v)
        {
            vert.x() = 0.5*width*(vert.x()+1.0);
            vert.y() = 0.5*height*(vert.y()+1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i)
        {
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
        }

        auto col_x = col[i[0]];
        auto col_y = col[i[1]];
        auto col_z = col[i[2]];

        t.setColor(0, col_x[0], col_x[1], col_x[2]);
        t.setColor(1, col_y[0], col_y[1], col_y[2]);
        t.setColor(2, col_z[0], col_z[1], col_z[2]);

        rasterize_triangle(t);
    }
}


typedef Eigen::Vector2i Index2D;
typedef std::priority_queue<Index2D, std::vector<Index2D>, std::function<bool(const Index2D&, const Index2D&)>> Index2D_PQ;

//Modified form https://rosettacode.org/wiki/Bitmap/Bresenhanm%27s_line_algorithm#c
void bresenham(int x0, int y0, int x1, int y1, Index2D_PQ& pixels) {
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int erro = (dx > dy ? dx : -dy) / 2;
    while(pixels.push(Index2D{x0, y0}), x0 != x1 || y0 != y1) {
        int e2 = erro;
        if(e2 > -dx) { erro -= dy; x0 += sx;}
        if(e2 <  dy) { erro += dx; y0 += sy;}
    } 
}

float get_z_interpolate(float x, float y, const Triangle& t) {
    auto[alpha, beta, gamma] = computeBarycentric2D(x, y, t.v);
    auto v = t.toVector4();
    float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
    float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
    z_interpolated *= w_reciprocal;
    return z_interpolated;
}

//Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle& t) {
    auto v = t.toVector4();
    auto color = t.getColor();
    auto cmp = [](const Index2D& a, const Index2D& b) {
        if (a.y() == b.y()) {
            return a.x() > b.x();
        }
        return a.y() > b.y();
    };
    Index2D_PQ edge_pixels(cmp);
    int x1 = floorf(v[0](0)), x2 = floorf(v[1](0)), x3 = floorf(v[2](0));
    int y1 = floorf(v[0](1)), y2 = floorf(v[1](1)), y3 = floorf(v[2](1));
    bresenham(x1, y1, x2, y2, edge_pixels);
    bresenham(x2, y2, x3, y3, edge_pixels);
    bresenham(x3, y3, x1, y1, edge_pixels);
    int index;
    while (!edge_pixels.empty()) {
        auto pixel = edge_pixels.top();
        edge_pixels.pop();
        int px = pixel.x(), py = pixel.y();
        // handle edge pixel
        int index;
        for (int dy = 0; dy < 2; dy ++) {
            for (int dx = 0; dx < 2; dx ++) {
                float x0 = (float)px + 0.5f*dx, y0 = (float)py + 0.5f*dy;
                if (insideTriangle(x0, y0, t.v)) {
                    float z = get_z_interpolate(x0, y0, t);
                    index = get_index(px, py) + (dy*2 + dx);
                    if (z < depth_buf[index]) {
                        depth_buf[index] = z;
                        frame_buf[index] = t.getColor();
                    }
                }
            }
        }
        // handle internal pixels
        if (py == edge_pixels.top().y()) {
            for (int x = px + 1; x < edge_pixels.top().x(); x++) {
                float z = get_z_interpolate(x, py, t);
                index = get_index(x, py);
                if (z < depth_buf[index]) {
                    for (int i = 0; i < 4; i++) {
                        depth_buf[index + i] = z;
                        frame_buf[index + i] = t.getColor();
                    }
                }
            }
        }
    }
}

void rst::rasterizer::set_model(const Eigen::Matrix4f& m)
{
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f& v)
{
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f& p)
{
    projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff)
{
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});std::fill(frame_buf_final.begin(), frame_buf_final.end(), Eigen::Vector3f{0, 0, 0});
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h)
{
    frame_buf.resize(w * h * 4);
    depth_buf.resize(w * h * 4);
    frame_buf_final.resize(w * h);
}

int rst::rasterizer::get_index(int x, int y)
{
    return ((height-1-y)*width + x)*4;
}

void rst::rasterizer::set_pixel(const Eigen::Vector3f& point, const Eigen::Vector3f& color)
{
    //old index: auto ind = point.y() + point.x() * width;
    auto ind = (height-1-point.y())*width + point.x();
    frame_buf[ind] = color;

}

std::vector<Eigen::Vector3f>& rst::rasterizer::frame_buffer() {
    for (int i = 0; i < frame_buf_final.size(); i++) {
        for (int j = 0; j < 4; j++ ) {
            frame_buf_final[i] += frame_buf[i*4 + j];
        }
        frame_buf_final[i] /= 4;
    }
    return frame_buf_final;
}
// clang-format on