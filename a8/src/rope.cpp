#include <iostream>
#include <vector>

#include "CGL/vector2D.h"

#include "mass.h"
#include "rope.h"
#include "spring.h"

namespace CGL {

    Rope::Rope(Vector2D start, Vector2D end, int num_nodes, float node_mass, float k, vector<int> pinned_nodes)
    {
        // TODO (Part 1): Create a rope starting at `start`, ending at `end`, and containing `num_nodes` nodes.
        
        masses.reserve(num_nodes);
        for (int i = 0; i < num_nodes; i++)
        {
            float t = (float) i / (num_nodes - 1);
            Vector2D pos = (1 - t) * start + t * end;
            masses.emplace_back(new Mass(pos, node_mass, false));
            if (i > 0)
                springs.emplace_back(new Spring(masses[i - 1], masses[i], k));
        }
        // Comment-in this part when you implement the constructor
        for (auto &i : pinned_nodes) {
            masses[i]->pinned = true;
        }
    }

    void Rope::simulateEuler(float delta_t, Vector2D gravity)
    {
        for (auto &s : springs)
        {
            // TODO (Part 2): Use Hooke's law to calculate the force on a node
            auto dist = (s->m1->position - s->m2->position).norm();
            auto f_ba = s->k * (dist - s->rest_length) / dist * (s->m2->position - s->m1->position);
            s->m1->forces += f_ba;
            s->m2->forces -= f_ba;
        }

        for (auto &m : masses)
        {
            if (!m->pinned)
            {
                // TODO (Part 2): Add the force due to gravity, then compute the new velocity and position
                m->forces += m->mass * gravity;
                // TODO (Part 2): Add global damping
                
                m->forces += -0.01 * m->velocity;
                m->velocity += delta_t / m->mass * m->forces;
                m->position += delta_t * m->velocity;
            }

            // Reset all forces on each mass
            m->forces = Vector2D(0, 0);
        }
    }

    void Rope::simulateVerlet(float delta_t, Vector2D gravity)
    {
            // TODO (Part 3): Simulate one timestep of the rope using explicit Verlet （solving constraints)
        
        const int ConstraintLoops = 1;
        for (int i = 0; i < ConstraintLoops; i++){
            for (auto &s: springs)
            {
                auto dx = s->m2->position - s->m1->position;
                auto dist = dx.norm();
                auto err = dist - s->rest_length;
                auto n = dx / dist;
                if (!s->m1->pinned && !s->m2->pinned)
                {
                    s->m1->move(0.5 * err * n);
                    s->m2->move(-0.5 * err * n);
                }
                else if (s->m1->pinned)
                {
                    s->m2->move(-err * n);
                }
                else if (s->m2->pinned)
                {
                    s->m1->move(err * n);
                }
            }
        }
        for (auto &m : masses)
        {
            if (!m->pinned)
            {
                Vector2D temp_position = m->position;
                // TODO (Part 3.1): Set the new position of the rope mass
                
                // TODO (Part 4): Add global Verlet damping
                m->forces += m->mass * gravity;
                m->position = m->position + (1 - 0.0002f)*(m->position - m->last_position) + m->forces / m->mass * delta_t * delta_t;
                m->last_position = temp_position;
                m->forces = Vector2D(0, 0);
            }
        }
    }
}
