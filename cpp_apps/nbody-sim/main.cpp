//*********************************************************************//
// N-Body Simulation
//
// Author:  Maxeler Technologies
//
// Imperial College Summer School, July 2012
//
//*********************************************************************//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <chrono>

/**
 * 3-D coordinates
 */
typedef struct {
    float x;
    float y;
    float z;
} coord3d_t;

/** Descriptor of a particle state */
typedef struct {
    coord3d_t p;
    coord3d_t v;
} particle_t;
void process_results(particle_t * cpu_particles, int N){

  std::ofstream ofile;
  ofile.open("outputs.txt");
  if (ofile.is_open()) {
    for (int i = 0; i < N; i++) {
        if (i % (N/3) == 0) {
            ofile << "p - " << cpu_particles[i].p.x << " - " << cpu_particles[i].p.y << " - " << cpu_particles[i].p.z << std::endl;
            ofile << "v - " << cpu_particles[i].v.x << " - " << cpu_particles[i].v.y << " - " << cpu_particles[i].v.z << std::endl;
        }
        cpu_particles[i].p.x += cpu_particles[i].p.y + cpu_particles[i].p.z;
    }
    ofile.close();
  } else {
    std::cout << "Failed to create output file!" << std::endl;
  }
}

/**
 * \brief Run the N-body simulation on the CPU.
 * \param [in]  N               Number of particles
 * \param [in]  EPS             Damping factor
 * \param [in]  m               Masses of the N particles
 * \param [in]  in_particles    Initial state of the N particles
 * \param [out] out_particles   Final state of the N particles after nt time-steps
 * \param [out] time            Execution time
 */
void run_cpu(int N, float EPS, float *m,
        const particle_t *in_particles, particle_t *out_particles,
        double *time)
{
    particle_t *p;
    p = (particle_t *) malloc(N * sizeof(particle_t));
    memcpy(p, in_particles, N * sizeof(particle_t));

    coord3d_t *a;
    a = (coord3d_t *) malloc(N * sizeof(coord3d_t));

    memset(a, 0, N * sizeof(coord3d_t));
    
    for (int q = 0; q < N; q++) {
        for (int j = 0; j < N; j++) {
            float rx = p[j].p.x - p[q].p.x;
            float ry = p[j].p.y - p[q].p.y;
            float rz = p[j].p.z - p[q].p.z;
            float dd = rx*rx + ry*ry + rz*rz + EPS;
            float d = 1/ (dd*sqrt(dd));
            float s = m[j] * d;
            a[q].x += rx * s;
            a[q].y += ry * s;
            a[q].z += rz * s;
        }
    }

    for (int i = 0; i < N; i++) {
        p[i].p.x += p[i].v.x;
        p[i].p.y += p[i].v.y;
        p[i].p.z += p[i].v.z;
        p[i].v.x += a[i].x;
        p[i].v.y += a[i].y;
        p[i].v.z += a[i].z;
    }

    memcpy(out_particles, p, N * sizeof(particle_t));

    free(p);
    free(a);
}

int main(int argc, char **argv)
{

    int N = atoi(argv[1]);
    float EPS = 100;

    if (EPS == 0) {
        fprintf(stderr, "EPS cannot be set to zero\n");
        exit(EXIT_FAILURE);
    }

    particle_t *particles;
    particles = (particle_t *) malloc(N * sizeof(particle_t));
    float *m;
    m  = (float *) malloc(N * sizeof(float));

    srand(100);
    for (int i = 0; i < N; i++)
    {
        m[i] = (float)rand()/100000;
        particles[i].p.x = (float)rand()/100000;
        particles[i].p.y = (float)rand()/100000;
        particles[i].p.z = (float)rand()/100000;
        particles[i].v.x = (float)rand()/100000;
        particles[i].v.y = (float)rand()/100000;
        particles[i].v.z = (float)rand()/100000;
    }

    double cpuTime = 0;
    particle_t *cpu_particles = NULL;

    cpu_particles = (particle_t *) malloc(N * sizeof(particle_t));

    printf("Running on CPU with %d particles...\n", N);
  auto before = std::chrono::high_resolution_clock::now();
    run_cpu(N, EPS, m, particles, cpu_particles, &cpuTime);
  auto after = std::chrono::high_resolution_clock::now();
  const auto run_time = std::chrono::duration_cast<std::chrono::nanoseconds>(after - before);
  double time = run_time.count() / 1e9;
  printf("E2E time: %.6fs\n", time);

    process_results(cpu_particles, N);

    free(particles);
    free(m);
    free(cpu_particles);

    return 0;
}
