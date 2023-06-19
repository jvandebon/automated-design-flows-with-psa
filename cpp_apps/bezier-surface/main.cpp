/*
 * Copyright (c) 2016 University of Cordoba and University of Illinois
 * All rights reserved.
 *
 * Developed by:    IMPACT Research Group
 *                  University of Cordoba and University of Illinois
 *                  http://impact.crhc.illinois.edu/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the 
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *      > Redistributions of source code must retain the above copyright notice,
 *        this list of conditions and the following disclaimers.
 *      > Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimers in the
 *        documentation and/or other materials provided with the distribution.
 *      > Neither the names of IMPACT Research Group, University of Cordoba, 
 *        University of Illinois nor the names of its contributors may be used 
 *        to endorse or promote products derived from this Software without 
 *        specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 */

#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <chrono>
#include <iostream>
#include <fstream>

typedef struct {
    float x;
    float y;
    float z;
} XYZ;

#define divceil(n, m) (((n)-1) / (m) + 1)
#define in_size 20 

void process_results(int out_size, XYZ* out){
  std::ofstream ofile;
  ofile.open("outputs.txt");
  if (ofile.is_open()) {
    for (int i = 0; i < out_size*out_size; i++) {
        if (i % ((out_size*out_size)/10) == 0) {
            ofile << i << " -- " << out[i].x << " " << out[i].y << " " <<  out[i].z << std::endl;
        } 
        out[i].x += out[i].y + out[i].z;
    }
    ofile.close();
  } else {
    std::cout << "Failed to create output file!" << std::endl;
  }
}

// Input Data -----------------------------------------------------------------
void read_input(XYZ *in) {

    const char *file_name     = "input/control.txt";

    // Open input file
    FILE *f = NULL;
    f       = fopen(file_name, "r");
    if(f == NULL) {
        puts("Error opening file");
        exit(-1);
    } else {
        printf("Read data from file %s.\n", file_name);
    } 

    // Store points from input file to array
    int k = 0;
    int ic = 0;
    XYZ v[10000];
    while(fscanf(f, "%f,%f,%f", &v[ic].x, &v[ic].y, &v[ic].z) == 3)
    {
        ic++;
    }
    for(int i = 0; i <= in_size; i++) {
        for(int j = 0; j <= in_size; j++) {
            in[i * (in_size + 1) + j].x = v[k].x;
            in[i * (in_size + 1) + j].y = v[k].y;
            in[i * (in_size + 1) + j].z = v[k].z;
            k = (k + 1) % 16;
        }
    }
}

// BezierBlend (http://paulbourke.net/geometry/beziefloat BezierBlend(int k, float mu, int n) {
inline float BezierBlend(int k, float mu, int n)
{
  int kn, nkn;
  float blend;
  blend = 1;

  kn = k;
  nkn = n - k;
  for (int nn = n; nn >= 1; nn--)
  {
    blend *= nn;
    if (kn > 1)
    {
      blend /= (float)kn;
      kn--;
    }
    if (nkn > 1)
    {
      blend /= (float)nkn;
      nkn--;
    }
  }
  if (k > 0)
  {
    blend *= pow(mu, (float)k);
  }
  if (n - k > 0)
  {
    blend *= pow(1 - mu, (float)(n - k));
  }
  return (blend);
}



void run(XYZ *in, XYZ *out, int out_size) {

    for(int i = 0; i < out_size; i++) {
        float mui = i / (float)(out_size - 1);
        for(int j = 0; j < out_size; j++) {
            float muj = j / (float)(out_size - 1);
            XYZ _out = {0, 0, 0};
            for(int ki = 0; ki < in_size+1; ki++) {
                float bi = BezierBlend(ki, mui, in_size);
                for(int kj = 0; kj < in_size+1; kj++) {
                    float bj = BezierBlend(kj, muj, in_size);
                    _out.x += (in[ki * (in_size + 1) + kj].x * bi * bj);
                    _out.y += (in[ki * (in_size + 1) + kj].y * bi * bj);
                    _out.z += (in[ki * (in_size + 1) + kj].z * bi * bj);
                }
            }
            out[i * out_size + j].x = _out.x;
            out[i * out_size + j].y = _out.y;
            out[i * out_size + j].z = _out.z;
        }
    }
}

int main(int argc, char **argv) {

    int out_size = 2000; // output resolution in both dimensions (default=300)

    if (argc > 1){
        out_size = atoi(argv[1]);
    }

    printf("Size in: %d, size out: %d\n", in_size, out_size);

    XYZ* h_in, *out;
    h_in = (XYZ *)malloc((in_size+1)*(in_size+1)*sizeof(XYZ));
    out = (XYZ *)malloc(out_size * out_size * sizeof(XYZ));
    
    const auto before_ = std::chrono::high_resolution_clock::now();
    read_input(h_in);
    const auto after_ = std::chrono::high_resolution_clock::now();
    const auto run_time_ = std::chrono::duration_cast<std::chrono::nanoseconds>(after_ - before_);
    double time_ = run_time_.count()/1e9;  
    printf("Input reading time: %.3gs\n", time_);

    // run the app 
    printf("Running blend...\n");
    const auto before = std::chrono::high_resolution_clock::now();
    run(h_in, out, out_size);
    const auto after = std::chrono::high_resolution_clock::now();
    const auto run_time = std::chrono::duration_cast<std::chrono::nanoseconds>(after - before);
    double time = run_time.count()/1e9;  
    printf("Compute time: %.5fs\n", time);
    
    process_results(out_size, out);

    free(h_in);
    free(out);
    return 0;
}


