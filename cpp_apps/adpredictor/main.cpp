#include <stdlib.h>
#include <cstdio>
#include <iostream>
#include <unistd.h>
#include <sys/time.h>
#include <cmath>
#include <stdint.h>
#include <fstream>
#include <cstring> 

#include <chrono>

using namespace std;

#define NUM_FEATURES 12
#define VOL_INC 50000

#define root2pi 2.50662827463100050242
#define p0 220.2068679123761
#define p1 221.2135961699311
#define p2 112.0792914978709
#define p3 33.91286607838300
#define p4 6.373962203531650
#define p5 0.7003830644436881
#define p6 0.3526249659989109E-01
#define q0 440.4137358247522
#define q1 793.8265125199484
#define q2 637.3336333788311
#define q3 296.5642487796737
#define q4 86.78073220294608
#define q5 16.06417757920695
#define q6 1.755667163182642
#define q7 0.8838834764831844E-1
#define cutoff 7.071
#define beta 0.001

void check_results(uint64_t volume, float *post_m, float *post_s)
{
  std::ofstream ofile;
  ofile.open("outputs.txt");
  if (ofile.is_open())
  {
    ofile << "\nst | i | post_m | post_s \n";
    for (uint64_t st = 0; st < volume; st++) {
      for (int i = 0; i < NUM_FEATURES; i++) {
        uint64_t idx = st * NUM_FEATURES + i;
        if (st % (volume / 10) == 0 && i == 0) {
          ofile << st << " " << i << " " << post_m[idx] << " " << post_s[idx] << std::endl;
        }
        post_m[idx] += post_s[idx];
        if (st % (volume / 5) == 0 && i == 0) {
          ofile << "TEST " << post_m[idx] << " " << post_s[idx] << std::endl;
        }
      }
    }
  }
  else
  {
    std::cout << "Failed to create output file!" << std::endl;
  }
}

float PDF(float z) {
   return exp(-z * z / 2) / root2pi;
}
float CDF(float z) {
        float zabs = fabs(z);
        float expntl = exp(-.5*zabs*zabs);
        float pdf = expntl/root2pi;

        int c1 = z > 37.0;
        int c2 = z < -37.0;
        int c3 = zabs < cutoff;
        float pA =  expntl*((((((p6*zabs + p5)*zabs + p4)*zabs + p3)*zabs +
          p2)*zabs + p1)*zabs + p0)/(((((((q7*zabs + q6)*zabs +
          q5)*zabs + q4)*zabs + q3)*zabs + q2)*zabs + q1)*zabs +
          q0);
   float pB = pdf/(zabs + 1.0/(zabs + 2.0/(zabs + 3.0/(zabs + 4.0/
          (zabs + 0.65)))));

   float pX = c3? pA : pB;
   float p = (z < 0.0) ? pX : 1 - pX;
   return c1? 1.0 : (c2 ? 0.0 : p);
}

float V (float t) {
   float cdf = CDF(t);
   int c0 = (cdf == 0.0);
   return c0? 0.0 : (PDF(t) / cdf);
}

float W (float t) {
   float v = V(t);
   return v * (v + t);
}

int main(int argc, char *argv[]) {

  if (argc < 3) {
    cerr << "syntax: " << argv[0] << " <db folder/> volume" << std::endl;
    return -1;
  }

  string db_folder = argv[1];
  uint64_t volume = atoi(argv[2]);

  ifstream db;
  db.open(db_folder + "/adp.db");

  if (!db.is_open()) {
     cerr << "Cannot open file: " << db_folder + "/adp.db" << "!" << std::endl;
     return -1;
  }

  uint64_t num_inputs;
  db >> num_inputs;

  int num_chunks = volume / num_inputs;
  float *y_in = new float[num_inputs];
  float *prior_m_in = new float[num_inputs * 12];
  float *prior_v_in = new float[num_inputs * 12];

  const auto before0 = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < num_inputs; i++) {
    db >> y_in[i];
    for (int j = 0; j < 12; j++) {
      uint64_t idx = i * 12 + j;
      db >> prior_m_in[idx];
      db >> prior_v_in[idx];
    }
  }
  const auto after0 = std::chrono::high_resolution_clock::now();
  const auto run_time0 = std::chrono::duration_cast<std::chrono::nanoseconds>(after0 - before0);
  double time0 = run_time0.count() / 1e9;
  printf("Input reading time: %.3gs\n", time0);

  db.close();


  std::cout << "Number of ad impressions: " << volume << "\n";
// double total = 0;
// for (int R = 0; R < 5; R++){
  float *y = new float[volume];
  float *prior_m = new float[volume * NUM_FEATURES];
  float *prior_v = new float[volume * NUM_FEATURES];
  float *post_m = new float[volume * NUM_FEATURES];
  float *post_s = new float[volume * NUM_FEATURES];

  for (int i = 0; i < num_chunks + 1; i++) {
    int num_to_copy = (i*num_inputs) + num_inputs >= volume ? (volume-(i*num_inputs)) : num_inputs;
    std::memcpy(y+(i*num_inputs), y_in, num_to_copy * sizeof(float));
    std::memcpy(prior_m+(i*12*num_inputs), prior_m_in, num_to_copy*12*sizeof(float));
    std::memcpy(prior_v+(i*12*num_inputs), prior_v_in, num_to_copy*12*sizeof(float));
  }

  const auto before = std::chrono::high_resolution_clock::now();

  for (uint64_t st = 0; st < volume; st++) {
    float s = 0.0;
    float m = 0.0;
    for (int i = 0; i < NUM_FEATURES; i++) {
        m += prior_m[st * NUM_FEATURES + i];
        s += prior_v[st * NUM_FEATURES + i];
    }
    float S = sqrt(beta * beta + s);
    float t = (y[st] * m) / S;
    for (int j = 0; j < NUM_FEATURES; j++) {
        post_m[st * NUM_FEATURES + j] = prior_m[st * NUM_FEATURES + j] + y[st] * (prior_v[st * NUM_FEATURES + j] / S) *  V(t);
        post_s[st * NUM_FEATURES + j] = sqrt(fabs(prior_v[st * NUM_FEATURES + j] * (1-(prior_v[st * NUM_FEATURES + j] / (S*S)) *  W(t))));
    }
  }

  const auto after = std::chrono::high_resolution_clock::now();
  const auto run_time = std::chrono::duration_cast<std::chrono::nanoseconds>(after - before);
  double time = run_time.count()/1e9;  
  printf("CPU execution time: %.5gs\n", time);
  // total += time;

  check_results(volume, post_m, post_s);

  delete[] prior_m;
  delete[] prior_v;
  delete[] post_m;
  delete[] post_s;
  delete[] y;
// }
// printf("Ave time: %.5gs\n", total/5.0);
  return 0;
}

