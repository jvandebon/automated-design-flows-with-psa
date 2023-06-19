#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>


using namespace std;

#define NUM_FEATURES 12

float frand(int min, int max) {
  return min + (float) rand() / ((float) RAND_MAX / (max - min));
}

float rand_y() {
  return (rand() % 2) ? 1.0 : -1.0;
}
float rand_m() {
  return frand(-6, 6);
}
float rand_v() {
  return frand(0, 100);
}


int main(int argc, char ** argv)
{
    // 8kiB = 8192 bytes = 2048 floats, volume = 170
    if (argc == 1) {
       cerr << "syntax: " << argv[0] << " <num impressions> " << endl;
       return -1;
    }

    long unsigned int volume = atoi(argv[1]);
    printf("Generating input files in with %lu impressions...\n", volume);


    ofstream db;
    db.open("adp.db");

    // first line:  volume size
    db << volume << endl;

    /* generate data:
       each line:
       y0
       m0
       v0
       m1
       v1
       ...
       y1...
    */
    for (int i = 0; i < volume; i++) {
      string tmp = "";
      // db << rand_y() << endl;
      tmp += std::to_string(rand_y()) + "\n";
      for (int j = 0; j < NUM_FEATURES; j++) {
        // db << rand_m() << endl;
        // auto m = rand_m();
        // db << rand_v() << endl;
        tmp += to_string(rand_m()) + "\n";
        tmp += to_string(rand_v()) + "\n";
      }
      db << tmp;
    }

    db.close();

    return 0;
}