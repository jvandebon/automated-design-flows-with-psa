#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#define DIM 16
#define CLASSES	1024

int main(int argc, char **argv) {

	int points = atoi(argv[1]);
	srand(100);
	// double total_time__ = 0;
	// for (int run = 0; run < 5; run++) {

	float *classes = (float *)malloc(DIM*CLASSES*sizeof(float));
	float *data = (float *)malloc(points*DIM*sizeof(float));
	float *radii2 = (float *)malloc(CLASSES*sizeof(float));
	int *out = (int *)malloc(points*CLASSES*sizeof(int));

	for(int i = 0; i < CLASSES*DIM; i++) 
		classes[i] = (float) rand();
	
	for(int i = 0; i < DIM*points; i++) 
		data[i] = (float) rand();

	for(int i = 0; i < CLASSES; i++) {
		radii2[i] = (float) rand();
		radii2[i] = radii2[i] * radii2[i];
	}

	for(int i = 0 ; i < CLASSES*points; i++) 
		out[i] = 0;

	auto before = std::chrono::high_resolution_clock::now();
	for(int i = 0; i < points; i++) {
		for(int j = 0; j < CLASSES; j++) {
			float dist2 = 0.0;
			for(int k = 0 ; k < DIM; k++) {
				dist2 += (classes[k+DIM*j]-data[DIM*i+k]) * (classes[k+DIM*j]-data[DIM*i+k]);
			}
			out[CLASSES*i+j] = (dist2 < radii2[j])? j : -1;
		}
	}
    auto after = std::chrono::high_resolution_clock::now();
    auto run_time = std::chrono::duration_cast<std::chrono::nanoseconds>(after - before);
    double time = run_time.count() / 1e9;
    printf("Execution time: %.6gs\n", time);
	// total_time__+=time;

	/* PROCESS RESULTS */
	int cnt = 0;
	for (int i = 0; i < points; i ++) {
		int classified = 0;
		for (int j = 0; j < CLASSES; j++) {
			if (out[i*CLASSES+j] != -1) {
				classified++;
			}
		}
		if (classified) { cnt++; }
	}
	printf("%d/%d points classified in at least one cluster\n", cnt, points);

	free(classes);
	free(data);
	free(radii2);
	free(out);
	// }

	// printf("AVERAGE: %.6gs\n", total_time__/5.0);

	return 0;
}

