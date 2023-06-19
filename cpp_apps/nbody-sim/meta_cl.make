CXX =clang++ #hipcc #dpcpp  
CXXFLAGS = -O2 -g -lm $(META_CL_CXXFLAGS)
LDFLAGS = $(META_CL_LDFLAGS)

EXE_NAME = nbody-sim
SOURCES = main.cpp
ARGS=32768
ORIG_ARGS=131072
SMALL_ARGS=4

run: $(EXE_NAME)
	./$(EXE_NAME) $(ARGS)

small: $(EXE_NAME)
	./$(EXE_NAME) $(SMALL_ARGS)

orig: $(EXE_NAME)
	./$(EXE_NAME) $(ORIG_ARGS)
	
$(EXE_NAME): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(EXE_NAME) $(SOURCES)

gpu: $(SOURCES)
	hipcc $(CXXFLAGS) -std=c++14 $(LDFLAGS) --ptxas-options=-v -o $(EXE_NAME)_gpu $(SOURCES)

run_gpu: $(EXE_NAME)_gpu
	./$(EXE_NAME)_gpu $(ORIG_ARGS)

run_omp: $(EXE_NAME)_omp
	./$(EXE_NAME)_omp $(ARGS)
	
$(EXE_NAME)_omp: $(SOURCES)
	g++ $(CXXFLAGS)  -fopenmp -o $(EXE_NAME)_omp $(SOURCES) $(LDFLAGS)

# FPGA REPORT
a10_report: a10_report.a
dev_a10.o: $(SOURCES)
	dpcpp $(CXXFLAGS)  -fintelfpga -c $^ -o $@ -DFPGA=1
a10_report.a: dev_a10.o
	dpcpp  $(CXXFLAGS)  -fintelfpga -fsycl-link $^ -o $@ -Xshardware

s10_report: s10_report.a
dev_s10.o: $(SOURCES)
	dpcpp  $(CXXFLAGS)  -fintelfpga -c $^ -o $@ -DFPGA=1
s10_report.a: dev_s10.o
	dpcpp  $(CXXFLAGS)  -fintelfpga -fsycl-link $^ -o $@ -Xshardware -Xsboard=/opt/intel/oneapi/intel_s10sx_pac:pac_s10_usm

clean:
	rm -f $(EXE_NAME) outputs.txt

