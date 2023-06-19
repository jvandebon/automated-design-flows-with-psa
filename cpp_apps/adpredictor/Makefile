CXX = g++
CXXFLAGS = -O2 -g -std=c++17 -lm

EXE_NAME = adp
SOURCES = main.cpp

run: $(EXE_NAME)
	./$(EXE_NAME) ./inputs_small 16384


$(EXE_NAME): main.cpp
	$(CXX) $(CXXFLAGS) -o $(EXE_NAME) $(SOURCES)

clean:
	rm -f $(EXE_NAME)
