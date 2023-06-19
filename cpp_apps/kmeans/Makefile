CXX = clang++
CXXFLAGS = -O2 -g -std=c++17

EXE_NAME = kmeans
SOURCES = main.cpp
ARGS=262144 

run: $(EXE_NAME)
	./$(EXE_NAME) $(ARGS)


$(EXE_NAME): main.cpp
	$(CXX) $(CXXFLAGS) -o $(EXE_NAME) $(SOURCES)

clean:
	rm -f $(EXE_NAME)

