CXX = g++
CXXFLAGS = -O2 -g

EXE_NAME = nbody
SOURCES = main.cpp

$(EXE_NAME): main.cpp
	$(CXX) $(CXXFLAGS) -o $(EXE_NAME) $(SOURCES)

run: $(EXE_NAME)
	./$(EXE_NAME) 262144

clean:
	rm -f $(EXE_NAME)

