CXX = g++
CXXFLAGS = -O2 -g

EXE_NAME = blend
SOURCES = main.cpp

$(EXE_NAME): main.cpp
	$(CXX) $(CXXFLAGS) -o $(EXE_NAME) $(SOURCES)

run: $(EXE_NAME)
	./$(EXE_NAME)
	
clean:
	rm -f $(EXE_NAME)

