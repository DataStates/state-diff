COMPILER:=g++
MAIN:=test_stream
OPTS:=-std=c++17 #--extended-lambda
INCLUDES:=-I${HOME}/research/anl/install/kokkos/installgpu/include
LIBS:=-L${HOME}/research/anl/install/kokkos/installgpu/lib -lkokkoscore
all:
	$(COMPILER) $(OPTS) -o $(MAIN) $(MAIN).cpp $(INCLUDES) $(LIBS)
clean:
	rm -rf $(MAIN)
