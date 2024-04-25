COMPILER:=g++
MAIN:=run_test
OPTS:=-D__BENCHMARK -D__INFO -D__ASSERT -D__USEKOKKOS -fopenmp -std=c++17 #--extended-lambda
INCLUDES:=-I../include -I${HOME}/kokkos-cpu/build/install/include
LIBS:=-L${HOME}/kokkos-cpu/build/install/lib64 -lkokkoscore -ldl

all:
	$(COMPILER) $(OPTS) -o $(MAIN) $(MAIN).cpp $(INCLUDES) $(LIBS)       
clean:
	rm -rf $(MAIN) *~
