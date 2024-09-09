COMPILER:=g++
MAIN:=serial_test
OPTS:=-D__BENCHMARK -D__INFO -D__ASSERT -D__USEKOKKOS -fopenmp -std=c++17 #--extended-lambda
INCLUDES:=-I${HOME}/research/anl/install/kokkos/install/include -I${HOME}/research/anl/install/boost_1_82_0/include
LIBS:=-L${HOME}/research/anl/install/kokkos/install/lib -L${HOME}/research/anl/install/boost_1_82_0/lib -lkokkoscore -lboost_serialization -ldl

all:
	$(COMPILER) $(OPTS) -o $(MAIN) $(MAIN).cpp $(INCLUDES) $(LIBS)       
clean:
	rm -rf $(MAIN) *~
