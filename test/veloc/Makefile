COMPILER:=g++
MAIN:=run_test
OPTS:=-D__BENCHMARK -D__INFO -D__ASSERT -D__USEKOKKOS -fopenmp -std=c++17
KOKKOS_DIR:=${HOME}/research/anl/install/kokkos/installgpu
STATEDIFF_DIR:=${HOME}/research/anl/state-diff/build
VELOC_DIR:=${HOME}/research/anl/install/veloc/statediffgpu
INCLUDES:=-I$(VELOC_DIR)/include -I$(STATEDIFF_DIR)/include -I$(STATEDIFF_DIR)/liburing/install/include -I/usr/lib/x86_64-linux-gnu/openmpi/include
LIBS:=-L$(VELOC_DIR)/lib -L/usr/lib/x86_64-linux-gnu/openmpi/lib -L$(STATEDIFF_DIR)/lib -L$(STATEDIFF_DIR)/liburing/install/lib -luring -lstatediff -lmpi_cxx -lmpi -lveloc-client
LD_PATH:=-Wl,-rpath,$(VELOC_DIR)/lib:$(STATEDIFF_DIR)/lib

all:
	$(COMPILER) $(OPTS) -o $(MAIN) $(MAIN).cpp $(INCLUDES) $(LIBS) $(LD_PATH)
clean:
	rm -rf $(MAIN) *~ *.dat