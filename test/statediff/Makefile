MAIN:=run_test
OPTS:=-D__BENCHMARK -D__DEBUG -D__ASSERT -fopenmp -std=c++17

COMPILER:=g++
KOKKOS_INSTALL:=${HOME}/research/anl/install/kokkos/installcpu
BUILD_DIR:=${HOME}/research/anl/state-diff/buildcpu

# COMPILER:=${HOME}/research/anl/install/kokkos/kokkos/bin/nvcc_wrapper --extended-lambda -lcuda
# KOKKOS_INSTALL:=${HOME}/research/anl/install/kokkos/installgpu
# BUILD_DIR:=${HOME}/research/anl/state-diff/build

INCLUDES:=-I$(BUILD_DIR)/include -I$(BUILD_DIR)/liburing/install/include -I$(KOKKOS_INSTALL)/include
LIBS:=-L$(BUILD_DIR)/lib -L$(KOKKOS_INSTALL)/lib -L$(BUILD_DIR)/liburing/install/lib -luring -lkokkoscore -lstatediff
LD_PATH:=-Wl,-rpath,$(BUILD_DIR)/lib

all:
	$(COMPILER) $(OPTS) -o $(MAIN) $(MAIN).cpp $(INCLUDES) $(LIBS) $(LD_PATH)
clean:
	rm -rf $(MAIN) *~ *.dat