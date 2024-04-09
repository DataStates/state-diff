COMPILER:=<path_to_nvcc_wrapper>
MAIN:=run_test
OPTS:=-D__BENCHMARK -D__INFO -D__ASSERT -D__USEKOKKOS -fopenmp --extended-lambda
INCLUDES:=-I../include -I<path_to_kokkos_install_dir>/include
LIBS:=-L<path_to_kokkos_install_dir>/lib -lkokkoscore

all:
	$(COMPILER) $(OPTS) -o $(MAIN) $(MAIN).cpp $(INCLUDES) $(LIBS)       
clean:
	rm -rf $(MAIN) *~