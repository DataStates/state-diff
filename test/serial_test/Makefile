COMPILER:=g++
MAINB:=test_boost
MAINC:=test_cereal
OPTS:=-D__BENCHMARK -D__INFO -D__ASSERT -D__USEKOKKOS -fopenmp -std=c++17 #--extended-lambda
INCLUDES:=-I${HOME}/research/anl/install/kokkos/install/include
LIBS:=-L${HOME}/research/anl/install/kokkos/install/lib -lkokkoscore
BOOST:=-I${HOME}/research/anl/install/boost_1_82_0/include -L${HOME}/research/anl/install/boost_1_82_0/lib -lboost_serialization
CEREAL:=-I${HOME}/research/anl/install/cereal/include

boost:
	$(COMPILER) $(OPTS) -o $(MAINB) $(MAINB).cpp $(INCLUDES) $(LIBS) $(BOOST)
cereal:
	$(COMPILER) $(OPTS) -o $(MAINC) $(MAINC).cpp $(INCLUDES) $(LIBS) $(CEREAL)
clean:
	rm -rf $(MAINB) $(MAINC) *.dat *.boost *.cereal *~
