MAIN:=run_test
OPTS:=-D __BENCHMARK -D __INFO -D __ASSERT -fopenmp

all:
	g++ $(OPTS) -o $(MAIN) $(MAIN).cpp -I../include
clean:
	rm -rf $(MAIN) *~ *.dat
