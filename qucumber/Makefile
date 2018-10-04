CXX	= g++	

ACCELERATE_LAPACK=-framework Accelerate

#Optimized running flags
#CXXFLAGS	= -O2 -DNDEBUG $(ACCELERATE_LAPACK) -DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE -I /usr/local/include/eigen3/  -std=c++11


#Debug-mode flags
CXXFLAGS =     -O2 -I /usr/local/include/eigen3/ -std=c++11



ma :
	$(CXX) main.cpp $(CXXFLAGS) $(LFLAGS) -o run.x


clean	:	cleano cleant cleanout cleanlog

cleano	:
	rm -f *.o

cleant	:
	rm -f *.*~

cleanout	:
	rm -f *.out

cleanlog	:
	rm -f *.log
