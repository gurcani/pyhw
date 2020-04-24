all: fhwak

fhwak: fhwak.o
	gcc -shared -o libfhwak.so fhwak.o -lm -lfftw3 -lfftw3_threads -fopenmp
fhwak.o: fhwak.c fhwak.h
	gcc -c -fpic fhwak.c -fopenmp
clean:
	rm -f test
	rm -f *.so *.o
