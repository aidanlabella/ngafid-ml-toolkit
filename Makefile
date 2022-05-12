CC=g++
CFLAGS=-I -Wall -Wextra -pedantic -ggdb -ltorch -ltorch_cpu
#DEPS = hellomake.h

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

main: main.o 
	$(CC) -o main main.o
