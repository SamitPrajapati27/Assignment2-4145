CC = gcc
CFLAGS = -O2 -Wall -I.
LDFLAGS_A = -pthread
LDFLAGS_B = -fopenmp

all: partA partB

partA: src/partA.c assignment2.h
	$(CC) $(CFLAGS) $(LDFLAGS_A) src/partA.c -o partA

partB: src/partB.c assignment2.h
	$(CC) $(CFLAGS) $(LDFLAGS_B) src/partB.c -o partB

clean:
	rm -f partA partB