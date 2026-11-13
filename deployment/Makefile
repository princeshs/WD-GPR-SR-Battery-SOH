# Makefile for compiling WD-GPR-SR symbolic equation shared library
CC = gcc
CFLAGS = -O3 -fPIC -Wall
LDFLAGS = -shared -lm

all: libequation.so

libequation.so: equation.c
	$(CC) $(CFLAGS) $(LDFLAGS) equation.c -o libequation.so

clean:
	rm -f libequation.so
