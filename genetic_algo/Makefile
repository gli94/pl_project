
CC = gcc
CFLAGS = -lm -fopenmp

m: GA_mc.c
	$(CC) $(CFLAGS) -o GA_mc GA_mc.c

s: GA_sc.c
	$(CC) $(CFLAGS) -o GA_sc GA_sc.c

clean:
	rm GA_mc GA_sc