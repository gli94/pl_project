#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <fstream>
#include <cstring>
#include <algorithm>

#include "cuda_simannealing.cuh"

#define N 9
#define BLOCK_SIZE 3
#define UNASSINED 0
#define MAXB 1000
#define OFFSET 40
#define NUM_TESTCASE 100
#define MAXL N*NUM_TESTCASE

void load(char *FileName, int *board) {
    FILE * a_file = fopen(FileName, "r");
    
    if (a_file == NULL) {
        printf("File load fail!\n"); return;
    }
    
    char temp;
    
    for (int i = 0; i < MAXL; i++) {
        for (int j = 0; j < N; j++) {
            if (!fscanf(a_file, "%c\n", &temp)) {
                printf("File loading error!\n");
                return;
            }
            
            if (temp >= '1' && temp <= '9') {
                board[i * N + j] = (int) (temp - '0');
            } else {
                board[i * N + j] = 0;
            }
        }
    }
}

void printBoard(int *board) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", board[i * N + j]);
        }
        printf("\n");
    }
}

int main()
{
    char filename[] = "sudoku_9x9_100_48.txt";
    
    int *data = new int [MAXL * N];
    
    int *board = new int[N * N];
    
    double execution_time[NUM_TESTCASE];
    
    load(filename, data);
    
    int *solved = new int[N * N];
    
    for (int i = 0; i < NUM_TESTCASE; i++)
    {
        memset(solved, 0, N * N * sizeof(int));
        
        for (int ii = 0; ii < N; ii++)
        {
            for (int jj = 0; jj < N; jj++)
            {
                board[ii * N + jj] = data[i * N * N + ii * N + jj];
            }
        }
        
        //load(filename, board);
        
        printf("Board #%d\n", i);
        printBoard(board);
        
        printf("\n");
        
        cuda_SimAnnealing(board, solved, &execution_time[i]);
        
        printf("Solved Board #%d\n", i);
        printBoard(solved);
        printf("\n");
    }
    
    FILE * f = fopen("sa_stats_bench_9x9_100_48.txt", "w");
    
    if (f == NULL)
    {
        printf("Error!\n");
        exit(1);
    }
    
    double sum = 0.0;
    double average = 0.0;
    double min = 10000.0;
    double max = 0.0;
    double variance = 0.0;
    
    for (int i = 0; i < NUM_TESTCASE; i++)
    {
        sum += execution_time[i];
        fprintf(f, "%lf ", execution_time[i]);
    }
    
    average = sum / NUM_TESTCASE;
    
    sum = 0.0;
    
    for (int i = 0; i < NUM_TESTCASE; i++)
    {
        if (execution_time[i] < min)
        {
            min = execution_time[i];
        }
        
        if (execution_time[i] > max)
        {
            max = execution_time[i];
        }
        
        sum += (execution_time[i] - average)*(execution_time[i] - average);
    }
    
    variance = sum / NUM_TESTCASE;
    
    fprintf(f, "\n");
    fprintf(f, "Average: %lf\n", average);
    fprintf(f, "Variance: %lf\n", variance);
    fprintf(f, "Min: %lf\n", min);
    fprintf(f, "Max: %lf\n", max);
    
    fclose(f);
    
    return 0;
}
