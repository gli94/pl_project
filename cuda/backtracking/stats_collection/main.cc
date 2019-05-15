#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <fstream>
#include <cstring>
#include <algorithm>

#include "cuda_backtracking.cuh"

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
    
    char filename[] = "sudoku_9x9_100_56.txt";
    
    int *data = new int [MAXL * N];
    
    int *board = new int[N * N];
    
    int num_solved = 0;
    int finished;
    
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
        
    cuda_Backtrack(board, solved, &execution_time[i], &finished);
        
        finished = 1;
        for (int ii = 0; ii < N; ii++)
        {
            for (int jj = 0; jj < N; jj++)
            {
                if (solved[ii*N + jj] == 0)
                {
                    finished = 0;
                    break;
                }
            }
        }
        
        num_solved+=finished;
    
         printf("Solved Board #%d\n", i);
    printBoard(solved);
        printf("\n");
    }
    
    printf("Number of solved board: %d\n", num_solved);
    
    FILE * f = fopen("bt_stats_bench_9x9_100_56.txt", "w");
    
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
    
    /*int *old_boards;
    int *new_boards;
    int *empty_spaces;
    int *empty_space_count;
    int *board_index;
    
    int sk = pow(2, 26);
    
    cudaMalloc(&empty_spaces, sk * sizeof(int));
    cudaMalloc(&empty_space_count, ((sk / 81) + 1) * sizeof(int));
    cudaMalloc(&old_boards, sk * sizeof(int));
    cudaMalloc(&new_boards, sk * sizeof(int));
    cudaMalloc(&board_index, sizeof(int));
    
    int total_boards = 1;
    
    cudaMemset(board_index, 0, sizeof(int));
    cudaMemset(new_boards, 0, sk * sizeof(int));
    cudaMemset(old_boards, 0, sk * sizeof(int));
    
    cudaMemcpy(old_boards, board, N * N * sizeof(int), cudeMemcpyHostToDevice);
    
    callBFSKernel(blocksPerGrid, threadsPerBlock, old_boards, new_boards, total_boards, board_index, empty_spaces, empty_space_count);
    
    int host_count;
    
    int iterations = 18;
    
    for (int i=0; i<iterations; i++)
    {
        cudaMemcpy(&host_count, board_index, sizeof(int), cudeMemcpyDeviceToHost);
        printf("total boards after an iteration %d: %d\n", i, host_count);
        cudaMemset(board_index, 0, sizeof(int));
        
        if((i % 2) == 0)
        {
            callBFSKernel(blocksPerGrid, threadsPerBlock, new_boards, old_boards, host_count, board_index, empty_spaces, empty_space_count);
        }
        else
        {
            callBFSKernel(blocksPerGrid, threadsPerBlock, old_boards, new_boards, host_count, board_index, empty_spaces, empty_space_count);
        }
    }
    
    cudaMemcpy(&host_count, board_index, sizeof(int), cudaMemcpyDeviceToHost);
    printf("new number of boards retrieved is %d\n", host_count);
    
    int *dev_finished;
    int *dev_solved;
    
    cudaMalloc(&dev_finished, sizeof(int));
    cudaMalloc(&dev_solved, N * N * sizeof(int));
    
    cudaMemset(dev_finished, 0, sizeof(int));
    cudaMemcpy(dev_solved, board, N * N * sizeof(int), cudaMemcpyHostToDevice);
    
    if((iterations % 2) == 1)
    {
        new_boards = old_boards;
    }
    
    cuda_sudokuBacktrack(blocksPerGrid, threadsPerBlock, new_boards, host_count, empty_spaces, empty_space_count, dev_finished, dev_solved);*/
    
    //int *solved = new int[N * N];
    //memset(solved, 0, N * N * sizeof(int));
    /*cudaMemcpy(solved, dev_solved, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    printBoard(solved);

    delete[] board;
    delete[] solved;
    
    cudaFree(empty_spaces);
    cudaFree(empty_space_count);
    cudaFree(new_boards);
    cudaFree(old_boards);
    cudaFree(board_index);
    
    cudaFree(dev_finished);
    cudaFree(dev_solved);*/
    
    return 0;
}
