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

void load(char *FileName, int *board) {
    FILE * a_file = fopen(FileName, "r");
    
    if (a_file == NULL) {
        printf("File load fail!\n"); return;
    }
    
    char temp;
    
    for (int i = 0; i < N; i++) {
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
    int blocksPerGrid = 1024;
    int threadsPerBlock = 256;
    
    char filename[] = "sudoku_board_hard.txt";
    
    int *board = new int[N * N];
    
    load(filename, board);
    
    int *solved = new int[N * N];
    memset(solved, 0, N * N * sizeof(int));
    
    cuda_SimAnnealing(board, solved);
    
    printBoard(solved);
    
    return 0;
}
