#include<stdio.h>
#include<stdlib.h>
#include<stdbool.h>
#include<time.h>
#include<math.h>

#define N 9
#define BLOCK_SIZE 3
#define UNASSINED 0
#define MAXL 900
#define MAXB 1000
#define NUM_TESTCASE 100

bool checkcomplete(int *grid, int Num, int *row, int *col)
{
    for (int i = 0; i < Num; i++)
    {
        for (int j = 0; j < Num; j++)
        {
            if (grid[i * Num + j] == UNASSINED)
            {
                *row = i;
                *col = j;
                return false;
            }
        }
    }
    
    return true;
}

bool checkrow(int *grid, int Num, int row, int value)
{
    for (int col = 0; col < Num; col++)
    {
        if (grid[row * Num + col] == value)
        {
            return false;
        }
    }
    
    return true;
}

bool checkcol(int *grid, int Num, int col, int value)
{
    for (int row = 0; row < Num; row++)
    {
        if (grid[row * Num + col] == value)
        {
            return false;
        }
    }
    
    return true;
}

bool checkbox(int *grid, int Num, int box_start_row, int box_start_col, int value)
{
    for (int row = box_start_row; row < box_start_row + BLOCK_SIZE; row++)
    {
        for (int col = box_start_col; col < box_start_col + BLOCK_SIZE; col++)
        {
            if (grid[row * Num + col] == value)
            {
                return false;
            }
        }
    }
    
    return true;
}

bool isvalid(int *grid, int Num, int row, int col, int value)
{
    if (checkrow(grid, Num, row, value) && checkcol(grid, Num, col, value) && checkbox(grid, Num, row - row % BLOCK_SIZE, col - col % BLOCK_SIZE, value) && (grid[row * Num + col] == UNASSINED))
    {
        return true;
    }
    else
    {
        return false;
    }
}

void printgrid(int * grid)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%d ", grid[i * N + j]);
        }
        printf("\n");
    }
}

bool sudoku_solver(int * grid)
{
    int row, col;
    
    if(checkcomplete(grid, N, &row, &col))
    {
        return true;
    }
    
    for (int value = 1; value < N + 1; value++)
    {
        if (isvalid(grid, N, row, col, value))
        {
            grid[row * N + col] = value;
            
            if (sudoku_solver(grid))
            {
                return true;
            }
            
            grid[row * N + col] = UNASSINED;
        }
    }
    
    return false;
}

int main()
{
    
    int grid[N * N] = {3, 0, 6, 5, 0, 8, 4, 0, 0,
        5, 2, 0, 0, 0, 0, 0, 0, 0,
        0, 8, 7, 0, 0, 0, 0, 3, 1,
        0, 0, 3, 0, 1, 0, 0, 8, 0,
        9, 0, 0, 8, 6, 3, 0, 0, 5,
        0, 5, 0, 0, 9, 0, 6, 0, 0,
        1, 3, 0, 0, 0, 0, 2, 5, 0,
        0, 0, 0, 0, 0, 0, 0, 7, 4,
        0, 0, 5, 2, 0, 6, 3, 0, 0};
        
        clock_t begin = clock();
        
        if (sudoku_solver(grid))
        {
            printgrid(grid);
        }
        else
        {
            printf("No valid solution for the game!\n");
        }
        
        clock_t end = clock();
        double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    
        
        printf("Execution time: %lfs\n", time_spent);
    
    
    
    /*int grid[N][N] = {{3, 0, 6, 5, 0, 8, 4, 0, 0},
     {5, 2, 0, 0, 0, 0, 0, 0, 0},
     {0, 8, 7, 0, 0, 0, 0, 3, 1},
     {0, 0, 3, 0, 1, 0, 0, 8, 0},
     {9, 0, 0, 8, 6, 3, 0, 0, 5},
     {0, 5, 0, 0, 9, 0, 6, 0, 0},
     {1, 3, 0, 0, 0, 0, 2, 5, 0},
     {0, 0, 0, 0, 0, 0, 0, 7, 4},
     {0, 0, 5, 2, 0, 6, 3, 0, 0}};*/
    
    /*int grid[N][N] = {{0, 0, 6, 0, 7, 0, 0, 0, 15, 0, 0, 0, 0, 0, 5, 9},
     {0, 0, 3, 0, 0, 6, 0, 14, 0, 0, 0, 1, 0, 2, 0, 0},
     {0, 13, 0, 14, 10, 0, 0, 15, 0, 0, 6, 11, 0, 0, 0, 1},
     {5, 2, 0, 15, 0, 12, 16, 0, 0, 9, 4, 0, 0, 0, 0, 7},
     {0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 13, 0, 14},
     {14, 0, 0, 6, 11, 10, 13, 1, 0, 0, 5, 3, 0, 8, 0, 0},
     {0, 0, 0, 12, 0, 0, 7, 0, 0, 0, 0, 13, 0, 0, 0, 0},
     {2, 0, 0, 0, 0, 0, 0, 3, 4, 1, 10, 0, 15, 0, 7, 0},
     {8, 0, 0, 0, 0, 0, 4, 0, 0, 6, 13, 9, 7, 0, 0, 0},
     {0, 0, 0, 2, 12, 0, 0, 0, 16, 0, 0, 8, 13, 0, 0, 0},
     {1, 14, 0, 0, 2, 0, 0, 10, 0, 3, 0, 15, 0, 6, 0, 0},
     {0, 4, 0, 3, 0, 14, 11, 6, 0, 0, 12, 0, 0, 0, 10, 0},
     {0, 9, 0, 0, 0, 1, 2, 7, 0, 11, 8, 12, 0, 0, 0, 16},
     {0, 5, 0, 0, 0, 0, 3, 11, 9, 0, 0, 10, 0, 7, 0, 0},
     {0, 12, 13, 0, 0, 0, 0, 0, 5, 15, 0, 0, 0, 9, 8, 0},
     {0, 16, 11, 10, 9, 0, 0, 0, 0, 0, 7, 0, 5, 0, 0, 0}};*/
    
    /*int grid[N][N] = {{0, 0, 0, 0, 0, 0, 0, 0, 0},
     {0, 0, 0, 0, 0, 3, 0, 8, 5},
     {0, 0, 1, 0, 2, 0, 0, 0, 0},
     {0, 0, 0, 5, 0, 7, 0, 0, 0},
     {0, 0, 4, 0, 0, 0, 1, 0, 0},
     {0, 9, 0, 0, 0, 0, 0, 0, 0},
     {5, 0, 0, 0, 0, 0, 0, 7, 3},
     {0, 0, 2, 0, 1, 0, 0, 0, 0},
     {0, 0, 0, 0, 4, 0, 0, 0, 9}};*/
    
    
    return 0;
}



