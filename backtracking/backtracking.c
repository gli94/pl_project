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

bool checkcomplete(int grid[N][N], int *row, int *col)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (grid[i][j] == UNASSINED)
            {
                *row = i;
                *col = j;
                return false;
            }
        }
    }
    
    return true;
}

bool checkrow(int grid[N][N], int row, int value)
{
    for (int col = 0; col < N; col++)
    {
        if (grid[row][col] == value)
        {
            return false;
        }
    }
    
    return true;
}

bool checkcol(int grid[N][N], int col, int value)
{
    for (int row = 0; row < N; row++)
    {
        if (grid[row][col] == value)
        {
            return false;
        }
    }
    
    return true;
}

bool checkbox(int grid[N][N], int box_start_row, int box_start_col, int value)
{
    for (int row = box_start_row; row < box_start_row + BLOCK_SIZE; row++)
    {
        for (int col = box_start_col; col < box_start_col + BLOCK_SIZE; col++)
        {
            if (grid[row][col] == value)
            {
                return false;
            }
        }
    }
    
    return true;
}

bool isvalid(int grid[N][N], int row, int col, int value)
{
    if (checkrow(grid, row, value) && checkcol(grid, col, value) && checkbox(grid, row - row % BLOCK_SIZE, col - col % BLOCK_SIZE, value) && (grid[row][col] == UNASSINED))
    {
        return true;
    }
    else
    {
        return false;
    }
}

void printgrid(int grid[N][N])
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%d ", grid[i][j]);
        }
        printf("\n");
    }
}

bool sudoku_solver(int grid[N][N])
{
    int row, col;
    
    if(checkcomplete(grid, &row, &col))
    {
        return true;
    }
    
    for (int value = 1; value < N + 1; value++)
    {
        if (isvalid(grid, row, col, value))
        {
            grid[row][col] = value;
            
            if (sudoku_solver(grid))
            {
                return true;
            }
            
            grid[row][col] = UNASSINED;
        }
    }
    
    return false;
}

int main()
{
    
    int data[N * NUM_TESTCASE][N];
    int grid[N][N];
    
    double execution_time[NUM_TESTCASE];
    double time_spent = 0.0;
    
    char buf[MAXB] = {0};
    
    int i = 0;
    int j = 0;
    
    
    FILE * file;
    file = fopen("sudoku_9x9_100_48.txt", "r");
    
    if (file == 0)
    {
        fprintf(stderr, "failed to open test.txt\n");
        return 1;
    }
    
    
    
    while (i < MAXL && fgets (buf, MAXB - 1, file))
    {
        sscanf(buf, "%d %d %d %d %d %d %d %d %d", &data[i][0], &data[i][1], &data[i][2], &data[i][3], &data[i][4], &data[i][5], &data[i][6], &data[i][7], &data[i][8]);
        i++;
    }
    
    fclose(file);
    
    for (i = 0; i < NUM_TESTCASE; i++)
    {
        for (int ii = 0; ii < N; ii++)
        {
            for (int jj = 0; jj < N; jj++)
            {
                grid[ii][jj] = data[ii + i * N][jj];
            }
        }
        
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
        time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
        
        execution_time[i] = time_spent;
        
        printf("Execution time: %lfs\n", time_spent);
        
    }
    
    FILE * f = fopen("bt_stats_bench_9x9_100_48.txt", "w");
    
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
    
    for (i = 0; i < NUM_TESTCASE; i++)
    {
        sum += execution_time[i];
        fprintf(f, "%lf ", execution_time[i]);
    }
    
    average = sum / NUM_TESTCASE;
    
    sum = 0.0;
    
    for (i = 0; i < NUM_TESTCASE; i++)
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


