#include<stdio.h>
#include<stdlib.h>
#include<stdbool.h>
#include<string.h>
#include<math.h>
#include<time.h>

#define N 9
#define BLOCK_SIZE 3
#define UNASSIGNED 0
#define TEMP 0.5
#define ALPHA 0.99999
#define MAX_ITER 1000000
#define MAXB 1000
#define NUM_TESTCASE 100
#define MAXL N*NUM_TESTCASE

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

void rand_init(int grid[N][N])
{
    int value;
    
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (grid[i][j] == UNASSIGNED)
            {
                while(1)
                {
                    value = (rand() % N) + 1;
                    if (checkbox(grid, i - i % BLOCK_SIZE, j - j % BLOCK_SIZE, value))
                    {
                        grid[i][j] = value;
                        break;
                    }
                }
            }
        }
    }
}

/*int get_cost(int grid[N][N])
{
    int cost = 0;
    int unique_num = 0;
    int * score_board = malloc(N * sizeof(int));
    
    for (int row = 0; row < N; row++)
    {
        memset(score_board, 0, N * sizeof(int));
        unique_num = 0;
        
        for (int i = 0; i < N; i++)
        {
            score_board[grid[row][i] - 1] += 1;
        }
        
        for (int j = 0; j < N; j++)
        {
            if (score_board[j] != 0)
            {
                unique_num++;
            }
        }
        
        cost += (-1) * unique_num;
    }
    
    for (int col = 0; col < N; col++)
    {
        memset(score_board, 0, N * sizeof(int));
        unique_num = 0;
        
        for (int i = 0; i < N; i++)
        {
            score_board[grid[i][col] - 1] += 1;
        }
        
        for (int j = 0; j < N; j++)
        {
            if (score_board[j] != 0)
            {
                unique_num++;
            }
        }
        
        cost += (-1) * unique_num;
    }
    
    free(score_board);
    return cost;
}*/

int get_cost(int grid[N][N])
{
    int cost = 0;
    int unique_num = 0;
    
    int i = 0;
    int j = 0;
    
    for (int row = 0; row < N; row++)
    {
        unique_num = 0;
        
        for (i = 0; i < N; i++)
        {
            for (j = 0; j < i; j++)
            {
                if(grid[row][j] == grid[row][i])
                {
                    break;
                }
            }
            
            if (i == j)
            {
                unique_num++;
            }
        }
        
        cost += (-1) * unique_num;
    }
    
    for (int col = 0; col < N; col++)
    {
        unique_num = 0;
        
        for (i = 0; i < N; i++)
        {
            for (j = 0; j < i; j++)
            {
                if(grid[j][col] == grid[i][col])
                {
                    break;
                }
            }
            
            if (i == j)
            {
                unique_num++;
            }
        }
        
        cost += (-1) * unique_num;
    }
    
    return cost;
}

void gen_candidate(int grid[N][N], int candidate[N][N], int initial_grid[N][N])
{
    for (int i = 0; i < N; i++)
    {
        memcpy(&candidate[i], &grid[i], sizeof(grid[0]));
    }
    
    int blockIdx = rand() % N;
    int element1_index = 0;
    int element2_index = 0;
    int row1 = 0;
    int col1 = 0;
    int row2 = 0;
    int col2 = 0;
    
    int empty_element_cnt = 0;
    
    while(1)
    {
        
        empty_element_cnt = 0;
        blockIdx = rand() % N;
        
        for (int i = 0; i < BLOCK_SIZE; i++)
        {
            for (int j = 0; j < BLOCK_SIZE; j++)
            {
                if (initial_grid[(blockIdx / BLOCK_SIZE) * BLOCK_SIZE + i][(blockIdx % BLOCK_SIZE) * BLOCK_SIZE + j] == UNASSIGNED)
                {
                    empty_element_cnt++;
                    row1 = (blockIdx / BLOCK_SIZE) * BLOCK_SIZE + i;
                    col1 = (blockIdx % BLOCK_SIZE) * BLOCK_SIZE + j;
                    
                }
            }
        }
        
        if (empty_element_cnt == 0)
        {
            blockIdx = rand() % N;
        }
        else
        {
            break;
        }
    }
    
    int sum = 0;
    
    if (empty_element_cnt == 1)
    {
        for (int i = 0; i < BLOCK_SIZE; i++)
        {
            for (int j = 0; j < BLOCK_SIZE; j++)
            {
                sum += initial_grid[(blockIdx / BLOCK_SIZE) * BLOCK_SIZE + i][(blockIdx % BLOCK_SIZE) * BLOCK_SIZE + j];
            }
        }
        
        candidate[row1][col1] = 45-sum;
        return;
    }
    
    
    while(1)
    {
        element1_index = rand() % N;
        element2_index = rand() % N;
        row1 = (blockIdx / BLOCK_SIZE) * BLOCK_SIZE + element1_index / BLOCK_SIZE;
        col1 = (blockIdx % BLOCK_SIZE) * BLOCK_SIZE + element1_index % BLOCK_SIZE;
        row2 = (blockIdx / BLOCK_SIZE) * BLOCK_SIZE + element2_index / BLOCK_SIZE;
        col2 = (blockIdx % BLOCK_SIZE) * BLOCK_SIZE + element2_index % BLOCK_SIZE;
        
        if ((element1_index != element2_index) && (initial_grid[row1][col1] == UNASSIGNED) && (initial_grid[row2][col2] == UNASSIGNED))
        {
            break;
        }
    }
    
    
    int tmp = 0;
    
    tmp = candidate[row1][col1];
    candidate[row1][col1] = candidate[row2][col2];
    candidate[row2][col2] = tmp;
    
    return;
}

void update_grid(int grid[N][N], int candidate[N][N])
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            grid[i][j] = candidate[i][j];
        }
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
    int candidate[N][N];
    int initial_grid[N][N];
    int current_cost = 0;
    int candidate_cost = 0;
    float delta_cost = 0.0;
    float T = TEMP;
    int count  = 0;
    
    update_grid(initial_grid, grid);
    
    //printf("Update grid done !\n");
    
    srand(time(NULL));
    
    rand_init(grid);
    
    //printf("Rand init done !\n");
    /*printgrid(grid);
    
    printf("Initial cost = %d\n", get_cost(grid));*/
    
    while(count < MAX_ITER)
    {
        gen_candidate(grid, candidate, initial_grid);
        //printf("Gen candidate done !\n");
        current_cost = get_cost(grid);
        candidate_cost = get_cost(candidate);
        delta_cost = (float)(current_cost - candidate_cost);
        
        /*printf("Iteration #%d:\n", count);
        printf("current_cost = %d\n", current_cost);
        printf("candidate_cost = %d\n", candidate_cost);
        printf("\n");*/
        
        if (exp(delta_cost / T) > ((float)(rand()) / RAND_MAX))
        {
            //printf("update!\n");
            //printgrid(candidate);
            //printf("\n");
            update_grid(grid, candidate);
            //printgrid(grid);
            current_cost = candidate_cost;
        }
        
        if (candidate_cost == (-2) * N * N)
        {
            update_grid(grid, candidate);
            break;
        }
        
        T = ALPHA * T;
        count++;
    }
    
    //printf("final cost = %d\n", current_cost);
    printf("Iterations: %d \n", count);
    if (get_cost(grid) == (-2) * N * N)
    {
        return true;
    }
    else
    {
        return false;
    }
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
    file = fopen("sudoku_9x9_100_40.txt", "r");
    
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
        
        printgrid(grid);
        
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
    
    FILE * f = fopen("sa_stats_bench_9x9_100_40.txt", "w");
    
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
    
    /*int grid[N][N] = {{5,3,0,0,7,0,0,0,0},
        {6,0,0,1,9,5,0,0,0},
        {0,9,8,0,0,0,0,6,0},
        {8,0,0,0,6,0,0,0,3},
        {4,0,0,8,0,3,0,0,1},
        {7,0,0,0,2,0,0,0,6},
        {0,6,0,0,0,0,2,8,0},
        {0,0,0,4,1,9,0,0,5},
        {0,0,0,0,8,0,0,7,9}};*/
    
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
    
    /* int grid[N][N] = {{0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 3, 0, 8, 5},
        {0, 0, 1, 0, 2, 0, 0, 0, 0},
        {0, 0, 0, 5, 0, 7, 0, 0, 0},
        {0, 0, 4, 0, 0, 0, 1, 0, 0},
        {0, 9, 0, 0, 0, 0, 0, 0, 0},
        {5, 0, 0, 0, 0, 0, 0, 7, 3},
        {0, 0, 2, 0, 1, 0, 0, 0, 0},
        {0, 0, 0, 0, 4, 0, 0, 0, 9}}; */
    
    /*int grid[N][N] = {{0, 0, 0, 0, 0, 0, 6, 8, 0},
        {0, 0, 0, 0, 7, 3, 0, 0, 9},
        {3, 0, 9, 0, 0, 0, 0, 4, 5},
        {4, 9, 0, 0, 0, 0, 0, 0, 0},
        {8, 0, 3, 0, 5, 0, 9, 0, 2},
        {0, 0, 0, 0, 0, 0, 0, 3, 6},
        {9, 6, 0, 0, 0, 0, 3, 0, 8},
        {7, 0, 0, 6, 8, 0, 0, 0, 0},
        {0, 2, 8, 0, 0, 0, 0, 0, 0}};*/
    
    
    return 0;
}





