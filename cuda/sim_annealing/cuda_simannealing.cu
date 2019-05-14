#include <cmath>
#include <cstdio>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <curand.h>
#include <curand_kernel.h>

#include "CycleTimer.h"
#include "cuda_simannealing.cuh"


#define N 9
#define BLOCK_SIZE 3
#define UNASSIGNED 0
#define TEMP 0.5
#define ALPHA 0.99999
#define MAX_ITER 1000000

__device__
bool checkbox(int * grid, int box_start_row, int box_start_col, int value)
{
    for (int row = box_start_row; row < box_start_row + BLOCK_SIZE; row++)
    {
        for (int col = box_start_col; col < box_start_col + BLOCK_SIZE; col++)
        {
            if (grid[row * N + col] == value)
            {
                return false;
            }
        }
    }
    
    return true;
}


__device__
void rand_init(int * grid, curandState* globalState, int ind)
{
    int value;
    
    curandState state;
    curand_init(clock64(), ind, 0, &state);
    
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (grid[i * N + j] == UNASSIGNED)
            {
                while(1)
                {
                    value = curand(&state) % N + 1;
                    if (checkbox(grid, i - i % BLOCK_SIZE, j - j % BLOCK_SIZE, value))
                    {
                        grid[i * N + j] = value;
                        break;
                    }
                }
            }
        }
    }
}

__device__
int get_cost(int * grid)
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
                if(grid[row * N + j] == grid[row * N + i])
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
                if(grid[j * N + col] == grid[i * N + col])
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

__device__
void gen_candidate(int * grid, int * candidate, int * initial_grid, curandState* globalState, int ind)
{
    
    for (int i = 0; i < N * N; i++)
    {
        candidate[i] = grid[i];
    }
    
    int blockIdx = 0;
    int element1_index = 0;
    int element2_index = 0;
    int row1 = 0;
    int col1 = 0;
    int row2 = 0;
    int col2 = 0;
    
    int empty_element_cnt = 0;
    
    curandState state;
    curand_init(clock64(), ind, 0, &state);
    while(1)
    {
        
        empty_element_cnt = 0;
        blockIdx = curand(&state) % N;
        
        for (int i = 0; i < BLOCK_SIZE; i++)
        {
            for (int j = 0; j < BLOCK_SIZE; j++)
            {
                if (initial_grid[((blockIdx / BLOCK_SIZE) * BLOCK_SIZE + i) * N + (blockIdx % BLOCK_SIZE) * BLOCK_SIZE + j] == UNASSIGNED)
                {
                    empty_element_cnt++;
                    row1 = (blockIdx / BLOCK_SIZE) * BLOCK_SIZE + i;
                    col1 = (blockIdx % BLOCK_SIZE) * BLOCK_SIZE + j;
                    
                }
            }
        }
        
        if (empty_element_cnt == 0)
        {
            blockIdx = curand(&state) % N;
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
                sum += initial_grid[((blockIdx / BLOCK_SIZE) * BLOCK_SIZE + i) * N + (blockIdx % BLOCK_SIZE) * BLOCK_SIZE + j];
            }
        }
        
        candidate[row1 * N + col1] = N * (N + 1) / 2 - sum;
        return;
    }
    
    curand_init(clock64(), ind, 0, &state);
    
    while(1)
    {
        element1_index = curand(&state) % N;
        element2_index = curand(&state) % N;
        row1 = (blockIdx / BLOCK_SIZE) * BLOCK_SIZE + element1_index / BLOCK_SIZE;
        col1 = (blockIdx % BLOCK_SIZE) * BLOCK_SIZE + element1_index % BLOCK_SIZE;
        row2 = (blockIdx / BLOCK_SIZE) * BLOCK_SIZE + element2_index / BLOCK_SIZE;
        col2 = (blockIdx % BLOCK_SIZE) * BLOCK_SIZE + element2_index % BLOCK_SIZE;
        
        if ((element1_index != element2_index) && (initial_grid[row1 * N + col1] == UNASSIGNED) && (initial_grid[row2 * N + col2] == UNASSIGNED))
        {
            break;
        }
    }
    
    
    int tmp = 0;
    
    tmp = candidate[row1 * N + col1];
    candidate[row1 * N + col1] = candidate[row2 * N + col2];
    candidate[row2 * N + col2] = tmp;
    
    return;
}

__device__
void update_grid(int * grid, int * candidate)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            grid[i * N + j] = candidate[i * N + j];
        }
    }
}

__global__
void init_memory(int * grids, int total_boards)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (index < total_boards)
    {
        for (int j = 0; j < N * N; j++)
        {
            grids[index * N * N + j] = grids[j];
        }
    }
    
}

__global__
void cuda_sim_annealing(int *grid,
                        const int num_boards,
                        int *candidate,
                        int *initial_grid,
                        int *finished,
                        int *solved,
                        curandState* devStates)
{
    
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    
    int *current_grid;
    int *current_candidate;
    int *current_initial_grid;
    
    while((*finished == 0) && (index < num_boards))
    {
        
        current_grid = grid + index * N * N;
        current_candidate = candidate + index * N * N;
        current_initial_grid = initial_grid + index * N * N;
    
        int current_cost = 0;
        int candidate_cost = 0;
        float delta_cost = 0.0;
        float T = TEMP;
        int count  = 0;
        
        curand_init(clock64(), index, 0, &devStates[index]);
    
        update_grid(current_initial_grid, current_grid);
    
    
        rand_init(current_grid, devStates, index);
    
    while((count < MAX_ITER) && (*finished == 0))
    {
        gen_candidate(current_grid, current_candidate, current_initial_grid, devStates, index);
        current_cost = get_cost(current_grid);
        candidate_cost = get_cost(current_candidate);
        delta_cost = (float)(current_cost - candidate_cost);
        
        if (exp(delta_cost / T) > curand_uniform(&devStates[index]))
        {
            update_grid(current_grid, current_candidate);
            current_cost = candidate_cost;
        }
        
        if (candidate_cost == (-2) * N * N)
        {
            *finished = 1;
            update_grid(current_grid, current_candidate);
            
            printf("Finished!\n");
            
            for (int i = 0; i < N * N; i++)
            {
                solved[i] = current_grid[i];
            }
            
            break;
        }
        
        T = ALPHA * T;
        count++;
    }
        
    index += gridDim.x * blockDim.x;
    }
}

void callSAKernel (const int blocksPerGrid,
                   const int threadsPerBlock,
                   int *grid,
                   const int num_boards,
                   int *candidate,
                   int *initial_grid,
                   int *finished,
                   int *solved,
                   curandState* devStates)
{
    cuda_sim_annealing<<<blocksPerGrid, threadsPerBlock>>>(grid, num_boards, candidate, initial_grid, finished, solved, devStates);
}

void cuda_SimAnnealing(int * board, int * solved)
{
    int blocksPerGrid = 1024;
    int threadsPerBlock = 256;
    
    curandState* devStates;
    cudaMalloc ( &devStates, blocksPerGrid * threadsPerBlock *sizeof( curandState ) );

    
    int *grids;
    int *candidate;
    int *initial_grid;
    
    int total_boards = blocksPerGrid * threadsPerBlock;
    
    cudaMalloc(&grids, total_boards * N * N * sizeof(int));
    cudaMalloc(&candidate, total_boards * N * N * sizeof(int));
    cudaMalloc(&initial_grid, total_boards * N * N * sizeof(int));

    cudaMemset(grids, 0, total_boards * N * N * sizeof(int));
    cudaMemset(candidate, 0, total_boards * N * N * sizeof(int));
    cudaMemset(initial_grid, 0, total_boards * N * N * sizeof(int));
    
    cudaMemcpy(grids, board, N * N * sizeof(int), cudaMemcpyHostToDevice);
    
    
    init_memory<<<blocksPerGrid, threadsPerBlock>>>(grids, total_boards);
    
    
    int *dev_finished;
    int *dev_solved;
    
    cudaMalloc(&dev_finished, sizeof(int));
    cudaMalloc(&dev_solved, N * N * sizeof(int));
    
    cudaMemset(dev_finished, 0, sizeof(int));
    cudaMemcpy(dev_solved, board, N * N * sizeof(int), cudaMemcpyHostToDevice);
    
    
    double startGPUTime = CycleTimer::currentSeconds();
    
    callSAKernel (blocksPerGrid, threadsPerBlock, grids, total_boards, candidate, initial_grid, dev_finished, dev_solved, devStates);
    cudaDeviceSynchronize();
    
    double endGPUTime = CycleTimer::currentSeconds();
    double timeKernel = endGPUTime - startGPUTime;
    
    printf("Execution time: %lfs\n", timeKernel);
    
    cudaMemcpy(solved, dev_solved, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    
    double endGPUTime2 = CycleTimer::currentSeconds();
    
    printf("Memcpy time: %lfs\n", endGPUTime2-endGPUTime);
    
    cudaFree(grids);
    cudaFree(candidate);
    cudaFree(initial_grid);
    
    cudaFree(dev_finished);
    cudaFree(dev_solved);
}

