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

#include "cuda_simannealing.cuh"


#define N 9
#define BLOCK_SIZE 3
#define UNASSIGNED 0
#define TEMP 0.5
#define ALPHA 0.99999
#define MAX_ITER 1000000

/*__device__
bool checkrow(int *grid, int Num, int row)
{
 
    bool seen[N];
    
    for (int i = 0; i < Num; i++)
    {
        seen[i] = false;
    }
        
    for (int col = 0; col < Num; col++)
    {
        int val = grid[row * Num + col];
        if (val > 0)
        {
            if(seen[val-1])
            {
                return false;
            }
            else
            {
                seen[val-1] = true;
            }
        }
    }
    
    return true;
}

__device__
bool checkcol(int *grid, int Num, int col)
{
    bool seen[N];
    
    for (int i = 0; i < Num; i++)
    {
        seen[i] = false;
    }
    
    for (int row = 0; row < Num; row++)
    {
        int val = grid[row * Num + col];
        if (val > 0)
        {
            if(seen[val-1])
            {
                return false;
            }
            else
            {
                seen[val-1] = true;
            }
        }
    }
    
    return true;
}

__device__
bool checkbox(int *grid, int Num, int box_start_row, int box_start_col)
{
    
    bool seen[N];
    
    for (int i = 0; i < Num; i++)
    {
        seen[i] = false;
    }
    
    for (int row = box_start_row; row < box_start_row + BLOCK_SIZE; row++)
    {
        for (int col = box_start_col; col < box_start_col + BLOCK_SIZE; col++)
        {
            int val = grid[row * Num + col];
            if (val > 0)
            {
                if(seen[val-1])
                {
                    return false;
                }
                else
                {
                    seen[val-1] = true;
                }
            }
            
        }
    }
    
    return true;
}

__device__
bool isvalid(int *grid, int Num, int row, int col)
{
    
    if (checkrow(grid, Num, row) && checkcol(grid, Num, col) && checkbox(grid, Num, row - row % BLOCK_SIZE, col - col % BLOCK_SIZE))
    {
        return true;
    }
    else
    {
        return false;
    }
}

__global__
void sudoku_backtrack( int *boards,
                       const int num_boards,
                       int *empty_spaces,
                       int *num_empty_spaces,
                       int *finished,
                       int *solved)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    
    int *currentBoard;
    int *currentEmptySpaces;
    int currentNumEmptySpaces;
    
    
    while((*finished == 0) && (index < num_boards))
    {
        
        int emptyIndex = 0;
        int row;
        int col;
        //int value = 0;
        
        currentBoard = boards + index * N * N;
        currentEmptySpaces = empty_spaces + index * N * N;
        currentNumEmptySpaces = num_empty_spaces[index];
        
        
        while ((emptyIndex >= 0) && (emptyIndex < currentNumEmptySpaces))
        {
            currentBoard[currentEmptySpaces[emptyIndex]]++;
            
            row = currentEmptySpaces[emptyIndex] / N;
            col = currentEmptySpaces[emptyIndex] % N;
 
            
            if(!isvalid(currentBoard, N, row, col))
            {
                if(currentBoard[currentEmptySpaces[emptyIndex]] >= 9)
                {
                    currentBoard[currentEmptySpaces[emptyIndex]] = 0;
                    emptyIndex--;
                }
            }
            else
            {
                //printf("Valid!\n");
               // printf("EmptyIndex = %d, EmptySpaces = %d \n", emptyIndex, currentNumEmptySpaces);
                //currentBoard[currentEmptySpaces[emptyIndex]] = value;
                //printf("Value filled in: %d\n", currentBoard[currentEmptySpaces[emptyIndex]]);
                //value = 0;
                emptyIndex++;
            }
        }
        
        if(emptyIndex == currentNumEmptySpaces)
        {
            *finished = 1;
            
            printf("Finished!\n");
            
            for (int i = 0; i < N * N; i++)
            {
                solved[i] = currentBoard[i];
            }
        }
        
        index += gridDim.x * blockDim.x;
    }
}


void cuda_sudokuBacktrack (const int blocksPerGrid,
                           const int threadsPerBlock,
                           int * boards,
                           const int num_boards,
                           int *empty_spaces,
                           int *num_empty_spaces,
                           int *finished,
                           int *solved)
{
    sudoku_backtrack<<<blocksPerGrid, threadsPerBlock>>>(boards, num_boards, empty_spaces, num_empty_spaces, finished, solved);
}

__global__
void
cudaBFSKernel(int *old_boards,
              int *new_boards,
              int total_boards,
              int *board_index,
              int *empty_spaces,
              int *empty_space_count) {
    
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // board_index must start at zero
    
    while (index < total_boards) {
        // find the next empty spot
        int found = 0;
        
        for (int i = (index * N * N); (i < (index * N * N) + N * N) && (found == 0); i++) {
            // found a open spot
            if (old_boards[i] == 0) {
                found = 1;
                // get the correct row and column shits
                int temp = i - N * N * index;
                int row = temp / N;
                int col = temp % N;
                
                // figure out which numbers work here
                for (int attempt = 1; attempt <= N; attempt++) {
                    int works = 1;
                    // row constraint, test various columns
                    for (int c = 0; c < N; c++) {
                        if (old_boards[row * N + c + N * N * index] == attempt) {
                            works = 0;
                        }
                    }
                    // column contraint, test various rows
                    for (int r = 0; r < N; r++) {
                        if (old_boards[r * N + col + N * N * index] == attempt) {
                            works = 0;
                        }
                    }
                    // box constraint
                    for (int r = BLOCK_SIZE * (row / BLOCK_SIZE); r < BLOCK_SIZE * (row / BLOCK_SIZE) + BLOCK_SIZE; r++) {
                        for (int c = BLOCK_SIZE * (col / BLOCK_SIZE); c < BLOCK_SIZE * (col / BLOCK_SIZE) + BLOCK_SIZE; c++) {
                            if (old_boards[r * N + c + N * N * index] == attempt) {
                                works = 0;
                            }
                        }
                    }
                    if (works == 1) {
                        // copy the whole board
                        
                        int next_board_index = atomicAdd(board_index, 1);
                        int empty_index = 0;
                        for (int r = 0; r < 9; r++) {
                            for (int c = 0; c < 9; c++) {
                                new_boards[next_board_index * 81 + r * 9 + c] = old_boards[index * 81 + r * 9 + c];
                                if (old_boards[index * 81 + r * 9 + c] == 0 && (r != row || c != col)) {
                                    empty_spaces[empty_index + 81 * next_board_index] = r * 9 + c;
                                    
                                    empty_index++;
                                }
                            }
                        }
                        empty_space_count[next_board_index] = empty_index;
                        new_boards[next_board_index * 81 + row * 9 + col] = attempt;
                        
                    }
                }
            }
        }
        
        index += blockDim.x * gridDim.x;
    }
}

void callBFSKernel( const int blocksPerGrid,
                    const int threadsPerBlock,
                   int *old_boards,
                   int *new_boards,
                   int total_boards,
                   int *board_index,
                   int *empty_spaces,
                   int *empty_space_count)
{
    cudaBFSKernel<<<blocksPerGrid, threadsPerBlock>>>(old_boards, new_boards, total_boards, board_index, empty_spaces, empty_space_count);
}

void cuda_Backtrack(int * board, int * solved)
{
    int blocksPerGrid = 1024;
    int threadsPerBlock = 256;
 
    int *old_boards;
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
    
    cudaMemcpy(old_boards, board, N * N * sizeof(int), cudaMemcpyHostToDevice);
    
    callBFSKernel(blocksPerGrid, threadsPerBlock, old_boards, new_boards, total_boards, board_index, empty_spaces, empty_space_count);
    
    int host_count;
    
    int iterations = 18;
    
    for (int i=0; i<iterations; i++)
    {
        cudaMemcpy(&host_count, board_index, sizeof(int), cudaMemcpyDeviceToHost);
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
    
    cuda_sudokuBacktrack(blocksPerGrid, threadsPerBlock, new_boards, host_count, empty_spaces, empty_space_count, dev_finished, dev_solved);
    
    cudaMemcpy(solved, dev_solved, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(empty_spaces);
    cudaFree(empty_space_count);
    cudaFree(new_boards);
    cudaFree(old_boards);
    cudaFree(board_index);
    
    cudaFree(dev_finished);
    cudaFree(dev_solved);
}*/

__device__ float generate( curandState* globalState, int ind )
{
    //int ind = threadIdx.x;
    curandState localState = globalState[ind];
    float RANDOM = curand_uniform( &localState );
    globalState[ind] = localState;
    return RANDOM;
}

__global__ void setup_kernel ( curandState * state, unsigned long seed )
{
    int id = threadIdx.x;
    curand_init ( seed, id, 0, &state[id] );
}

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
                    //value = (rand() % N) + 1;
                    //value = (((int) (generate(globalState, ind) * 1000000)) % N) + 1;
                    //printf("Stuck1!\n");
                    value = curand(&state) % N + 1;
                    //printf("value=%d\n", value);
                    if (checkbox(grid, i - i % BLOCK_SIZE, j - j % BLOCK_SIZE, value))
                    {
                        grid[i * N + j] = value;
                        break;
                    }
                }
                //printf("randinit done!\n");
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
    
    //cudaMemcpy(candidate, grid, N * N * sizeof(int), cudaMemcpyDeviceToDevice);
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
        //blockIdx = rand() % N;
        //blockIdx = (int) (generate(globalState, ind) * 1000000) % N;
        blockIdx = curand(&state) % N;
        //printf("Stuck2!\n");
        
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
            //blockIdx = rand() % N;
            //blockIdx = (int) (generate(globalState, ind) * 1000000) % N;
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
        //element1_index = rand() % N;
        //element2_index = rand() % N;
        //printf("Stuck3!\n");
        //element1_index = (int) (generate(globalState, ind) * 1000000) % N;
        //element2_index = (int) (generate(globalState, ind) * 1000000) % N;
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
    
        //printf("Update grid done !\n");
    
        //srand(time(NULL));
    
        rand_init(current_grid, devStates, index);
    
       //printf("Rand init done !\n");
       /*printgrid(grid);
     
       printf("Initial cost = %d\n", get_cost(grid));*/
    
    while((count < MAX_ITER) && (*finished == 0))
    {
        gen_candidate(current_grid, current_candidate, current_initial_grid, devStates, index);
        //printf("Gen candidate done !\n");
        current_cost = get_cost(current_grid);
        candidate_cost = get_cost(current_candidate);
        delta_cost = (float)(current_cost - candidate_cost);
        
         //printf("Iteration #%d:\n", count);
         //printf("current_cost = %d\n", current_cost);
         //printf("candidate_cost = %d\n", candidate_cost);
         //printf("\n");
        
        if (exp(delta_cost / T) > curand_uniform(&devStates[index]))
        {
            //printf("update!\n");
            //printgrid(candidate);
            //printf("\n");
            update_grid(current_grid, current_candidate);
            //printgrid(grid);
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
    //printf("final cost = %d\n", current_cost);
    //printf("Iterations: %d \n", count);
    
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
    
 
    //setup_kernel <<< 1, blocksPerGrid * threadsPerBlock >>> ( devStates,unsigned(time(NULL)) );

    
    
    int *grids;
    int *candidate;
    int *initial_grid;
    //int *new_boards;
    //int *empty_spaces;
    //int *empty_space_count;
    //int *board_index;
    
    int total_boards = blocksPerGrid * threadsPerBlock;
    
   // int sk = pow(2, 26);
    
    //cudaMalloc(&empty_spaces, sk * sizeof(int));
    //cudaMalloc(&empty_space_count, ((sk / 81) + 1) * sizeof(int));
    cudaMalloc(&grids, total_boards * N * N * sizeof(int));
    cudaMalloc(&candidate, total_boards * N * N * sizeof(int));
    cudaMalloc(&initial_grid, total_boards * N * N * sizeof(int));
    //cudaMalloc(&new_boards, sk * sizeof(int));
    //cudaMalloc(&board_index, sizeof(int));
    
    
    //cudaMemset(board_index, 0, sizeof(int));
    //cudaMemset(new_boards, 0, sk * sizeof(int));
    cudaMemset(grids, 0, total_boards * N * N * sizeof(int));
    cudaMemset(candidate, 0, total_boards * N * N * sizeof(int));
    cudaMemset(initial_grid, 0, total_boards * N * N * sizeof(int));
    
    cudaMemcpy(grids, board, N * N * sizeof(int), cudaMemcpyHostToDevice);
    
    /*for (int i = 1; i < total_boards; i++)
    {
        for (int j = 0; j < N * N; j++)
        {
            //grids[i * N * N + j] = grids[j];
            printf("%d\n ", grids[i*N*N + j]);
        }
    }*/
    
    init_memory<<<blocksPerGrid, threadsPerBlock>>>(grids, total_boards);
    
    /*for (int j = 0; j < N * N; j++)
    {
        printf("%d\n ", grids[j]);
    }*/
            
    
    //callBFSKernel(blocksPerGrid, threadsPerBlock, old_boards, new_boards, total_boards, board_index, empty_spaces, empty_space_count);
    
    /*int host_count;
    
    int iterations = 18;
    
    for (int i=0; i<iterations; i++)
    {
        cudaMemcpy(&host_count, board_index, sizeof(int), cudaMemcpyDeviceToHost);
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
    printf("new number of boards retrieved is %d\n", host_count);*/
    
    int *dev_finished;
    int *dev_solved;
    
    cudaMalloc(&dev_finished, sizeof(int));
    cudaMalloc(&dev_solved, N * N * sizeof(int));
    
    cudaMemset(dev_finished, 0, sizeof(int));
    cudaMemcpy(dev_solved, board, N * N * sizeof(int), cudaMemcpyHostToDevice);
    
    /*if((iterations % 2) == 1)
    {
        new_boards = old_boards;
    }*/
    
    //cuda_sudokuBacktrack(blocksPerGrid, threadsPerBlock, new_boards, host_count, empty_spaces, empty_space_count, dev_finished, dev_solved);
    callSAKernel (blocksPerGrid, threadsPerBlock, grids, total_boards, candidate, initial_grid, dev_finished, dev_solved, devStates);
    
    cudaMemcpy(solved, dev_solved, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(grids);
    cudaFree(candidate);
    cudaFree(initial_grid);
    
    cudaFree(dev_finished);
    cudaFree(dev_solved);
}

