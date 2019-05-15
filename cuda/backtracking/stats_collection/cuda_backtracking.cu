#include <cmath>
#include <cstdio>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cuda_backtracking.cuh"
#include "CycleTimer.h"

#define N 9
#define BLOCK_SIZE 3
#define UNASSINED 0

__device__
void clearBitmap(bool *map, int size) {
    for (int i = 0; i < size; i++) {
        map[i] = false;
    }
}


/**
 * This device checks the entire board to see if it is valid.
 *
 * board: this is a N * N sized array that stores the board to check. Rows are stored contiguously,
 *        so to access row r and col c, use board[r * N + c]
 */
__device__
bool validBoard(const int *board) {
    bool seen[N];
    clearBitmap(seen, N);
    
    // check if rows are valid
    for (int i = 0; i < N; i++) {
        clearBitmap(seen, N);
        
        for (int j = 0; j < N; j++) {
            int val = board[i * N + j];
            
            if (val != 0) {
                if (seen[val - 1]) {
                    return false;
                } else {
                    seen[val - 1] = true;
                }
            }
        }
    }
    
    // check if columns are valid
    for (int j = 0; j < N; j++) {
        clearBitmap(seen, N);
        
        for (int i = 0; i < N; i++) {
            int val = board[i * N + j];
            
            if (val != 0) {
                if (seen[val - 1]) {
                    return false;
                } else {
                    seen[val - 1] = true;
                }
            }
        }
    }
    
    int n = BLOCK_SIZE;
    
    // finally check if the sub-boards are valid
    for (int ridx = 0; ridx < n; ridx++) {
        for (int cidx = 0; cidx < n; cidx++) {
            clearBitmap(seen, N);
            
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    int val = board[(ridx * n + i) * N + (cidx * n + j)];
                    
                    if (val != 0) {
                        if (seen[val - 1]) {
                            return false;
                        } else {
                            seen[val-1] = true;
                        }
                    }
                }
            }
        }
    }
    
    
    // if we get here, then the board is valid
    return true;
}

__device__
bool validBoard(const int *board, int changed) {
    
    int r = changed / 9;
    int c = changed % 9;
    
    // if changed is less than 0, then just default case
    if (changed < 0) {
        return validBoard(board);
    }
    
    if ((board[changed] < 1) || (board[changed] > 9)) {
        return false;
    }
    
    bool seen[N];
    clearBitmap(seen, N);
    
    // check if row is valid
    for (int i = 0; i < N; i++) {
        int val = board[r * N + i];
        
        if (val != 0) {
            if (seen[val - 1]) {
                return false;
            } else {
                seen[val - 1] = true;
            }
        }
    }
    
    // check if column is valid
    clearBitmap(seen, N);
    for (int j = 0; j < N; j++) {
        int val = board[j * N + c];
        
        if (val != 0) {
            if (seen[val - 1]) {
                return false;
            } else {
                seen[val - 1] = true;
            }
        }
    }
    
    int n = BLOCK_SIZE;
    // finally check if the sub-board is valid
    int ridx = r / n;
    int cidx = c / n;
    
    clearBitmap(seen, N);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int val = board[(ridx * n + i) * N + (cidx * n + j)];
            
            if (val != 0) {
                if (seen[val - 1]) {
                    return false;
                } else {
                    seen[val - 1] = true;
                }
            }
        }
    }
    
    // if we get here, then the board is valid
    return true;
}

__device__
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

void cuda_Backtrack(int * board, int * solved, double *exec_time, int *finished)
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
    
    double total_time = 0.0;
    
    double startGPUTime = CycleTimer::currentSeconds();
    callBFSKernel(blocksPerGrid, threadsPerBlock, old_boards, new_boards, total_boards, board_index, empty_spaces, empty_space_count);
    double endGPUTimeBFS = CycleTimer::currentSeconds();
    
    total_time += (endGPUTimeBFS - startGPUTime);
    
    int host_count;
    
    int iterations = 18;
    
    for (int i=0; i<iterations; i++)
    {
        cudaMemcpy(&host_count, board_index, sizeof(int), cudaMemcpyDeviceToHost);
        printf("total boards after an iteration %d: %d\n", i, host_count);
        cudaMemset(board_index, 0, sizeof(int));
        
        if((i % 2) == 0)
        {
            double startTime1_1 = CycleTimer::currentSeconds();
            callBFSKernel(blocksPerGrid, threadsPerBlock, new_boards, old_boards, host_count, board_index, empty_spaces, empty_space_count);
            double endTime1_1 = CycleTimer::currentSeconds();
            total_time += (endTime1_1 - startTime1_1);
        }
        else
        {
            double startTime1_2 = CycleTimer::currentSeconds();
            callBFSKernel(blocksPerGrid, threadsPerBlock, old_boards, new_boards, host_count, board_index, empty_spaces, empty_space_count);
            double endTime1_2 = CycleTimer::currentSeconds();
            total_time += (endTime1_2 - startTime1_2);
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
    
     double startGPUTime1 = CycleTimer::currentSeconds();
    if ((iterations % 2) == 1) {
        double startTime1_3 = CycleTimer::currentSeconds();
    cuda_sudokuBacktrack(blocksPerGrid, threadsPerBlock, old_boards, host_count, empty_spaces, empty_space_count, dev_finished, dev_solved);
    cudaDeviceSynchronize();
        double endTime1_3 = CycleTimer::currentSeconds();
        total_time += (endTime1_3 - startTime1_3);
    }
    else {
        double startTime1_4 = CycleTimer::currentSeconds();
        cuda_sudokuBacktrack(blocksPerGrid, threadsPerBlock, new_boards, host_count, empty_spaces, empty_space_count, dev_finished, dev_solved);
        cudaDeviceSynchronize();
        double endTime1_4 = CycleTimer::currentSeconds();
        total_time += (endTime1_4 - startTime1_4);
    }
    double endGPUTime = CycleTimer::currentSeconds();
    double timeKernel = endGPUTime - startGPUTime;
    
    *exec_time = total_time;
    
    cudaMemcpy(finished, dev_finished, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Execution time: %lfs\n", timeKernel);
    printf("Backtracking kernel time: %lfs\n", endGPUTime - startGPUTime1);
    printf("BFS kernel time: %lfs\n", endGPUTimeBFS-startGPUTime);
    
    cudaMemcpy(solved, dev_solved, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    double endGPUTime2 = CycleTimer::currentSeconds();
    
    printf("Memcpy time: %lfs\n", endGPUTime2-endGPUTime);
    
    cudaFree(empty_spaces);
    cudaFree(empty_space_count);
    cudaFree(new_boards);
    cudaFree(old_boards);
    cudaFree(board_index);
    
    cudaFree(dev_finished);
    cudaFree(dev_solved);
}
