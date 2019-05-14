#include "cuda_backtracking.cuh"
/* #include <cmath>
#include <cstdio>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#define N 9
#define BLOCK_SIZE 3
#define UNASSINED 0
*/

__device__
bool checkrow(int *grid, int N, int row, int value)
{
    for (int col = 0; col < N; col++)
    {
        if (grid[row * N + col] == value)
        {
            return false;
        }
    }
    
    return true;
}

__device__
bool checkcol(int *grid, int N, int col, int value)
{
    for (int row = 0; row < N; row++)
    {
        if (grid[row * N + col] == value)
        {
            return false;
        }
    }
    
    return true;
}

__device__
bool checkbox(int *grid, int N, int box_start_row, int box_start_col, int value)
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
bool isvalid(int *grid, int N, int row, int col, int value)
{
    if (checkrow(grid, N, row, value) && checkcol(grid, N, col, value) && checkbox(grid, N, row - row % BLOCK_SIZE, col - col % BLOCK_SIZE, value) && (grid[row * N + col] == UNASSINED))
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
    
    int row;
    int col;
    int value = 0;
    
    while((*finished == 0) && (index < num_boards))
    {
        
        int emptyIndex = 0;
        
        currentBoard = boards + index * N * N;
        currentEmptySpaces = empty_spaces + index * N * N;
        currentNumEmptySpaces = num_empty_spaces[index];
        
        
        while ((emptyIndex >= 0) && (emptyIndex < currentNumEmptySpaces))
        {
            //currentBoard[currentEmptySpaces[emptyIndex]]++;
            value++;
            
            row = currentEmptySpaces[emptyIndex] / N;
            col = currentEmptySpaces[emptyIndex] % N;
            
            if(!isvalid(currentBoard, N, row, col, value))
            {
                if(value >= 9)
                {
                    currentBoard[currentEmptySpaces[emptyIndex]] = 0;
                    emptyIndex--;
                }
            }
            else
            {
                currentBoard[currentEmptySpaces[emptyIndex]] = value;
                value = 0;
                emptyIndex++;
            }
        }
        
        if(emptyIndex == currentNumEmptySpaces)
        {
            *finished = 1;
            
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
void cudaBFSKernel (int *old_boards,
                    int *new_boards,
                    int total_boards,
                    int *board_index,
                    int *empty_spaces,
                    int *empty_space_count
                    )
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int *current_old_board;
    
    while (index < total_boards)
    {
        int found = 0;
        current_old_board = old_boards + index * N * N;
        
        for (int i = index * N * N; (i < (index * N * N + N * N) && (found == 0)); i++)
        {
            if (old_boards[i] == UNASSINED)
            {
                found = 1;
                int row = (i - index * N * N) / N;
                int col = (i - index * N * N) % N;
                
                for (int attempt = 1; attempt <= N; attempt++)
                {
                    int works = 1;
                    
                    if (!isvalid(current_old_board, N, row, col, attempt))
                    {
                        works = 0;
                    }
                    
                    if (works == 1)
                    {
                        int next_board_index = atomicAdd(board_index, 1);
                        int empty_index = 0;
                        
                        for (int r = 0; r < N; r++)
                        {
                            for (int c = 0; c < N; c++)
                            {
                                new_boards[next_board_index * N * N + r * N + c] = old_boards[index * N * N + r * N + c];
                                if (old_boards[index * N * N + r * N + c] == 0 && (r != row || c != col))
                                {
                                    empty_spaces[empty_index + next_board_index * N * N] = r * N + c;
                                    empty_index++;
                                }
                            }
                        }
                        
                        empty_space_count[next_board_index] = empty_index;
                        new_boards[next_board_index * N * N + row * N + col] = attempt;
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
    cudaBFSKernel<<<blocksPerGrid, threadsPerBlock>>>(*old_boards, *new_boards, total_boards, *board_index, *empty_spaces, *empty_space_count);
}
