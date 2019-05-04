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
        /*if ((grid[row * Num + col] == value) && col != c);
        {
            return false;
        }*/
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
        /*if ((grid[row * Num + col] == value) && row != r)
        {
            return false;
        }*/
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
            /*if ((grid[row * Num + col] == value) && (row != r) && (col != c))
            {
                return false;
            }*/
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
    /*if ((grid[row * N + col] < 1) || (grid[row * N + col] > 9)) {
        return false;
    }*/
    
    if (checkrow(grid, Num, row) && checkcol(grid, Num, col) && checkbox(grid, Num, row - row % BLOCK_SIZE, col - col % BLOCK_SIZE)/* && (grid[row * Num + col] == UNASSINED)*/)
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

/*__global__
void sudoku_backtrack(int *boards,
                     const int numBoards,
                     int *emptySpaces,
                     int *numEmptySpaces,
                     int *finished,
                     int *solved) {
    
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    
    int *currentBoard;
    int *currentEmptySpaces;
    int currentNumEmptySpaces;
    
    
    while ((*finished == 0) && (index < numBoards)) {
        
        int emptyIndex = 0;
        
        currentBoard = boards + index * 81;
        currentEmptySpaces = emptySpaces + index * 81;
        currentNumEmptySpaces = numEmptySpaces[index];
        
        while ((emptyIndex >= 0) && (emptyIndex < currentNumEmptySpaces)) {
            
            currentBoard[currentEmptySpaces[emptyIndex]]++;
            
            if (!validBoard(currentBoard, currentEmptySpaces[emptyIndex])) {
                
                // if the board is invalid and we tried all numbers here already, backtrack
                // otherwise continue (it will just try the next number in the next iteration)
                if (currentBoard[currentEmptySpaces[emptyIndex]] >= 9) {
                    currentBoard[currentEmptySpaces[emptyIndex]] = 0;
                    emptyIndex--;
                }
            }
            // if valid board, move forward in algorithm
            else {
                emptyIndex++;
            }
            
        }
        
        if (emptyIndex == currentNumEmptySpaces) {
            // solved board found
            *finished = 1;
            
            // copy board to output
            for (int i = 0; i < N * N; i++) {
                solved[i] = currentBoard[i];
            }
        }
        
        index += gridDim.x * blockDim.x;
    }
}*/


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

/*__global__
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
        //current_old_board = old_boards + index * N * N;
        
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
                    
                    //if (!isvalid(current_old_board, N, row, col, attempt))
                    //{
                    //    works = 0;
                    //}
                    
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
                    for (int r = n * (row / n); r < n; r++) {
                        for (int c = n * (col / n); c < n; c++) {
                            if (old_boards[r * N + c + N * N * index] == attempt) {
                                works = 0;
                            }
                        }
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
}*/

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
    
    double startGPUTime = CycleTimer::currentSeconds();
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
    double endGPUTime = CycleTimer::currentSeconds();
    double timeKernel = endGPUTime - startGPUTime;
    
    printf("Execution time: %lfs\n", timeKernel);
    
    //int *solved = new int[N * N];
    //memset(solved, 0, N * N * sizeof(int));
    cudaMemcpy(solved, dev_solved, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    double endGPUTime2 = CycleTimer::currentSeconds();
    
    printf("Memcpy time: %lfs\n", endGPUTime2-endGPUTime);
    //printBoard(solved);
    
    //delete[] board;
    //delete[] solved;
    
    cudaFree(empty_spaces);
    cudaFree(empty_space_count);
    cudaFree(new_boards);
    cudaFree(old_boards);
    cudaFree(board_index);
    
    cudaFree(dev_finished);
    cudaFree(dev_solved);
}
