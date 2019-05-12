#ifndef CUDA_BACKTRACKING_CUH
#define CUDA_BACKTRACKING_CUH

void cuda_sudokuBacktrack (const int blocksPerGrid,
                           const int threadsPerBlock,
                           int * boards,
                           const int num_boards,
                           int *empty_spaces,
                           int *num_empty_spaces,
                           int *finished,
                           int *solved);

void callBFSKernel( const int blocksPerGrid,
                   const int threadsPerBlock,
                   int *old_boards,
                   int *new_boards,
                   int total_boards,
                   int *board_index,
                   int *empty_spaces,
                   int *empty_space_count);

void cuda_Backtrack(int * board, int * solved, double *exec_time);

#endif
