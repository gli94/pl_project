#ifndef CUDA_BACKTRACKING_CUH
#define CUDA_BACKTRACKING_CUH

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <fstream>
#include <cstring>
#include <algorithm>

#include "cuda.h"
#include "cuda_runtime.h"

#define N 9
#define BLOCK_SIZE 3
#define UNASSINED 0

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

#endif
