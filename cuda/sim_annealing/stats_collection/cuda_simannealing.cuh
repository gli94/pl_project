#ifndef CUDA_SIMANNEALING_CUH
#define CUDA_SIMANNEALING_CUH


void callSAKernel (const int blocksPerGrid,
                   const int threadsPerBlock,
                   int *grid,
                   const int num_boards,
                   int *candidate,
                   int *initial_grid,
                   int *finished,
                   int *solved);

void cuda_SimAnnealing(int * board, int * solved, double * exec_time);

#endif
