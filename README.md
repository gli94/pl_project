This project exploits different implementations of sudoku solvers. Specifically, we want to evaluate and compare serial C implementations and parallel implementations. We tried 3 different algorithms: Backtracking, Simulated Annealing and Genetic Algorithms. For the Backtracking and Simulated Annealing, we tried both C implementation and CUDA implementation. For the Genetic Algorithms, we tried C implementation and OpenMP implementation.

# Compile the code

   ## 1. Backtracking: C implementation
    
   ```    
   cd ./backtracking
   gcc backtracking.c -o backtracking
   ./backtracking
   ```    
        
  This will generate statistics for a specific benchmark. Avaliable benchmarks are: ```sudoku_9x9_100_32.txt```,   ```sudoku_9x9_100_40.txt```, ```sudoku_9x9_100_48.txt```, ```sudoku_9x9_100_56.txt```, ```sudoku_9x9_100_64.txt```. To get statistics for different benchmarks, change the input file name in ```main.cc```. Note that this implementation may not give solutions for ```sudoku_9x9_100_64.txt```.
     
   To evaluate single puzzle, use the code ```backtracking_test.c```.
     
   ``` 
   gcc backtracking_test.c -o backtracking_test
   ./backtracking
   ```  
     
   ## 2. Backtracking: CUDA implementation
   
   First, set up the CUDA environment. We tried to run the code on compute node of Maverick2 at TACC. Do the following to set up the environment:
   
   ```
   module load gcc
   module load cuda
   module list
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/apps/cuda/8.0/lib64
   ```
   Execute the following commands to compile and run the code:
   
   ```
   cd ./cuda/backtracking
   make
   ./cudaBacktracking
   ```
   
   This code will evaluate a single puzzle. Available puzzles are ```sudoku_board.txt``` and ```sudoku_board_hard.txt```, where ```sudoku_board_hard.txt``` is a hard problem with only 17 fixed elements.
   
   In order to get statistics for multiple benchmarks, navigate to ```stats_collection``` folder:
   
   ```
   cd ./cuda/backtracking/stats_collection
   make
    ./cudaBacktracking
   ```
   
   Available benchmarks are: ```sudoku_9x9_100_32.txt```,   ```sudoku_9x9_100_40.txt```, ```sudoku_9x9_100_48.txt```, ```sudoku_9x9_100_56.txt```, ```sudoku_9x9_100_64.txt```. To get statistics for different benchmarks, change the input file name in ```main.cc```.
   
   ## 3. Simulated Annealing: C implementation
    
   ```    
   cd ./sim_annealing
   gcc sim_annealing.c -lm -o sim_annealing
   ./sim_annealing
   ```    
        
  This will generate statistics for a specific benchmark. Avaliable benchmarks are: ```sudoku_9x9_100_32.txt```,   ```sudoku_9x9_100_40.txt```, ```sudoku_9x9_100_48.txt```, ```sudoku_9x9_100_56.txt```, ```sudoku_9x9_100_64.txt```. To get statistics for different benchmarks, change the input file name in ```main.cc```. Note that this implementation may not give solutions for ```sudoku_9x9_100_64.txt```.
     
   To evaluate single puzzle, use the code ```sim_annealing_test.c```.
     
   ```  
   gcc sim_annealing_test.c -lm -o sim_annealing_test
   ./sim_annealing
   ```  
     
   ## 4. Simulated Annealing: CUDA implementation
   
   First, set up the CUDA environment.
   
   ```
   module load gcc
   module load cuda
   module list
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/apps/cuda/8.0/lib64
   ```
   Execute the following commands to compile and run the code:
   
   ```
   cd ./cuda/sim_annealing
   make
   ./cudaSimAnnealing
   ```
   
   This code will evaluate a single puzzle. Available puzzles are ```sudoku_board.txt``` and ```sudoku_board_hard.txt```, where ```sudoku_board_hard.txt``` is a hard problem with only 17 fixed elements.
   
   In order to get statistics for multiple benchmarks, navigate to ```stats_collection``` folder:
   
   ```
   cd ./cuda/sim_annealing/stats_collection
   make
    ./cudaSimAnnealing
   ```
   
   Available benchmarks are: ```sudoku_9x9_100_32.txt```,   ```sudoku_9x9_100_40.txt```, ```sudoku_9x9_100_48.txt```, ```sudoku_9x9_100_56.txt```, ```sudoku_9x9_100_64.txt```. To get statistics for different benchmarks, change the input file name in ```main.cc```.
  
  ## 5. Generating the benchmarks
  
    
   ```    
   cd ./sudokuGen
   gcc sudokuGen.c -o sudokuGen
   ./sudokuGen
   ``` 
   
   K ia the number of empty elements on the Sudoku board and N is the size of one side of the Sudoku board.
  
   
   
    
    
      
      
        
     
      
    

