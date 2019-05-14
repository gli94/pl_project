#include<stdio.h>
#include<stdint.h>
#include<stdlib.h>
#include<stdbool.h>
#include<string.h>
#include<math.h>
#include<time.h>

#define N 9
#define UNASSIGNED 0
#define MUTATE 50
#define MAX_ITER 1000000
#define CHRMSMS 20


int len_chrom[N];
uint8_t grid[N][N] = {{3, 0, 6, 5, 0, 8, 4, 0, 0},
	        		  {5, 2, 0, 0, 0, 0, 0, 0, 0},
	        		  {0, 8, 7, 0, 0, 0, 0, 3, 1},
	        		  {0, 0, 3, 0, 1, 0, 0, 8, 0},
	        		  {9, 0, 0, 8, 6, 3, 0, 0, 5},
	        		  {0, 5, 0, 0, 9, 0, 6, 0, 0},
	        		  {1, 3, 0, 0, 0, 0, 2, 5, 0},
	        		  {0, 0, 0, 0, 0, 0, 0, 7, 4},
	        		  {0, 0, 5, 2, 0, 6, 3, 0, 0}};

typedef struct{
	int fitness;
	uint8_t *gene[N];
}chromosome;

void fitness_sort(chromosome ch[])
{
	chromosome temp;
	for(int i=0; i<N; i++)
		temp.gene[i] = malloc(len_chrom[i]*sizeof(uint8_t));

	for (int i = 0 ; i < CHRMSMS; i++)
	{
		int pos = i;
	    for (int j = i+1 ; j < CHRMSMS; j++)
	    	if (ch[j].fitness < ch[pos].fitness)
	    		pos = j;
    	temp    = ch[i];
    	ch[i]   = ch[pos];
    	ch[pos] = temp;
	}
	//free(temp.gene);
}

// int get_cost(uint8_t grid[N][N])
// {
//     int cost = 0;
//     int unique_num = 0;
//     int * score_board = malloc(N * sizeof(int));
    
//     for (int row = 0; row < N; row++)
//     {
//         memset(score_board, 0, N * sizeof(int));
//         unique_num = 0;
        
//         for (int i = 0; i < N; i++)
//             score_board[grid[row][i] - 1] += 1;
        
//         for (int j = 0; j < N; j++){
//             if (score_board[j] != 0)
//                 unique_num++;
//         }
//         cost += (-1) * unique_num;
//     }
    
//     for (int col = 0; col < N; col++)
//     {
//         memset(score_board, 0, N * sizeof(int));
//         unique_num = 0;
        
//         for (int i = 0; i < N; i++)
//             score_board[grid[i][col] - 1] += 1;
        
//         for (int j = 0; j < N; j++){
//             if (score_board[j] != 0)
//                 unique_num++;
//         }
//         cost += (-1) * unique_num;
//     }
//     free(score_board);
//     return cost;
// }

int get_cost_new(uint8_t grid[N][N])
{
    int cost = 0;    
    for (int r=0; r<N; r++)
    {
        for(int c=0; c<N; c++)
        {
        	for(int k=r+1; k<N; k++)
        		if(grid[r][c]==grid[k][c])
        		{
        			cost++;
        			//printf("row match\n");
        		}

        	for(int k=c+1; k<N; k++)
        		if(grid[r][c]==grid[r][k])
        		{
        			cost++;
        			//printf("col match\n");
        		}
        }
    }
    return cost;
}

void rand_init(uint8_t *gene[N])
{
    for (int q = 0; q < N; q++){
    	uint8_t subb[N];
    	int t=0;
    	for(int i=0; i<3; i++)
    		for(int j=0; j<3; j++)
    		{
    			// printf("%d ", ((q/3)*3)+i);
    			// printf("%d ", ((q%3)*3)+j);
    			// printf("%d\n", grid[((q/3)*3)+i][((q%3)*3)+j]);
    			subb[t] = grid[((q/3)*3)+i][((q%3)*3)+j];
    			t++;
    		}

    	// for(int i=0; i<N; i++)
    	// 	printf("%d \t", subb[i]);
    	// printf("\n");

    	int k=0;
    	for(int i=0; i<N; i++)
    	{
	        if (subb[i] == UNASSIGNED)
	        {
	        	uint8_t r_no;
	        	while(1)
	        	{
	            	r_no = (rand() % N) + 1;
	            	int conflict = 0;
	            	for(int j=0; j<N; j++)
	            		if(r_no==subb[j])
	            		{
	            			conflict = 1;
	            			break;
	            		}
	            	if(conflict==0)
	            		break;
	            }
	            subb[i] = r_no;
	            gene[q][k]=r_no;
	            k++;
	        }
	    }
    }
}

void chromosome_init(chromosome ch[], int len)
{
	for(int i=0; i<len; i++)
	{
		ch[i].fitness = 100;
		for(int j=0; j<N; j++)
		{
			ch[i].gene[j] = malloc(len_chrom[j]*sizeof(uint8_t));
		}
		rand_init(ch[i].gene);
	}
}

void chromosome_burn(chromosome ch[])
{
	for(int i=0; i<CHRMSMS; i++)
	{
		for(int j=0; j<N; j++)
			free(ch[i].gene[j]);
	}
}

void chrom2grid(uint8_t ch_grid[N][N], chromosome ch)
{
	int l=0;
	for(int i=0; i<N; i+=3)
		for(int j=0; j<N; j+=3)
		{
			int k=0;
			for(int a=0; a<3; a++)
				for(int b=0; b<3; b++)
				{
					if(grid[i+a][j+b]==UNASSIGNED)
					{
						ch_grid[i+a][j+b] = ch.gene[l][k];
						k++;
					}
					else
						ch_grid[i+a][j+b] = grid[i+a][j+b];
				}
			l++;
		}
}

chromosome cross_mutate(chromosome p1, chromosome p2)
{
	chromosome child;
	child.fitness = 100;
	for(int i=0; i<N; i++)
	{
		child.gene[i] = malloc(len_chrom[i]*sizeof(uint8_t));
		int p = (rand() % 100);
		if(p<50)
			for(int j=0;j<len_chrom[i]; j++)
				child.gene[i][j] = p1.gene[i][j];
		else
			for(int j=0;j<len_chrom[i]; j++)
				child.gene[i][j] = p2.gene[i][j];
		
		p = (rand() % 100);
		if(p<MUTATE)
		{
			int r1 = rand()%len_chrom[i];
			int r2 = rand()%len_chrom[i];
			uint8_t temp = child.gene[i][r1];
			child.gene[i][r1] = child.gene[i][r2];
			child.gene[i][r2] = temp;
		}
	}
	return child;	
}

void calc_fitness(chromosome ch[])
{
	uint8_t temp_grid[N][N];
	for(int i=0; i<CHRMSMS; i++)
	{
		chrom2grid(temp_grid, ch[i]);
		ch[i].fitness = get_cost_new(temp_grid);
	}
}

int main()
{
	// uint8_t solved[N][N] = {{7,3,5,6,1,4,8,9,2},
	//         {8,4,2,9,7,3,5,6,1},
	//         {9,6,1,2,8,5,3,7,4},
	//         {2,8,6,3,4,9,1,5,7},
	//         {4,1,3,8,5,7,9,2,6},
	//         {5,7,9,1,2,6,4,3,8},
	//         {1,5,7,4,9,2,6,8,3},
	//         {6,9,4,7,3,8,2,1,5},
	//         {3,2,8,5,6,1,7,4,9}};
	// int c = get_cost_new(solved);
	// printf("cost : %d\n",c);
	// return 0;

	long t = time(0);
	srand(1556693891);
	printf("%ld\n", t);

	int k=0;
	for(int i=0; i<N; i+=3)
		for(int j=0; j<N; j+=3)
		{
			len_chrom[k]=0;
			for(int a=0; a<3; a++)
				for(int b=0; b<3; b++)
					if(grid[i+a][j+b]==0)
						len_chrom[k]+=1;
			k++;
		}
	// for(int i=0; i<N; i++)
	// 	printf("len : %d\n", len_chrom[i]);

	// return 0;

	chromosome ch[CHRMSMS];
	chromosome_init(ch, CHRMSMS);

	printf("init done!\n");
	
	// for(int i=0; i<N; i++)
	// {
	// 	for(int j=0; j<len_chrom[i]; j++)
	// 		printf("%d ",ch[0].gene[i][j]);
	// 	printf("\n");
	// }
	// printf("\n");

	// uint8_t ch_grid[N][N];
	// chrom2grid(ch_grid, ch[0]);
	// for(int i=0; i<N; i++)
	// {
	// 	for(int j=0; j<N; j++)
	// 		printf("%d ",ch_grid[i][j]);
	// 	printf("\n");
	// }

	clock_t begin = clock();

	unsigned long generation = 0;
	int exit_code;
	int prev_best_fitness = 100;
	int stuck = 0;
	while(1)
	{
		//termination clause
		if( generation>MAX_ITER )
		{
			exit_code = 1;
			break;
		}
		if( ch[0].fitness <= 0 )
		{
			exit_code = 0;
			break;
		}

		// printf("not broken\n");

		//survival of the fittest
		calc_fitness(ch);
		// if(generation%1000==0){
		// 	printf("fitness calculated\n");
		// 	for(int i=0; i<CHRMSMS; i++)
		// 		printf("%d ", ch[i].fitness);
		// 	printf("\n");
		// }
		fitness_sort(ch);
		// if(generation%1000==0){
		// 	printf("sorting done\n");
		// 	for(int i=0; i<CHRMSMS; i++)
		// 		printf("%d ", ch[i].fitness);
		// 	printf("\n");
		// }

		int best_fitness = ch[0].fitness;
		if(best_fitness <= prev_best_fitness)
			stuck++;
		else
			stuck = 0;
		if(stuck>10000)
		{
			// printf("local max encounterd resettin...\n");
			chromosome_init(ch, CHRMSMS);
			stuck=0;
		}
		
		//eliminate lower half genes
		//create new genes from the gene pool
		for(int i=CHRMSMS/2; i<CHRMSMS; i++)
		{
			int r1 = rand() % (CHRMSMS/2);
			int r2 = rand() % (CHRMSMS/2);
			ch[i] = cross_mutate(ch[r1], ch[r2]);
			//free(child.gene);
		}

		// if(generation%1000==0){
		// 	printf("still fitness for gen %ld : %d\n",generation, ch[0].fitness);
		// }

		generation++;
		prev_best_fitness = ch[0].fitness;
		// printf("generation : %ld\n", generation);
	}

	clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Execution time: %lfs\n", time_spent);

	switch(exit_code)
	{
		case 0:
			printf("solution found!\n");
			printf("generation : %ld\n", generation);
			uint8_t solution[N][N];
			chrom2grid(solution, ch[0]);
			for(int i=0; i<N; i++)
			{
				for(int j=0; j<N; j++)
					printf("%d ", solution[i][j]);
				printf("\n");
			}
			break;
		case 1:
			printf("no solution found within given max generation!\n");
			printf("best fitness : %d\n", ch[0].fitness);
			printf("worst fitness : %d\n", ch[CHRMSMS/2-1].fitness);
			chrom2grid(solution, ch[0]);
			for(int i=0; i<N; i++)
			{
				for(int j=0; j<N; j++)
					printf("%d ", solution[i][j]);
				printf("\n");
			}
			break;
	}

	chromosome_burn(ch);
	return 0;
}