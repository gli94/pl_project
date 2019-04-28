#include<stdio.h>
#include<stdint.h>
#include<stdlib.h>
#include<stdbool.h>
#include<string.h>
#include<math.h>
#include<time.h>

#define N 9
#define UNASSIGNED 0
#define MUTATE 0.01
#define MAX_ITER 1000000
#define CHRMSMS 20


int len_chrom;
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
	uint8_t *gene;
}chromosome;

void fitness_sort(chromosome ch[])
{
	chromosome temp;
	temp.gene = malloc(len_chrom*sizeof(uint8_t));

	for (int i = 0 ; i < CHRMSMS - 1; i++)
	    for (int j = 0 ; j < CHRMSMS - i - 1; j++)
	    	if (ch[j].fitness > ch[j+1].fitness) /* For decreasing order use < */
	    	{
	        	temp    = ch[j];
	        	ch[j]   = ch[j+1];
	        	ch[j+1] = temp;
	    	}

	//free(temp.gene);
}

int get_cost(uint8_t grid[N][N])
{
    int cost = 0;
    int unique_num = 0;
    int * score_board = malloc(N * sizeof(int));
    
    for (int row = 0; row < N; row++)
    {
        memset(score_board, 0, N * sizeof(int));
        unique_num = 0;
        
        for (int i = 0; i < N; i++)
            score_board[grid[row][i] - 1] += 1;
        
        for (int j = 0; j < N; j++){
            if (score_board[j] != 0)
                unique_num++;
        }
        cost += (-1) * unique_num;
    }
    
    for (int col = 0; col < N; col++)
    {
        memset(score_board, 0, N * sizeof(int));
        unique_num = 0;
        
        for (int i = 0; i < N; i++)
            score_board[grid[i][col] - 1] += 1;
        
        for (int j = 0; j < N; j++){
            if (score_board[j] != 0)
                unique_num++;
        }
        cost += (-1) * unique_num;
    }
    free(score_board);
    return cost;
}

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

        	for(int i=r; i<(r + 3 - (r%3)); i++)
        		for(int j=c; j<(c + 3 - (c%3)); j++)
        		{
        			if(r==i && j==c)
        				continue;
        			else if(grid[r][c]==grid[i][j])
        			{
        				cost++;
        				//printf("sub match\n");
        			}
        		}
        }
    }
    return cost;
}

void rand_init(uint8_t *gene)
{
    for (int i = 0; i < len_chrom; i++)
        if (gene[i] == UNASSIGNED)
            gene[i] = (rand() % N) + 1;
}

void chromosome_init(chromosome ch[], int len)
{
	for(int i=0; i<len; i++)
	{
		ch[i].fitness = 100;
		ch[i].gene = malloc(len_chrom*sizeof(uint8_t));
		rand_init(ch[i].gene);
	}
}

void chromosome_burn(chromosome ch[], int len)
{
	for(int i=0; i<len; i++)
	{
		free(ch[i].gene);
	}
}

void chrom2grid(uint8_t ch_grid[N][N], chromosome ch)
{
	int k = 0;
	for(int i=0; i<N; i++)
		for(int j=0; j<N; j++)
		{
			if(grid[i][j]==UNASSIGNED)
				ch_grid[i][j] = ch.gene[k++];
			else
				ch_grid[i][j] = grid[i][j];
		}
}

void cross_mutate(chromosome p1, chromosome p2, chromosome child)
{
	for(int i=0; i<len_chrom; i++)
	{
		int p = (rand() % 100);
		if(p<45)
			child.gene[i] = p1.gene[i];
		else if(p>=45 && p<90)
			child.gene[i] = p2.gene[i];
		else
			child.gene[i] = (rand() % N) + 1;
	}
	
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

	srand(time(0));

	len_chrom = 0;
	for(int i=0; i<N; i++)
		for(int j=0; j<N; j++)
			if(grid[i][j]==0)
				len_chrom++;
	// printf("len : %d\n", len_chrom);

	chromosome ch[CHRMSMS];
	chromosome_init(ch, CHRMSMS);
	
	// for(int i=0; i<len_chrom; i++)
	// 	printf("%d ",ch[0].gene[i]);
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
		if(generation==0)
			printf("intial best fitness : %d\n", ch[0].fitness);
		// printf("fitness calculated\n");
		fitness_sort(ch);

		int best_fitness = ch[0].fitness;
		if(best_fitness >= prev_best_fitness)
			stuck++;
		else
			stuck = 0;
		if(stuck>10000)
		{
			printf("local max encounterd resettin...\n");
			chromosome_init(ch, CHRMSMS);
			stuck=0;
		}
		// printf("soring done\n");
		//eliminate lower half genes
		//create new genes from the gene pool
		for(int i=CHRMSMS/2; i<CHRMSMS; i++)
		{
			chromosome child;
			child.gene = malloc(len_chrom*sizeof(uint8_t));
			int r1 = rand() % (CHRMSMS/2);
			int r2 = rand() % (CHRMSMS/2);
			cross_mutate(ch[r1], ch[r2], child);
			ch[i] = child;
			//free(child.gene);
		}

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
			chrom2grid(solution, ch[0]);
			for(int i=0; i<N; i++)
			{
				for(int j=0; j<N; j++)
					printf("%d ", solution[i][j]);
				printf("\n");
			}
			break;
	}

	chromosome_burn(ch, CHRMSMS);
	return 0;
}