#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

int N = 9;
int SRN = 3;
int K = 50;
int mat[100][9][9];

int randomGenerator(int num)
{
    return (rand() % num + 1);
}

bool unUsedInBox(int rowStart, int colStart, int num, int iter)
{
    for(int i=0; i<SRN; i++)
    {
        for(int j=0; j<SRN; j++)
        {
            if(mat[iter][rowStart+i][colStart+j] == num)
            {
                return false;
            }
        }
    }
    return true;
}

void fillBox(int row, int col, int iter)
{
    int num;
    for(int i=0; i<SRN; i++)
    {
        for(int j=0; j<SRN; j++)
        {
            do
            {
                num = randomGenerator(N);
            }
            while(!unUsedInBox(row, col, num, iter));
            
            mat[iter][row+i][col+j] = num;
        }
    }
}

void fillDiagonal(int iter)
{
    for (int i=0; i<N; i+=SRN)
    {
        fillBox(i, i, iter);
    }
}

bool unUsedInRow(int i, int num, int iter)
{
    for(int j=0; j<N; j++)
    {
        if(mat[iter][i][j] == num)
        {
            return false;
        }
    }
    return true;
}

bool unUsedInCol(int j, int num, int iter)
{
    for(int i=0; i<N; i++)
    {
        if(mat[iter][i][j] == num)
        {
            return false;
        }
    }
    return true;
}

bool checkIfSafe(int i, int j, int num, int iter)
{
    return (unUsedInRow(i, num, iter) && unUsedInCol(j, num, iter) && unUsedInBox(i-i%SRN, j-j%SRN, num, iter));
}

bool fillRemaining(int i, int j, int iter)
{
    if((j >= N) && (i < N-1))
    {
        i = i+1;
        j = 0;
    }
    if((i >= N) && (j >= N))
    {
        return true;
    }
    if(i < SRN)
    {
        if(j < SRN)
        {
            j = SRN;
        }
    }
    else if(i < N-SRN)
    {
        if (j == (int)(i/SRN)*SRN)
        {
            j = j + SRN;
        }
    }
    else
    {
        if (j == N-SRN)
        {
            i = i+1;
            j = 0;
            if(i >= N)
            {
                return true;
            }
        }
    }
    
    for(int num = 1; num <= N; num++)
    {
        if(checkIfSafe(i, j, num, iter))
        {
            mat[iter][i][j] = num;
            if (fillRemaining(i, j+1, iter))
            {
                return true;
            }
            mat[iter][i][j] = 0;
        }
    }
    return false;
}

void removeKDigits(int iter)
{
    int count = K;
    while(count != 0)
    {
        int cellId = randomGenerator(N*N);
        int i = (cellId / N);
        int j = (cellId % N);
        if(j  != 0)
        {
            j = j-1;
        }
        
        if(mat[iter][i][j] != 0)
        {
            count--;
            mat[iter][i][j] = 0;
        }
    }
}

void printSudoku(int iter, FILE** writeFile)
{
    for(int i=0; i<N; i++)
    {
        for(int j=0; j<N; j++)
        {
            fprintf(*writeFile, "%d ", mat[iter][i][j]);
        }
        fprintf(*writeFile, "\n");
    }
}

void fillValues(int iter)
{
    fillDiagonal(iter);
    
    fillRemaining(0, SRN, iter);
    
    removeKDigits(iter);
}

int main()
{
    FILE *f = fopen("file.txt", "w");
    if (f == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }
    
    for (int iter=0; iter<100; iter++)
    {
        fillValues(iter);
        printSudoku(iter, &f);
//        printf("\n");
    }
    fclose(f);
}
