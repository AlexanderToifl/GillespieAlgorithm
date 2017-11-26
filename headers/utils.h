#include <stdio.h>

//error handling
void gpuAssert(cudaError_t code, const char *file, int line, int abort)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#define CUDA_ERR(ans) gpuAssert((ans), __FILE__, __LINE__, 1)



void curandAssert(curandStatus_t code, const char *file, int line, int abort)
{
   if (code != CURAND_STATUS_SUCCESS) 
   {
      fprintf(stderr,"curandAssert in %s %d\n", file, line);
      if (abort) exit(code);
   }
}
#define CURAND_ERR(ans) curandAssert((ans), __FILE__, __LINE__, 1) 


//matrix helper macros

#define IDX2C(i,j,ld) (((j)*(ld))+(i)) // column-major format for matrices
#define C2IDX_1(k,ld) ((k)%(ld)) // index 1
#define C2IDX_2(k,ld) ((k)/(ld)) // index 2


//matrix helper functions

void printMat(double *A, int nrows, int ncols);
void printIntMat(int *A, int nrows, int ncols);

void printMat(double *A, int nrows, int ncols)
{
    printf("[\t");
    for(int iy = 0; iy < nrows; ++iy)
    {
        for(int ix = 0; ix < ncols; ++ix)
        {
           printf("%.2f ", A[iy + ix * nrows]);
        }
        printf("\n\t");
    }
    
    printf("\n]\n");
}    

void printIntMat(int *A, int nrows, int ncols)
{
    printf("[\t");
    for(int iy = 0; iy < nrows; ++iy)
    {
        for(int ix = 0; ix < ncols; ++ix)
        {
           printf("%d ", A[iy + ix * nrows]);
        }
        printf("\n\t");
    }
    
    printf("\n]\n");
} 
