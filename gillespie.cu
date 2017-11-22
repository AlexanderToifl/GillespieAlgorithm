#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <time.h>
#include <math.h>


#define DEBUG 0

#define THREADS_PER_BLOCK 32

#define IDX2C(i,j,ld) (((j)*(ld))+(i)) // column-major format for matrices
#define C2IDX_1(k,ld) ((k)%(ld)) // index 1
#define C2IDX_2(k,ld) ((k)/(ld)) // index 2

// NOTE: CUBLAS uses column-major format for matrices

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


// NOTE: CUBLAS uses column-major format for matrices


void rateMatrix(double *K, int nrows, int ncols, double kup, double kdown);
__device__ __host__ void printMat(double *A, int nrows, int ncols);



//TODO sample discrete distribution - parallel algorithm
__global__ void reaction(double *d_A, int *d_X, double *norm, double *random_number, int dim)
{
    
    
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    
    double scan_entry = 0;
    int random_idx = 0;
    int k = 0, l = 0;
   
    scan_entry = d_A[0]/(*norm);
    
    
    
    if( i < 1)
    {
        for(int j = 0; j < dim * dim; ++j)
        {
            scan_entry = scan_entry + d_A[j]/(*norm);
            
            if( *random_number < scan_entry)
            {
               random_idx = j;
               
               
               //k -> l
               k = C2IDX_1(random_idx,dim);
               l = C2IDX_2(random_idx,dim);
               
               d_X[k] -= 1;
               d_X[l] += 1;
               
               break;
            }
            
        }
    } 
    
    __syncthreads();
    
    //printf("k = %d, l = %d X[k] = %d, X[l] = %d\n", k, l, d_X[k], d_X[l]);
}



//TODO change to parallel algorithm
__global__ void reduce(double *d_A, double *d_res, int size)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    
    double res = 0;
    
    if( i < 1)
    {
        for(int j = 0; j < size; ++j)
        {
            res += d_A[j];
        } 
    
    }
    
    *d_res = res;
}

__global__ void transitionMatrix(double *d_K, int *d_X, double *d_A, int dim)
{
    int j = threadIdx.x + blockDim.x * blockIdx.x;
    int i = threadIdx.y + blockDim.y * blockIdx.y;
    
    
    if( i < dim && j < dim)
    {
        d_A[IDX2C(i,j,dim)] = (double) d_X[i] * d_K[IDX2C(i,j,dim)];
    }

}

__global__ void exponentialRN(double *d_R, double *d_time, double *a, int pos)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    double old_time = 0;
    
    if( i < 1)
    {
        if(pos > 0)
            old_time = d_time[pos - 1];
        
        d_time[pos] = 1/ *a * log(1/d_R[pos]) + old_time;
        
       // printf("time = %lf\n", d_time[pos]);
       // printf("old_time = %lf\n", old_time);
       // printf("d_R[pos] = %lf\n", d_R[pos]);
    }
    
    
    

}

int main ( void ){
    double *d_r1, *d_r2;
    double *d_A, *d_time;
    double *d_a;
    
    curandGenerator_t gen;
    srand(time(NULL));
    int _seed = rand();
    
    long n = 2 << 22;
    int dim = 50; 
    
    
    double *K = (double *) calloc ( dim * dim,  sizeof(double) );
    rateMatrix(K, dim, dim, 1.2, 10);
    
    printMat(K, dim, dim);
    
    int *X = (int *) calloc ( dim ,  sizeof(int) );
    
    //initial condition
    X[ (int) ceil(10) ] = 5000;
    for(int o = 0; o < dim; ++o)
        printf("%d ", X[o]);
    
    printf("\n");
    
    //move to device
    double *d_K;
    int *d_X;
    CUDA_ERR(cudaMalloc((void **)&d_K, dim * dim * sizeof(double)));
    CUDA_ERR(cudaMalloc((void **)&d_X, dim * sizeof(int)));
    
    CUDA_ERR(cudaMemcpy(d_K, K, dim * dim * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_ERR(cudaMemcpy(d_X, X, dim * sizeof(int), cudaMemcpyHostToDevice));
    
    //now K and X are in device memory
    
    //allocate space for transitionMatrix A and sum a
    CUDA_ERR(cudaMalloc((void **)&d_A, dim * dim * sizeof(double)));
    CUDA_ERR(cudaMalloc((void **)&d_a, 1 * sizeof(double)));
    
    //allocate space for time steps - TODO: at the moment array size == size of random numbers
    CUDA_ERR(cudaMalloc((void **)&d_time, sizeof(double) * n));
    CUDA_ERR(cudaMemset( d_time, 0,  sizeof(double) * n));
    
    //allocate space for random numbers
    CUDA_ERR(cudaMalloc((void **)&d_r1, sizeof(double) * n));
    CUDA_ERR(cudaMalloc((void **)&d_r2, sizeof(double) * n));
    CURAND_ERR(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_ERR(curandSetPseudoRandomGeneratorSeed(gen, _seed));
    
    CURAND_ERR(curandGenerateUniformDouble(gen, d_r1, n)); //generate the r1
    CURAND_ERR(curandGenerateUniformDouble(gen, d_r2, n)); //generate the r2
    

    
    dim3 dimGrid(ceil(dim/(float) THREADS_PER_BLOCK), ceil(dim/(float) THREADS_PER_BLOCK));
    dim3 dimBlocks(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    
    printf("dimGrid = %d x %d\n", dimGrid.x, dimGrid.y);
    printf("dimBlocks = %d x %d\n", dimBlocks.x, dimBlocks.y);
    
    
    int i = 0; //step index
    int max_steps = 10000;
    
    
    int *h_tmp_X = (int*) malloc( dim * sizeof(int) );
    double *h_tmp_time = (double*) malloc(1 * sizeof(double) );
    for( ; i < max_steps; ++i)
    {
        
        //calculate K
        transitionMatrix<<< dimGrid, dimBlocks>>>(d_K, d_X, d_A, dim);
        //determine a
        reduce<<<1,1>>>(d_A, d_a, dim * dim);
        //time delta dt
        exponentialRN<<<1,1>>>(d_r1, d_time, d_a, i);



        //which reaction?
        //inclusive scan A/a result: [0.02, 0.1, 0.4, 0.56, 1]
        //search highest entry, such that d_r2 < entry  possible to parallize?
        reaction<<<1,1>>>(d_A, d_X, d_a, &d_r2[i], dim);
        
        
        
        if( (i % 1000) == 0)
        {
            CUDA_ERR(cudaMemcpy(h_tmp_X, d_X, dim * sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_ERR(cudaMemcpy(h_tmp_time, &d_time[i], 1 * sizeof(double), cudaMemcpyDeviceToHost));
            
            printf("time = %lf\nX = [", *h_tmp_time);
            
            for(int j = 0; j < dim; ++j)
                printf("%d ",h_tmp_X[j]);
                
            printf("]\n");
    
        }
        
     }
     
    int time_length = i + 1;
    
    printf("time_length = %d\n", i);
    
    double *h_time = (double*) malloc( time_length * sizeof(double));
     
    CUDA_ERR(cudaMemcpy(X, d_X, dim * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_ERR(cudaMemcpy(h_time, d_time, time_length * sizeof(double), cudaMemcpyDeviceToHost));
     
    printf("Finished! Results:\ntime = %lf, X = [", h_time[time_length - 2]);
    for ( int u = 0; u < dim; ++u)
        printf("%d ", X[u]);
    printf("]\n");

   
    
    
#if DEBUG    
    double *hostData = (double*) malloc( n * sizeof(double) );
    CUDA_ERR(cudaMemcpy(hostData, d_r2, n * sizeof(double), cudaMemcpyDeviceToHost));
    
    for(int j = 0; j < n; ++j)
    {
        printf("%.2f ",hostData[j]);
    
        if( (j+1) % 8 == 0)
            printf("\n");
    }
    printf("\n");
    
    free(hostData);
    
    double *A = (double*) malloc( dim * dim * sizeof(double) );
    CUDA_ERR(cudaMemcpy(A, d_A, dim * dim *sizeof(double), cudaMemcpyDeviceToHost));
    
    printMat(A, dim, dim);
    
    free(A);
    
    
#endif    
    
    
    CURAND_ERR(curandDestroyGenerator(gen));
    CUDA_ERR(cudaFree(d_r1));
    CUDA_ERR(cudaFree(d_r2));
    CUDA_ERR(cudaFree(d_time));
    CUDA_ERR(cudaFree(d_A));
    CUDA_ERR(cudaFree(d_a));
    CUDA_ERR(cudaFree(d_X));
    CUDA_ERR(cudaFree(d_K));
    
    free(h_tmp_X);
    free(h_tmp_time);
    free(h_time);
    free(X);
    free(K);
 
    //d_r1 = NULL;
 
    return EXIT_SUCCESS ;
}


void rateMatrix(double *K, int nrows, int ncols, double kup, double kdown)
{    
   
    for ( int j = 1; j < ncols; ++j)
    {
        K[IDX2C( j - 1, j, nrows)] = kup;
        printf("%d ", IDX2C( j - 1, j, nrows));
        
    }
   
    printf("\n");
    for ( int j = 0; j < ncols - 1; ++j)
    {
        K[IDX2C( j + 1, j, nrows)] = kdown; 
        printf("%d ", IDX2C(j + 1, j, nrows));
    }
    
    printf("\n");
}

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



