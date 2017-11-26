#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <time.h>
#include <math.h>

#include "utils.h"
#include "gillespie_kernels.h"

#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random/uniform_real_distribution.h>


#define DEBUG 0
#define PRINT_CONFIG 0

//gillespie alogrithm for a problem with dim states, exactly n time steps are calculated
//curand implementation, not optimized
void gillespie_n_curand(double *K, int *X, int dim, long max_steps, int threads_per_block);

//thrust implementation
void gillespie_n_thrust(double *K, int *X, int dim, long max_steps,  int threads_per_block);

//create banded matrix K
void rateMatrix(double *K, int nrows, int ncols, double kup, double kdown); 

//functor for uniform distribution of doubles in [0,1)
struct uniform_gen
{
    __host__ __device__  uniform_gen(double _a, double _b) : a(_a), b(_b) {;}
    __device__ double operator()(const unsigned int n) const
    {
        thrust::default_random_engine rng;
        thrust::uniform_real_distribution<double> dist(a, b);
        rng.discard(n);
        return dist(rng);
    }
    double a, b;
};



void rateMatrix(double *K, int nrows, int ncols, double kup, double kdown)
{    
    for ( int j = 1; j < ncols; ++j)
        K[IDX2C( j - 1, j, nrows)] = kup;

    for ( int j = 0; j < ncols - 1; ++j)
        K[IDX2C( j + 1, j, nrows)] = kdown; 
}


void gillespie_n_curand(double *K, int *X, int dim, long max_steps, int threads_per_block)
{
    const long PRINT_STEPS = 10000;
    curandGenerator_t gen;
    srand(time(NULL));
    int _seed = rand();
    long n = max_steps;

    //move K and X to device memory
    double *d_K;
    int *d_X;
    
    CUDA_ERR(cudaMalloc((void **)&d_K, dim * dim * sizeof(double)));
    CUDA_ERR(cudaMalloc((void **)&d_X, dim * sizeof(int)));
    
    CUDA_ERR(cudaMemcpy(d_K, K, dim * dim * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_ERR(cudaMemcpy(d_X, X, dim * sizeof(int), cudaMemcpyHostToDevice));
    
    //allocate space for transitionMatrix A and sum a
    double *d_A;
    double *d_a;
    
    CUDA_ERR(cudaMalloc((void **)&d_A, dim * dim * sizeof(double)));
    CUDA_ERR(cudaMalloc((void **)&d_a, 1 * sizeof(double)));
    
    //allocate space for time steps - TODO: at the moment array size == size of random numbers
    double *d_time;
    CUDA_ERR(cudaMalloc((void **)&d_time, sizeof(double) * n));
    CUDA_ERR(cudaMemset( d_time, 0,  sizeof(double) * n));
    
    //allocate space for random numbers
    double *d_r1, *d_r2;
    CUDA_ERR(cudaMalloc((void **)&d_r1, sizeof(double) * n));
    CUDA_ERR(cudaMalloc((void **)&d_r2, sizeof(double) * n));
    CURAND_ERR(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_ERR(curandSetPseudoRandomGeneratorSeed(gen, _seed));
    
    CURAND_ERR(curandGenerateUniformDouble(gen, d_r1, n)); //generate the r1
    CURAND_ERR(curandGenerateUniformDouble(gen, d_r2, n)); //generate the r2

    dim3 dimGrid(ceil(dim/(float) threads_per_block), ceil(dim/(float) threads_per_block));
    dim3 dimBlocks(threads_per_block, threads_per_block);
    
    //printf("dimGrid = %d x %d\n", dimGrid.x, dimGrid.y);
    //printf("dimBlocks = %d x %d\n", dimBlocks.x, dimBlocks.y);
    
    long i = 0; //step index
    
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
        //search highest entry, such that d_r2 < entry
        reaction<<<1,1>>>(d_A, d_X, d_a, &d_r2[i], dim);
        //print temporary steps
        if( (i % PRINT_STEPS) == 0)
        {
            CUDA_ERR(cudaMemcpy(h_tmp_X, d_X, dim * sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_ERR(cudaMemcpy(h_tmp_time, &d_time[i], 1 * sizeof(double), cudaMemcpyDeviceToHost));
            
            printf("time = %lf\nX = [", *h_tmp_time);
            
            for(int j = 0; j < dim; ++j)
                printf("%d ",h_tmp_X[j]);
                
            printf("]\n");
        }
     }
     
    long time_length = max_steps;
    
    double *h_time = (double*) malloc( time_length * sizeof(double));
     
    CUDA_ERR(cudaMemcpy(X, d_X, dim * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_ERR(cudaMemcpy(h_time, d_time, time_length * sizeof(double), cudaMemcpyDeviceToHost));
     
    printf("Finished! Results:\ntime = %lf, X = [", h_time[time_length - 2]);
    for ( int u = 0; u < dim; ++u)
        printf("%d ", X[u]);
    printf("]\n");


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
}

void gillespie_n_thrust(double *K, int *X, int dim, long max_steps, int threads_per_block)
{
    const long PRINT_STEPS = 10000;
    thrust::device_vector<double> d_K(K, &K[dim * dim]);
    thrust::device_vector<int> d_X(X, &X[dim]);
    thrust::device_vector<double> d_A(dim*dim);
    double a = 0;
    double time = 0;
    double dt = 0;
    long n = max_steps;
    
    dim3 dimGrid(ceil(dim/(float) threads_per_block), ceil(dim/(float) threads_per_block));
    dim3 dimBlocks(threads_per_block, threads_per_block);
    
    dim3 nThreads(256);
    
    thrust::device_vector<double> d_r1(n);
    thrust::device_vector<double> d_r2(n);
    thrust::counting_iterator<unsigned int> index_sequence_begin1(rand());
    thrust::transform(thrust::device, index_sequence_begin1, index_sequence_begin1 + n, d_r1.begin(),  uniform_gen(0.0,1.0));
    

    thrust::counting_iterator<unsigned int> index_sequence_begin2(rand());
    thrust::transform(thrust::device, index_sequence_begin2, index_sequence_begin2 + n, d_r2.begin(),  uniform_gen(0.0,1.0));
 

    
    for(long i = 0 ; i < max_steps; ++i)
    {
        transitionMatrix<<< dimGrid, dimBlocks>>>(thrust::raw_pointer_cast(&d_K[0]), thrust::raw_pointer_cast(&d_X[0]), thrust::raw_pointer_cast(&d_A[0]), dim);
         
        a  = thrust::reduce(thrust::device, d_A.begin(),d_A.end());
        
        dt = 1 / a * log(1/d_r1[i]); //calculation is executed on the CPU
        time += dt;

        //normalize A -> A/a
        thrust::transform(d_A.begin(),d_A.end(),thrust::make_constant_iterator(1/a),
                  d_A.begin(),
                  thrust::multiplies<double>()); 
         
        //scan A/a
        thrust::inclusive_scan(thrust::device, d_A.begin(),d_A.end(), d_A.begin());
        
        reaction_scanned<<<1, nThreads>>>(thrust::raw_pointer_cast(&d_A[0]), thrust::raw_pointer_cast(&d_X[0]), thrust::raw_pointer_cast(&d_r2[i]), dim);
        
        //print time and X periodically
        if( i % PRINT_STEPS == 0)
        {
            std::cout << "time = " << time << ", step = " << i << ", X = [ ";
            thrust::copy(d_X.begin(), d_X.end(), std::ostream_iterator<double>(std::cout, " "));
            std::cout << " ]\n";
        }
        
    }
    
    std::cout << "FINISHED:\ntime = " << time << ", X = [ ";
    thrust::copy(d_X.begin(), d_X.end(), std::ostream_iterator<double>(std::cout, " "));
    std::cout << " ]\n";
    
}

int main ( void ){
    long n =  100000;
    int dim = 100; 
    
    
    double *K = (double *) calloc ( dim * dim,  sizeof(double) );
    rateMatrix(K, dim, dim, 1.2, 10);

#if PRINT_CONFIG    
    printMat(K, dim, dim);
#endif    
    int *X = (int *) calloc ( dim ,  sizeof(int) );
    
    //initial condition
    X[ (int) ceil(dim / 10) ] = 5000;
#if PRINT_CONFIG      
    printIntMat(X, 1, dim);
#endif    
    gillespie_n_curand(K, X, dim, n, THREADS_PER_BLOCK);
    gillespie_n_thrust(K, X, dim, n,THREADS_PER_BLOCK);
    

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
    
 
    free(X);
    free(K);
 
    return EXIT_SUCCESS ;
}



