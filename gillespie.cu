#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <time.h>
#include <math.h>

#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random/uniform_real_distribution.h>

#define DEBUG 0
#define PRINT_CONFIG 0

#define THREADS_PER_BLOCK 32

#define IDX2C(i,j,ld) (((j)*(ld))+(i)) // column-major format for matrices
#define C2IDX_1(k,ld) ((k)%(ld)) // index 1
#define C2IDX_2(k,ld) ((k)/(ld)) // index 2


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

//gillespie alogrithm for a problem with dim states, exactly n time steps are calculated
//curand implementation
void gillespie_n_curand(double *K, int *X, int dim, long max_steps, int threads_per_block);

//thrust implementation
void gillespie_n_thrust(double *K, int *X, int dim, long max_steps,  int threads_per_block);

//host helper functions
void rateMatrix(double *K, int nrows, int ncols, double kup, double kdown); //create banded matrix K
void printMat(double *A, int nrows, int ncols);
void printIntMat(int *A, int nrows, int ncols);

__global__ void transitionMatrix(double *d_K, int *d_X, double *d_A, int dim);



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


//not optimized kernels
__global__ void reaction(double *d_A, int *d_X, double *norm, double *random_number, int dim); //TODO change to parallel algorithm
__global__ void reduce(double *d_A, double *d_res, int size); //TODO possible to parallize?
__global__ void exponentialRN(double *d_R, double *d_time, double *a, int pos);


//TODO sample discrete distribution - parallel algorithm, d_A is assumed to be already normalized
__global__ void reaction_normalized(double *d_A, int *d_X, double *random_number, int dim);



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

//TODO sample discrete distribution - parallel algorithm, d_A is assumed to be already normalized
__global__ void reaction_normalized(double *d_A, int *d_X, double *random_number, int dim)
{
    
    
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    
    double scan_entry = 0;
    int random_idx = 0;
    int k = 0, l = 0;
   
    scan_entry = d_A[0];
    
    if( i < 1)
    {
        for(int j = 0; j < dim * dim; ++j)
        {
            scan_entry = scan_entry + d_A[j];
            
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


void gillespie_n_curand(double *K, int *X, int dim, long max_steps, int threads_per_block)
{
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
    
    printf("dimGrid = %d x %d\n", dimGrid.x, dimGrid.y);
    printf("dimBlocks = %d x %d\n", dimBlocks.x, dimBlocks.y);
    
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
        //search highest entry, such that d_r2 < entry  possible to parallize?
        reaction<<<1,1>>>(d_A, d_X, d_a, &d_r2[i], dim);

        //print temporary steps
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
    //double *d_K;
    thrust::device_vector<double> d_K(K, &K[dim * dim]);
    thrust::device_vector<int> d_X(X, &X[dim]);
    thrust::device_vector<double> d_A(dim*dim);
    double a = 0;
    double time = 0;
    double dt = 0;
    long n = max_steps;
    
    dim3 dimGrid(ceil(dim/(float) threads_per_block), ceil(dim/(float) threads_per_block));
    dim3 dimBlocks(threads_per_block, threads_per_block);
    
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
        
        dt = 1 / a * log(1/d_r1[i]);
        time += dt;

        //normalize A
        thrust::transform(d_A.begin(),d_A.end(),thrust::make_constant_iterator(1/a),
                  d_A.begin(),
                  thrust::multiplies<double>()); 
         
        /*std::cout << "A = [ ";
        thrust::copy(d_A.begin(), d_A.end(), std::ostream_iterator<double>(std::cout, " "));
        std::cout << " ]\n";
         */
        
        //sample discrete distribution given by normalized A
        //and change d_X accordingly
        //purely sequential algorithm O(n)
       
        
        reaction_normalized<<<1,1>>>(thrust::raw_pointer_cast(&d_A[0]), thrust::raw_pointer_cast(&d_X[0]), thrust::raw_pointer_cast(&d_r2[i]), dim);
        

        if( i % 1000 == 0)
        {
            std::cout << "time = " << time << ", step = " << i << ", X = [ ";
            thrust::copy(d_X.begin(), d_X.end(), std::ostream_iterator<double>(std::cout, " "));
            std::cout << " ]\n";
        }
        
    }
    
    std::cout << "FINISHED:\ntime = " << time << ", X = [ ";
    thrust::copy(d_X.begin(), d_X.end(), std::ostream_iterator<double>(std::cout, " "));
    std::cout << " ]\n";
    
   
    //printMat(thrust::raw_pointer_cast(&d_A[0]), dim, dim);

}

int main ( void ){
    
    
   
    
    long n =  50000;
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
   //gillespie_n_curand(K, X, dim, n, THREADS_PER_BLOCK);
    
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


void rateMatrix(double *K, int nrows, int ncols, double kup, double kdown)
{    
   
    for ( int j = 1; j < ncols; ++j)
    {
        K[IDX2C( j - 1, j, nrows)] = kup;
    }

    for ( int j = 0; j < ncols - 1; ++j)
    {
        K[IDX2C( j + 1, j, nrows)] = kdown; 
    }

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


