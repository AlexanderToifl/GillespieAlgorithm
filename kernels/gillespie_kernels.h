#ifndef GILL_KERNEL

#define THREADS_PER_BLOCK 32
#define TILE_W 8
#define MAX_N_THREADS 512


//simple kernel that calculates transition matrix A A[i,j] = K[i,j] * X[i]
__global__ void transitionMatrix(double *d_K, int *d_X, double *d_A, int dim);
__global__ void reaction_scanned(double* d_A_scanned, int* d_X, double* d_rng,  int dim);


//not optimized kernels for pure CUDA C implementation
__global__ void reaction(double *d_A, int *d_X, double *norm, double *random_number, int dim); //TODO change to parallel algorithm
__global__ void reduce(double *d_A, double *d_res, int size); //TODO possible to parallize?
__global__ void exponentialRN(double *d_R, double *d_time, double *a, int pos);




__global__ void transitionMatrix(double *d_K, int *d_X, double *d_A, int dim)
{
    int j = threadIdx.x + blockDim.x * blockIdx.x;
    int i = threadIdx.y + blockDim.y * blockIdx.y;
    
    if( i < dim && j < dim)
    {
        d_A[IDX2C(i,j,dim)] = (double) d_X[i] * d_K[IDX2C(i,j,dim)];
    }
}

//reaction after scan
__global__ void reaction_scanned(double* d_A_scanned, int* d_X, double* d_rng,  int dim)
{
    __shared__ int indices[MAX_N_THREADS];
    
    double rn = *d_rng;
    int perthread = ceil(dim * dim / (double) blockDim.x);
    
    indices[threadIdx.x] = dim * dim;
    
    for(int j = 0; j < perthread; ++j)
    {
        if( rn < d_A_scanned[perthread * threadIdx.x + j])
        {
            indices[threadIdx.x] = perthread * threadIdx.x + j;
            break;
        }
    }
    
    __syncthreads();
    
    //now reduce indices array[0:blockDim.x] with operator min
    for(unsigned int s = blockDim.x/2; s >= 1; s /= 2)
    {
        __syncthreads();
        
        if(threadIdx.x < s)
        {         
            indices[threadIdx.x] = min(indices[threadIdx.x], indices[threadIdx.x + s]);
        }
    }
    
    __syncthreads();
    
    if(threadIdx.x == 0)
    {
       // printf("indices[0] = %d\n", indices[0]);
        int k = C2IDX_1(indices[0],dim);
        int l = C2IDX_2(indices[0],dim);
        
        d_X[k] -= 1;
        d_X[l] += 1;
    }
}


//not optimized kernels for pure CUDA C implementation

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

#endif


