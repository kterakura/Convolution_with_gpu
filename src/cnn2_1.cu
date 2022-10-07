#include <iostream>
#include <random>
#include <iomanip>
#include "common.h"

#define ASIZE_X 1024
#define ASIZE_Y 1024
#define BSIZE_X 3
#define BSIZE_Y 3
#define CSIZE_X (ASIZE_X-BSIZE_X +1)
#define CSIZE_Y (ASIZE_Y-BSIZE_Y +1)
#define BLOCK_SHARED_X 128
#define NUM_KERNEL 5

using std::cout;
using std::endl;

void initialData(float *in,  const int size)
{
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<float> distr(0, 1);

    for (int i = 0; i < size; i++)
    {
        in[i] = (float)( distr(eng) );
    }

    return;
}

void h_cnn(float *A, float *B, float *C, int idx){
    for(int y=0; y<CSIZE_Y; ++y){
        for(int x=0; x<CSIZE_X; ++x){
            for(int i=y; i<y+BSIZE_Y; ++i){
                for(int j=x; j<x+BSIZE_X; ++j){
                    C[CSIZE_Y*y+x + idx*CSIZE_X*CSIZE_Y] += A[ASIZE_X*i+j]*B[(i-y)*BSIZE_X+j-x];
                    // printf("A = %f  idx = %d, B = %f  idx = %d, C = %f  idx = %d\n",  A[ASIZE_X*i+j],ASIZE_X*i+j, B[(i-y)*BSIZE_X+j-x + idx*BSIZE_X*BSIZE_Y], (i-y)*BSIZE_X+j-x + idx*BSIZE_X*BSIZE_Y, C[CSIZE_Y*y+x + idx*CSIZE_X*CSIZE_Y], CSIZE_Y*y+x + idx*CSIZE_X*CSIZE_Y);
                }
            }
        }
    }
    return;
}


__global__ void d_cnn(float *A, float *B, float *C, int i){
    __shared__ float tile_A1[BLOCK_SHARED_X+2];
    __shared__ float tile_A2[BLOCK_SHARED_X+2];
    __shared__ float tile_A3[BLOCK_SHARED_X+2];
    __shared__ float tile_B[BSIZE_X * BSIZE_Y];
    tile_B[0] = B[0];
    tile_B[1] = B[1];
    tile_B[2] = B[2];
    tile_B[3] = B[3];
    tile_B[4] = B[4];
    tile_B[5] = B[5];
    tile_B[6] = B[6];
    tile_B[7] = B[7];
    tile_B[8] = B[8];
    


    // int idx = blockIdx.x * blockDim.x + threadIdx.x - blockIdx.x*2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    // if(idx<1 || idy < 1 || idx >= ASIZE_X-1 || idy >= ASIZE_Y-1){
    //     return;
    // }

    tile_A1[tx] = A[(idy)*ASIZE_X + idx];
    tile_A2[tx] = A[(idy+1)*ASIZE_X + idx];
    tile_A3[tx] = A[(idy+2)*ASIZE_X + idx];
    if(tx==BLOCK_SHARED_X-1){
        tile_A1[tx+1] = A[(idy)*ASIZE_X + idx+1];
        tile_A2[tx+1] = A[(idy+1)*ASIZE_X + idx+1];
        tile_A3[tx+1] = A[(idy+2)*ASIZE_X + idx+1];
        tile_A1[tx+2] = A[(idy)*ASIZE_X + idx+2];
        tile_A2[tx+2] = A[(idy+1)*ASIZE_X + idx+2];
        tile_A3[tx+2] = A[(idy+2)*ASIZE_X + idx+2];
    }

    __syncthreads();

    if(idx >= CSIZE_X || idy >= CSIZE_Y){
        return;
    }

    C[(idy)*CSIZE_X+(idx)] += tile_A1[tx]*tile_B[0];
    C[(idy)*CSIZE_X+(idx)] += tile_A1[tx+1]*tile_B[1];
    C[(idy)*CSIZE_X+(idx)] += tile_A1[tx+2]*tile_B[2];
    C[(idy)*CSIZE_X+(idx)] += tile_A2[tx]*tile_B[3];
    C[(idy)*CSIZE_X+(idx)] += tile_A2[tx+1]*tile_B[4];
    C[(idy)*CSIZE_X+(idx)] += tile_A2[tx+2]*tile_B[5];
    C[(idy)*CSIZE_X+(idx)] += tile_A3[tx]*tile_B[6];
    C[(idy)*CSIZE_X+(idx)] += tile_A3[tx+1]*tile_B[7];
    C[(idy)*CSIZE_X+(idx)] += tile_A3[tx+2]*tile_B[8];
}

__global__ void d_cnn1(float *A, float *B, float *C, int i){
    using clock_value_t = long long;
    clock_value_t start = clock64();
    clock_value_t cycles_elapsed;
    do { cycles_elapsed = clock64() - start; } 
    while (cycles_elapsed < 10000);
    __shared__ float tile_A1[BLOCK_SHARED_X+2];
    __shared__ float tile_A2[BLOCK_SHARED_X+2];
    __shared__ float tile_A3[BLOCK_SHARED_X+2];
    __shared__ float tile_B[BSIZE_X * BSIZE_Y];
    tile_B[0] = B[0];
    tile_B[1] = B[1];
    tile_B[2] = B[2];
    tile_B[3] = B[3];
    tile_B[4] = B[4];
    tile_B[5] = B[5];
    tile_B[6] = B[6];
    tile_B[7] = B[7];
    tile_B[8] = B[8];
    


    // int idx = blockIdx.x * blockDim.x + threadIdx.x - blockIdx.x*2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    // if(idx<1 || idy < 1 || idx >= ASIZE_X-1 || idy >= ASIZE_Y-1){
    //     return;
    // }

    tile_A1[tx] = A[(idy)*ASIZE_X + idx];
    tile_A2[tx] = A[(idy+1)*ASIZE_X + idx];
    tile_A3[tx] = A[(idy+2)*ASIZE_X + idx];
    if(tx==BLOCK_SHARED_X-1){
        tile_A1[tx+1] = A[(idy)*ASIZE_X + idx+1];
        tile_A2[tx+1] = A[(idy+1)*ASIZE_X + idx+1];
        tile_A3[tx+1] = A[(idy+2)*ASIZE_X + idx+1];
        tile_A1[tx+2] = A[(idy)*ASIZE_X + idx+2];
        tile_A2[tx+2] = A[(idy+1)*ASIZE_X + idx+2];
        tile_A3[tx+2] = A[(idy+2)*ASIZE_X + idx+2];
    }

    __syncthreads();

    if(idx >= CSIZE_X || idy >= CSIZE_Y){
        return;
    }

    C[(idy)*CSIZE_X+(idx)] += tile_A1[tx]*tile_B[0];
    C[(idy)*CSIZE_X+(idx)] += tile_A1[tx+1]*tile_B[1];
    C[(idy)*CSIZE_X+(idx)] += tile_A1[tx+2]*tile_B[2];
    C[(idy)*CSIZE_X+(idx)] += tile_A2[tx]*tile_B[3];
    C[(idy)*CSIZE_X+(idx)] += tile_A2[tx+1]*tile_B[4];
    C[(idy)*CSIZE_X+(idx)] += tile_A2[tx+2]*tile_B[5];
    C[(idy)*CSIZE_X+(idx)] += tile_A3[tx]*tile_B[6];
    C[(idy)*CSIZE_X+(idx)] += tile_A3[tx+1]*tile_B[7];
    C[(idy)*CSIZE_X+(idx)] += tile_A3[tx+2]*tile_B[8];
}

int main(){
    double cnn_start, cnn_time;
    int n_streams = NUM_KERNEL;
    cudaStream_t *streams = (cudaStream_t *)malloc(n_streams * sizeof(cudaStream_t));
    for(int i=0; i<n_streams; i++){
        cudaStreamCreate(&streams[i]);
    }
    dim3 block(BLOCK_SHARED_X,1);
    dim3 grid((ASIZE_X + BLOCK_SHARED_X -2 -1) / ( BLOCK_SHARED_X -2 ), ASIZE_Y);
    // size_t nBytes = ASIZE_X * ASIZE_Y * sizeof(float);
    // allocate host memory
    float *h_A = (float *)malloc(ASIZE_X * ASIZE_Y * sizeof(float));
    float **h_B = (float **)malloc(BSIZE_X * BSIZE_Y * sizeof(float));
    float *h_C = (float *)malloc(CSIZE_X * CSIZE_Y * sizeof(float) * NUM_KERNEL);
    float **ans_C = (float **)malloc(sizeof(float) * NUM_KERNEL);
    

    /* allocate space for device copies of a */
    float *d_A,*d_B1,*d_B2,*d_B3,*d_B4,*d_B5,*d_C1,*d_C2,*d_C3,*d_C4,*d_C5;
    cudaMalloc( (void **) &d_A, ASIZE_X * ASIZE_Y * sizeof(float ));
    cudaMalloc( (void **) &d_B1, BSIZE_X * BSIZE_Y * sizeof(float));
    cudaMalloc( (void **) &d_B2, BSIZE_X * BSIZE_Y * sizeof(float));
    cudaMalloc( (void **) &d_B3, BSIZE_X * BSIZE_Y * sizeof(float));
    cudaMalloc( (void **) &d_B4, BSIZE_X * BSIZE_Y * sizeof(float));
    cudaMalloc( (void **) &d_B5, BSIZE_X * BSIZE_Y * sizeof(float));
    cudaMalloc( (void **) &d_C1, CSIZE_X * CSIZE_Y * sizeof(float));
    cudaMalloc( (void **) &d_C2, CSIZE_X * CSIZE_Y * sizeof(float));
    cudaMalloc( (void **) &d_C3, CSIZE_X * CSIZE_Y * sizeof(float));
    cudaMalloc( (void **) &d_C4, CSIZE_X * CSIZE_Y * sizeof(float));
    cudaMalloc( (void **) &d_C5, CSIZE_X * CSIZE_Y * sizeof(float));


    initialData(h_A, ASIZE_X * ASIZE_Y);

    for(int i=0; i<NUM_KERNEL; i++){
        h_B[i] = (float *)malloc(BSIZE_X * BSIZE_Y * sizeof(float));
        initialData(h_B[i], BSIZE_X * BSIZE_Y);
    }
    
    
    cnn_start = cpuSecond();
    for(int i=0; i<NUM_KERNEL; i++){
        //cpu
        h_cnn(h_A, h_B[i], h_C, i);
    }
    cnn_time = cpuSecond() - cnn_start;
    printf("cpu time: %f\n", cnn_time);

    for(int i=0; i<NUM_KERNEL; i++){
        ans_C[i] = (float *)malloc(CSIZE_X * CSIZE_Y * sizeof(float));
    }

    cnn_start = cpuSecond();
    CHECK(cudaMemcpy( d_A, h_A, ASIZE_X * ASIZE_Y * sizeof(float), cudaMemcpyHostToDevice ));

    CHECK(cudaMemcpyAsync( d_B1, h_B[0], BSIZE_X * BSIZE_Y * sizeof(float), cudaMemcpyHostToDevice, streams[0]));
    CHECK(cudaMemcpyAsync( d_B2, h_B[1], BSIZE_X * BSIZE_Y * sizeof(float), cudaMemcpyHostToDevice, streams[1]));
    CHECK(cudaMemcpyAsync( d_B3, h_B[2], BSIZE_X * BSIZE_Y * sizeof(float), cudaMemcpyHostToDevice, streams[2]));
    CHECK(cudaMemcpyAsync( d_B4, h_B[3], BSIZE_X * BSIZE_Y * sizeof(float), cudaMemcpyHostToDevice, streams[3]));
    CHECK(cudaMemcpyAsync( d_B5, h_B[4], BSIZE_X * BSIZE_Y * sizeof(float), cudaMemcpyHostToDevice, streams[4]));
    
    d_cnn1<<< grid, block, (BSIZE_X * BSIZE_Y+3*(BLOCK_SHARED_X+2))*sizeof(float), streams[0] >>>(d_A, d_B1, d_C1, 0);
    d_cnn<<< grid, block, (BSIZE_X * BSIZE_Y+3*(BLOCK_SHARED_X+2))*sizeof(float), streams[1] >>>(d_A, d_B2, d_C2, 1);
    d_cnn<<< grid, block, (BSIZE_X * BSIZE_Y+3*(BLOCK_SHARED_X+2))*sizeof(float), streams[2] >>>(d_A, d_B3, d_C3, 2);
    d_cnn<<< grid, block, (BSIZE_X * BSIZE_Y+3*(BLOCK_SHARED_X+2))*sizeof(float), streams[3] >>>(d_A, d_B4, d_C4, 3);
    d_cnn<<< grid, block, (BSIZE_X * BSIZE_Y+3*(BLOCK_SHARED_X+2))*sizeof(float), streams[4] >>>(d_A, d_B5, d_C5, 4);
    CHECK(cudaMemcpyAsync(ans_C[0], d_C1, CSIZE_X * CSIZE_Y * sizeof(float), cudaMemcpyDeviceToHost, streams[0]));
    CHECK(cudaMemcpyAsync(ans_C[1], d_C2, CSIZE_X * CSIZE_Y * sizeof(float), cudaMemcpyDeviceToHost, streams[1]));
    CHECK(cudaMemcpyAsync(ans_C[2], d_C3, CSIZE_X * CSIZE_Y * sizeof(float), cudaMemcpyDeviceToHost, streams[2]));
    CHECK(cudaMemcpyAsync(ans_C[3], d_C4, CSIZE_X * CSIZE_Y * sizeof(float), cudaMemcpyDeviceToHost, streams[3]));
    CHECK(cudaMemcpyAsync(ans_C[4], d_C5, CSIZE_X * CSIZE_Y * sizeof(float), cudaMemcpyDeviceToHost, streams[4]));

    cnn_time = cpuSecond() - cnn_start;
    printf("gpu time: %f\n", cnn_time);

    for(int i=0; i<n_streams; i++){
        cudaStreamDestroy(streams[i]);
    }

    // for(int i=0; i<CSIZE_X*CSIZE_Y*NUM_KERNEL; ++i){
    //     printf("%d =  %f, %f \n", i, h_C[i], ans_C[i]);
    // }
    // checkResult(h_C, ans_C, CSIZE_X*CSIZE_Y*NUM_KERNEL);
    cudaFree(d_A);
    cudaFree(d_B1);
    cudaFree(d_C1);
    cudaFree(d_B2);
    cudaFree(d_C2);
    cudaFree(d_B3);
    cudaFree(d_C3);
    cudaFree(d_B4);
    cudaFree(d_C4);
    cudaFree(d_B5);
    cudaFree(d_C5);
    free(h_A);
    free(h_B);
    free(h_C);
    free(ans_C);
    return;
}


/*
    h_A[0]=0;h_A[1]=1;h_A[2]=2;h_A[3]=3;h_A[4]=1;h_A[5]=3;h_A[6]=4;h_A[7]=0;
    h_A[8]=3;h_A[9]=4;h_A[10]=5;h_A[11]=1;h_A[12]=4;h_A[13]=5;h_A[14]=7;h_A[15]=8;
    h_B[0]=0;h_B[1]=1;h_B[2]=2;h_B[3]=1;h_B[4]=3;h_B[5]=4;h_B[6]=3;h_B[7]=4;h_B[8]=5;
    for(int i=0; i<ASIZE_X-2; ++i){
        for(int j=0; j<ASIZE_X-2; ++j){
            printf("%f ", h_C[i*(ASIZE_X-2) + j]);
        }
        printf("\n");
    }
*/