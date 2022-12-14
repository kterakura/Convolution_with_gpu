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
#define NUM_KERNEL 32

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
                    C[CSIZE_Y*y+x + idx*CSIZE_X*CSIZE_Y] += A[ASIZE_X*i+j]*B[(i-y)*BSIZE_X+j-x + idx*BSIZE_X*BSIZE_Y];
                }
            }
        }
    }
    return;
}

__global__ void d_cnn(float *A, float *B, float *C){
    __shared__ float tile_A1[BLOCK_SHARED_X+2];  //Allocate shared memory of the same size as the number of threads in the block + 2
    __shared__ float tile_A2[BLOCK_SHARED_X+2];
    __shared__ float tile_A3[BLOCK_SHARED_X+2];
    __shared__ float tile_B[BSIZE_X * BSIZE_Y];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    if(tx < 9){
        tile_B[tx] = B[tx];
    }

    //下二つをshared memoryにstoreする
    tile_A1[tx] = A[(idy)*ASIZE_X + idx];
    tile_A2[tx] = A[(idy+1)*ASIZE_X + idx];
    tile_A3[tx] = A[(idy+2)*ASIZE_X + idx];

    //last two
    if(tx==BLOCK_SHARED_X-1){
        tile_A1[tx+1] = A[(idy)*ASIZE_X + idx+1];
        tile_A2[tx+1] = A[(idy+1)*ASIZE_X + idx+1];
        tile_A3[tx+1] = A[(idy+2)*ASIZE_X + idx+1];
        tile_A1[tx+2] = A[(idy)*ASIZE_X + idx+2];
        tile_A2[tx+2] = A[(idy+1)*ASIZE_X + idx+2];
        tile_A3[tx+2] = A[(idy+2)*ASIZE_X + idx+2];
    }

    __syncthreads();

    //no ends needed
    if(idx >= CSIZE_X || idy >= CSIZE_Y){
        return;
    }

    //x→x+2,y→y+2
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
    // allocate host memory
    float *h_A, *h_B, *h_C, *ans_C;
    CHECK(cudaHostAlloc((void**)&h_A, ASIZE_X * ASIZE_Y * sizeof(float), cudaHostAllocDefault));
    CHECK(cudaHostAlloc((void**)&h_B, BSIZE_X * BSIZE_Y * sizeof(float) * NUM_KERNEL, cudaHostAllocDefault));
    CHECK(cudaHostAlloc((void**)&h_C, CSIZE_X * CSIZE_Y * sizeof(float) * NUM_KERNEL, cudaHostAllocDefault));
    CHECK(cudaHostAlloc((void**)&ans_C, CSIZE_X * CSIZE_Y * sizeof(float) * NUM_KERNEL, cudaHostAllocDefault));
    initialData(h_A, ASIZE_X * ASIZE_Y);
    initialData(h_B, BSIZE_X * BSIZE_Y * NUM_KERNEL);
    memset(h_C, 0, CSIZE_X * CSIZE_Y * sizeof(float));
    memset(ans_C, 0, CSIZE_X * CSIZE_Y * sizeof(float));

    // cnn with cpu
    cnn_start = cpuSecond();
    for(int i=0; i<NUM_KERNEL; i++){
        //cpu
        h_cnn(h_A, h_B, h_C, i);
    }
    cnn_time = cpuSecond() - cnn_start;
    printf("cpu time: %f\n", cnn_time);

    // allocate global memory
    float *d_A,*d_B,*d_C;
    cudaMalloc( (void **) &d_A, ASIZE_X * ASIZE_Y * sizeof(float ));
    cudaMalloc( (void **) &d_B, BSIZE_X * BSIZE_Y * sizeof(float) * NUM_KERNEL);
    cudaMalloc( (void **) &d_C, CSIZE_X * CSIZE_Y * sizeof(float) * NUM_KERNEL);
    dim3 block(BLOCK_SHARED_X,1);
    dim3 grid((ASIZE_X + BLOCK_SHARED_X -2 -1) / ( BLOCK_SHARED_X -2 ), ASIZE_Y);
    
    //make stream
    int n_streams = NUM_KERNEL;
    cudaStream_t *streams = (cudaStream_t *)malloc(n_streams * sizeof(cudaStream_t));
    for(int i=0; i<n_streams; i++){
        cudaStreamCreate(&streams[i]);
    }
    
    //cnn with gpu
    cnn_start = cpuSecond();
    CHECK(cudaMemcpyAsync( d_A, h_A, ASIZE_X * ASIZE_Y * sizeof(float), cudaMemcpyHostToDevice ));
    int bytesPerStream_B = BSIZE_X*BSIZE_Y;
    int bytesPerStream_C = CSIZE_X*CSIZE_Y;
    for(int i=0; i<n_streams; i++){
        int offset_B = i * bytesPerStream_B;
        int offset_C = i * bytesPerStream_C;
        CHECK(cudaMemcpyAsync( &d_B[offset_B], &h_B[offset_B], BSIZE_X * BSIZE_Y * sizeof(float), cudaMemcpyHostToDevice, streams[i]));
        d_cnn<<< grid, block, (BSIZE_X * BSIZE_Y+3*(BLOCK_SHARED_X+2))*sizeof(float), streams[i] >>>(d_A, &d_B[offset_B], &d_C[offset_C]);
        CHECK(cudaMemcpyAsync(&ans_C[offset_C], &d_C[offset_C], CSIZE_X * CSIZE_Y * sizeof(float), cudaMemcpyDeviceToHost, streams[i]));
    }
    
    //ホストと同期
    for(int i=0; i<n_streams; i++){
        cudaStreamSynchronize(streams[i]);
    }

    cnn_time = cpuSecond() - cnn_start;
    printf("gpu time: %f\n", cnn_time);

    CHECK(cudaGetLastError());
    checkResult(h_C, ans_C, CSIZE_X*CSIZE_Y*NUM_KERNEL);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    CHECK(cudaFreeHost(h_A));
    CHECK(cudaFreeHost(h_B));
    CHECK(cudaFreeHost(h_C));
    CHECK(cudaFreeHost(ans_C));
    for(int i=0; i<n_streams; i++){
        cudaStreamDestroy(streams[i]);
    }
    return;
}