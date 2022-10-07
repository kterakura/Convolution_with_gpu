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

__global__ void d_cnn(float A[1024][1024], float B[32][3][3], float C[32][1022][1022]){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int size_x = blockDim.x * gridDim.x;
    const int size_y = blockDim.x * gridDim.x;
    const int size = blockDim.x * gridDim.x * blockDim.y * gridDim.y;
    const int pos = idx + idy*size_y;

    const long long int N = 3*3*32*1022*1022;
    for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 3);
		const int i2 = ((idx /= 3	) % 3);
		const int i3 = ((idx /= 3	) % 32);
		const int i4 = ((idx /= 32	) % 1022);
		const int i5 = ((idx /= 1022	) % 1022);

		atomicAdd(&C[i3][i4][i5], B[i3][i1][i2] * A[i1 + i4][i2 +i5]);
	}


    
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
    dim3 block(1024,1);
    dim3 grid((ASIZE_X + BLOCK_SHARED_X -2 -1) / ( BLOCK_SHARED_X -2 ), ASIZE_Y);
    
    //make stream
    // int n_streams = NUM_KERNEL;
    // cudaStream_t *streams = (cudaStream_t *)malloc(n_streams * sizeof(cudaStream_t));
    // for(int i=0; i<n_streams; i++){
    //     cudaStreamCreate(&streams[i]);
    // }
    
    //cnn with gpu
    cnn_start = cpuSecond();
    CHECK(cudaMemcpy( d_A, h_A, ASIZE_X * ASIZE_Y * sizeof(float), cudaMemcpyHostToDevice ));
    CHECK(cudaMemcpy( d_B, h_B, BSIZE_X * BSIZE_Y * NUM_KERNEL *  sizeof(float), cudaMemcpyHostToDevice));
    d_cnn<<< grid, block>>>((float (*)[1024])d_A, (float (*)[3][3])d_B, (float (*)[1022][1022])d_C);
    CHECK(cudaMemcpy(ans_C, d_C, CSIZE_X * CSIZE_Y * NUM_KERNEL * sizeof(float), cudaMemcpyDeviceToHost));
    // int bytesPerStream_B = BSIZE_X*BSIZE_Y;
    // int bytesPerStream_C = CSIZE_X*CSIZE_Y;
    // for(int i=0; i<n_streams; i++){
    //     int offset_B = i * bytesPerStream_B;
    //     int offset_C = i * bytesPerStream_C;
    //     CHECK(cudaMemcpyAsync( &d_B[offset_B], &h_B[offset_B], BSIZE_X * BSIZE_Y * sizeof(float), cudaMemcpyHostToDevice, streams[i]));
    //     d_cnn<<< grid, block, (BSIZE_X * BSIZE_Y+3*(BLOCK_SHARED_X+2))*sizeof(float), streams[i] >>>(d_A, &d_B[offset_B], &d_C[offset_C]);
    //     CHECK(cudaMemcpyAsync(&ans_C[offset_C], &d_C[offset_C], CSIZE_X * CSIZE_Y * sizeof(float), cudaMemcpyDeviceToHost, streams[i]));
    // }
    
    //ホストと同期
    // for(int i=0; i<n_streams; i++){
    //     cudaStreamSynchronize(streams[i]);
    // }

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
    // for(int i=0; i<n_streams; i++){
    //     cudaStreamDestroy(streams[i]);
    // }
    return;
}