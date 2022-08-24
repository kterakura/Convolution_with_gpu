
#include <cuda_runtime.h>
#include <stdio.h>

#include <sys/types.h>
#include <sys/timeb.h>

#define CHECK(call)                                                  \
{                                                                    \
    const cudaError_t error = call;                                  \
    if (error != cudaSuccess)                                        \
    {                                                                \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                \
        printf("code:%d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                          \
        exit(1);                                                     \
    }                                                                \
}

double cpuSecond()
{
    _timeb tp;
    _ftime(&tp);
    return ((double)tp.time + (double)tp.millitm / 1000.0);
}

void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-3;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i],
                   gpuRef[i], i);
            break;
        }
    }

    if (match) printf("Arrays match.\n\n");

    return;
}