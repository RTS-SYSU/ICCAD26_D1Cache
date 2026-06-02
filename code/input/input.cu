#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <cmath>
#include <cuda.h>
#include <memory>

#define NPTS 1024   // 16384
#define NDIM 4

#define USE_uniform false   

__global__ void Match1(const float *__restrict d_pts1,
                       const float *__restrict d_pts2,
                       float *__restrict d_score,
                       int *__restrict d_index,
                       int NDIM2) {
    int p1 = threadIdx.x + blockDim.x * blockIdx.x;
    float max_score = 0.0f;
    int index = -1;
    #pragma unroll 1

    for (int p2 = 0; p2 < NPTS; ++p2) {
        asm volatile("// LOOP_BEGIN,1");
        float score = 0.0f;
        #pragma unroll 1
        for (int d = 0; d < NDIM; ++d) {
            asm volatile("// LOOP_BEGIN,2");
            float f1 = d_pts1[p1 * NDIM2 + d];
            __syncthreads();
            float f2 = d_pts2[p2 * NDIM + d];
            __syncthreads();
            score += f1 * f2;
        }
        if (score > max_score) {
            max_score = score;
            index = p2;
        }
        asm volatile("// LOOP_END,2");
    }
    asm volatile("// LOOP_END,1");

    d_score[p1] = max_score;
    __syncthreads();
    d_index[p1] = index;
    __syncthreads();
}

int main() {
    const int stride1 = USE_uniform ? (NDIM + 1) : NDIM;
    const size_t size1 = stride1 * NPTS;     
    const size_t size2 = NDIM * NPTS;       

    size_t total_floats = size1 + size2;
    size_t total_bytes = total_floats * sizeof(float) + 32; 
    std::vector<float> data(total_floats + 8); 
    void *ptr = data.data();

    size_t space = total_bytes;
    float *h_pts1 = (float*)std::align(32, size1 * sizeof(float), ptr, space);
    if (!h_pts1) {
        std::cerr << "Alignment failed for h_pts1" << std::endl;
        return 1;
    }

    float *h_pts2 = h_pts1 + size1;


    for (int i = 0; i < NPTS; ++i) {
        float sum1 = 0.0f, sum2 = 0.0f;
        for (int d = 0; d < NDIM; ++d) {
            h_pts1[i * stride1 + d] = (float)rand() / RAND_MAX;
            h_pts2[i * NDIM + d] = (float)rand() / RAND_MAX;
            sum1 += h_pts1[i * stride1 + d];
            sum2 += h_pts2[i * NDIM + d];
        }
        if (USE_uniform) {
            h_pts1[i * stride1 + NDIM] = 0.0f;
        }

        sum1 = sqrt(NDIM) / sum1;
        sum2 = sqrt(NDIM) / sum2;
        for (int d = 0; d < NDIM; ++d) {
            h_pts1[i * stride1 + d] *= sum1;
            h_pts2[i * NDIM + d] *= sum2;
        }
    }

    float *d_pts1, *d_pts2, *d_score;
    int *d_index;
    size_t pts1_bytes = size1 * sizeof(float);
    size_t pts2_bytes = size2 * sizeof(float);
    size_t score_bytes = sizeof(float) * NPTS;
    size_t idx_bytes = sizeof(int) * NPTS;

    cudaMalloc((void **)&d_pts1, pts1_bytes);
    cudaMalloc((void **)&d_pts2, pts2_bytes);
    cudaMalloc((void **)&d_score, score_bytes);
    cudaMalloc((void **)&d_index, idx_bytes);

    cudaMemcpy(d_pts1, h_pts1, pts1_bytes, cudaMemcpyHostToDevice);
    czudaMemcpy(d_pts2, h_pts2, pts2_bytes, cudaMemcpyHostToDevice);

    dim3 grid(1, 1, 1);
    dim3 block(256, 1, 1);

    std::vector<int>   h_index_gpu(NPTS);
    std::vector<float> h_score_gpu(NPTS);

    Match1<<<grid, block>>>(d_pts1, d_pts2, d_score, d_index, stride1);
    cudaDeviceSynchronize();

    cudaMemcpy(h_score_gpu.data(), d_score, score_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_index_gpu.data(), d_index, idx_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_pts1);
    cudaFree(d_pts2);
    cudaFree(d_score);
    cudaFree(d_index);

    return 0;
}