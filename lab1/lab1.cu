#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t _err = (call);                                                \
    if (_err != cudaSuccess) {                                                \
      fprintf(stderr, "ERROR:\nCUDA call failed at %s:%d: %s\n",              \
              __FILE__, __LINE__, cudaGetErrorString(_err));                  \
      exit(0);                                                                \
    }                                                                         \
  } while (0)

__global__ void square_vector(float* a, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = blockDim.x * gridDim.x;
  for (int i = idx; i < n; i += offset) {
    float v = a[i];
    a[i] = v * v;
  }
}

int main() {
  int n;
  if (scanf("%d", &n) != 1) {
    fprintf(stderr, "ERROR:\nНе удалось прочитать n\n");
    return 0;
  }
  if (n < 0) {
    fprintf(stderr, "ERROR:\nn должно быть неотрицательным\n");
    return 0;
  }

  float* h = nullptr;
  if (n > 0) {
    h = (float*)malloc(sizeof(float) * (size_t)n);
    if (!h) {
      fprintf(stderr, "ERROR:\nПроблема с malloc\n");
      return 0;
    }
    for (int i = 0; i < n; i++) {
      if (scanf("%f", &h[i]) != 1) {
        fprintf(stderr, "ERROR:\nНе смог прочитать элемент массива\n");
        free(h);
        return 0;
      }
    }
  }

  if (n > 0) {
    float* d = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d, sizeof(float) * (size_t)n));
    CUDA_CHECK(cudaMemcpy(d, h, sizeof(float) * (size_t)n, cudaMemcpyHostToDevice));

    int threadsPerBlock = 1024;
    int blocks = 1024;

    // cudaEvent_t start, stop;
    // CUDA_CHECK(cudaEventCreate(&start));
    // CUDA_CHECK(cudaEventCreate(&stop));

    // CUDA_CHECK(cudaEventRecord(start));
    square_vector<<<blocks, threadsPerBlock>>>(d, n);
    // CUDA_CHECK(cudaEventRecord(stop));

    // CUDA_CHECK(cudaEventSynchronize(stop));
    // CUDA_CHECK(cudaGetLastError());

    // float ms = 0.0f;
    // CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    // printf("GPU kernel time: %.6f ms\n", ms);

    // CUDA_CHECK(cudaEventDestroy(start));
    // CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaMemcpy(h, d, sizeof(float) * (size_t)n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d));
  }

  for (int i = 0; i < n; i++) {
    if (i) printf(" ");
    printf("%.9e", h[i]);
  }
  printf("\n");

  free(h);
  return 0;
}
