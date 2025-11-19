#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

static inline void print_cuda_error(const char* where, cudaError_t err) {
  if (err != cudaSuccess) {
    fprintf(stderr, "ERROR: CUDA failure at %s: %s\n", where, cudaGetErrorString(err));
  }
}

__global__ void bubble_sort(float* a, int n) {
  for (int i = 0; i < n - 1; ++i) {
    for (int j = 0; j < n - 1 - i; ++j) {
      float x = a[j], y = a[j + 1];
      if (x > y) { a[j] = y; a[j + 1] = x; }
    }
  }
}

int main() {
  int n;
  if (scanf("%d", &n) != 1) { fprintf(stderr, "ERROR: failed to read n\n"); return 0; }
  if (n < 0) { fprintf(stderr, "ERROR: n must be non-negative\n"); return 0; }
  float* h = nullptr;
  if (n > 0) {
    h = (float*)malloc(sizeof(float) * (size_t)n);
    if (!h) { fprintf(stderr, "ERROR: malloc failed\n"); return 0; }
    for (int i = 0; i < n; ++i) {
      if (scanf("%f", &h[i]) != 1) { fprintf(stderr, "ERROR: failed to read array element\n"); free(h); return 0; }
    }
  }
  float* d = nullptr;
  cudaError_t err;
  if (n > 0) {
    err = cudaMalloc((void**)&d, sizeof(float) * (size_t)n);
    if (err) { print_cuda_error("cudaMalloc", err); free(h); return 0; }
    err = cudaMemcpy(d, h, sizeof(float) * (size_t)n, cudaMemcpyHostToDevice);
    if (err) { print_cuda_error("cudaMemcpy H2D", err); cudaFree(d); free(h); return 0; }
    bubble_sort<<<1,1>>>(d, n);
    err = cudaGetLastError();
    if (err) { print_cuda_error("kernel launch", err); cudaFree(d); free(h); return 0; }
    err = cudaDeviceSynchronize();
    if (err) { print_cuda_error("cudaDeviceSynchronize", err); cudaFree(d); free(h); return 0; }
    err = cudaMemcpy(h, d, sizeof(float) * (size_t)n, cudaMemcpyDeviceToHost);
    if (err) { print_cuda_error("cudaMemcpy D2H", err); cudaFree(d); free(h); return 0; }
    cudaFree(d);
  }
  for (int i = 0; i < n; ++i) {
    if (i) printf(" ");
    printf("%.6e", h[i]);
  }
  printf("\n");
  free(h);
  return 0;
}
