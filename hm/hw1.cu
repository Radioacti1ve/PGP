#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

static inline void print_cuda_error(const char* where, cudaError_t err) {
  if (err != cudaSuccess) {
    fprintf(stderr, "ERROR: CUDA failure at %s: %s\n", where, cudaGetErrorString(err));
  }
}

__global__ void solve_kernel(float a, float b, float c, int* code, float* r1, float* r2) {
  const float EPS = 1e-7f;
  bool a0 = fabsf(a) < EPS;
  bool b0 = fabsf(b) < EPS;
  bool c0 = fabsf(c) < EPS;
  if (a0) {
    if (b0) {
      if (c0) { *code = 3; }
      else     { *code = 4; }
    } else {
      *r1 = -c / b; *code = 5;
    }
    return;
  }
  float D = b*b - 4.f*a*c;
  if (D > EPS) {
    float s = sqrtf(D);
    *r1 = (-b + s) / (2.f*a);
    *r2 = (-b - s) / (2.f*a);
    *code = 0;
  } else if (fabsf(D) <= EPS) {
    *r1 = (-b) / (2.f*a);
    *code = 1;
  } else {
    *code = 2;
  }
}

int main() {
  float a, b, c;
  if (scanf("%f %f %f", &a, &b, &c) != 3) {
    fprintf(stderr, "ERROR: failed to read input coefficients\n");
    return 0;
  }
  int *d_code = nullptr;
  float *d_r1 = nullptr, *d_r2 = nullptr;
  cudaError_t err;
  err = cudaMalloc((void**)&d_code, sizeof(int));            if (err) { print_cuda_error("cudaMalloc(code)", err); return 0; }
  err = cudaMalloc((void**)&d_r1, sizeof(float));            if (err) { print_cuda_error("cudaMalloc(r1)", err);   cudaFree(d_code); return 0; }
  err = cudaMalloc((void**)&d_r2, sizeof(float));            if (err) { print_cuda_error("cudaMalloc(r2)", err);   cudaFree(d_code); cudaFree(d_r1); return 0; }
  solve_kernel<<<1,1>>>(a, b, c, d_code, d_r1, d_r2);
  err = cudaGetLastError();
  if (err) { print_cuda_error("kernel launch", err); cudaFree(d_code); cudaFree(d_r1); cudaFree(d_r2); return 0; }
  err = cudaDeviceSynchronize();
  if (err) { print_cuda_error("cudaDeviceSynchronize", err); cudaFree(d_code); cudaFree(d_r1); cudaFree(d_r2); return 0; }
  int code = -1;
  float r1 = 0.f, r2 = 0.f;
  cudaMemcpy(&code, d_code, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&r1,   d_r1,   sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&r2,   d_r2,   sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_code);
  cudaFree(d_r1);
  cudaFree(d_r2);
  switch (code) {
    case 0: printf("%.6f %.6f\n", r1, r2); break;
    case 1:
    case 5: printf("%.6f\n", r1); break;
    case 2: printf("imaginary\n"); break;
    case 3: printf("any\n"); break;
    case 4:
    default: printf("incorrect\n"); break;
  }
  return 0;
}
