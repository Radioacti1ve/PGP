#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
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

__global__ void roberts_kernel(cudaTextureObject_t tex, uchar4* dst, int w, int h) {
  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  int y0 = blockIdx.y * blockDim.y + threadIdx.y;
  int offset_x = blockDim.x * gridDim.x;
  int offset_y = blockDim.y * gridDim.y;

  for (int y = y0; y < h; y += offset_y) {
    for (int x = x0; x < w; x += offset_x) {
      uchar4 p00 = tex2D<uchar4>(tex, x, y);
      uchar4 p01 = tex2D<uchar4>(tex, x+1, y);
      uchar4 p10 = tex2D<uchar4>(tex, x, y+1);
      uchar4 p11 = tex2D<uchar4>(tex, x+1, y+1);

      float y00 = 0.299f * p00.x + 0.587f * p00.y + 0.114f * p00.z;
      float y01 = 0.299f * p01.x + 0.587f * p01.y + 0.114f * p01.z;
      float y10 = 0.299f * p10.x + 0.587f * p10.y + 0.114f * p10.z;
      float y11 = 0.299f * p11.x + 0.587f * p11.y + 0.114f * p11.z;

      float gx = y11 - y00;
      float gy = y10 - y01;
      float g  = sqrtf(gx * gx + gy * gy);

      unsigned char v = (g > 255.0f) ? 255 : (unsigned char)g;
      dst[y * w + x] = make_uchar4(v, v, v, p00.w);
    }
  }
}

int main() {
  std::string in_path, out_path;
  if (!std::getline(std::cin, in_path)) {
    fprintf(stderr, "ERROR:\nНе удалось прочитать путь входного файла\n");
    return 0;
  }
  if (!std::getline(std::cin, out_path)) {
    fprintf(stderr, "ERROR:\nНе удалось прочитать путь выходного файла\n");
    return 0;
  }

  std::ifstream fin(in_path, std::ios::binary);
  if (!fin) {
    fprintf(stderr, "ERROR:\nНе удалось открыть входной файл: %s\n", in_path.c_str());
    return 0;
  }

  int w = 0, h = 0;
  if (!fin.read(reinterpret_cast<char*>(&w), sizeof(int))) {
    fprintf(stderr, "ERROR:\nНе удалось прочитать ширину\n");
    return 0;
  }
  if (!fin.read(reinterpret_cast<char*>(&h), sizeof(int))) {
    fprintf(stderr, "ERROR:\nНе удалось прочитать высоту\n");
    return 0;
  }

  size_t pixels = static_cast<size_t>(w) * static_cast<size_t>(h);
  std::vector<uchar4> host_img(pixels);
  if (!fin.read(reinterpret_cast<char*>(host_img.data()), sizeof(uchar4) * pixels)) {
    fprintf(stderr, "ERROR:\nНе удалось прочитать пиксельные данные\n");
    return 0;
  }
  fin.close();

  cudaArray* cuArr = nullptr;
  cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
  CUDA_CHECK(cudaMallocArray(&cuArr, &ch, static_cast<size_t>(w), static_cast<size_t>(h))); // выделяем память под массив
  CUDA_CHECK(cudaMemcpy2DToArray(
    cuArr, 0, 0,
    host_img.data(), static_cast<size_t>(w) * sizeof(uchar4),
    static_cast<size_t>(w) * sizeof(uchar4), static_cast<size_t>(h),
    cudaMemcpyHostToDevice)); // копируем в ранее выделнный массив

  cudaResourceDesc resDesc;
  std::memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = cuArr;

  cudaTextureDesc texDesc;
  std::memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0]   = cudaAddressModeClamp; // выход за границы → крайний пиксель
  texDesc.addressMode[1]   = cudaAddressModeClamp;
  texDesc.filterMode       = cudaFilterModePoint; // без интерполяции
  texDesc.readMode         = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;

  cudaTextureObject_t texObj = 0;
  CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));

  uchar4* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_out), sizeof(uchar4) * pixels));

  dim3 blocks(512, 512);
  dim3 threads(32, 32);

  // cudaEvent_t start, stop;
  // CUDA_CHECK(cudaEventCreate(&start));
  // CUDA_CHECK(cudaEventCreate(&stop));
  // CUDA_CHECK(cudaEventRecord(start));

  roberts_kernel<<<blocks, threads>>>(texObj, d_out, w, h);

  // CUDA_CHECK(cudaEventRecord(stop));
  // CUDA_CHECK(cudaEventSynchronize(stop));
  // float ms = 0.0f;
  // CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  // printf("GPU kernel time: %.6f ms\n", ms);
  // CUDA_CHECK(cudaEventDestroy(start));
  // CUDA_CHECK(cudaEventDestroy(stop));

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(host_img.data(), d_out, sizeof(uchar4) * pixels, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaDestroyTextureObject(texObj));
  CUDA_CHECK(cudaFreeArray(cuArr));
  CUDA_CHECK(cudaFree(d_out));

  std::ofstream fout(out_path, std::ios::binary);
  if (!fout) {
    fprintf(stderr, "ERROR:\nНе удалось открыть выходной файл: %s\n", out_path.c_str());
    return 0;
  }
  if (!fout.write(reinterpret_cast<const char*>(&w), sizeof(int)) ||
      !fout.write(reinterpret_cast<const char*>(&h), sizeof(int)) ||
      !fout.write(reinterpret_cast<const char*>(host_img.data()), sizeof(uchar4) * pixels)) {
    fprintf(stderr, "ERROR:\nНе удалось записать выходные данные\n");
    return 0;
  }
  fout.close();

  return 0;
}
