#include <cstdio>
#include <cstdlib>
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

struct Pixel {
  int x, y;
};

constexpr int MAXN_CLASSES = 32;
__constant__ double3 class_means[MAXN_CLASSES];
__constant__ double inverse_cov[MAXN_CLASSES][3][3];

__global__ void classify_kernel(uchar4* img, int num_classes, int width, int height) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int idx = tid; idx < width * height; idx += stride) {
    uchar4 pixel = img[idx];
    
    double best_score = -1e18;
    int best_class = 0;

    for (int cls = 0; cls < num_classes; ++cls) {
      double diff[3] = {
        pixel.x - class_means[cls].x,
        pixel.y - class_means[cls].y,
        pixel.z - class_means[cls].z
      };

      double temp[3] = {0.0, 0.0, 0.0};
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          temp[i] += diff[j] * inverse_cov[cls][i][j];
        }
      }

      double score = 0.0;
      for (int i = 0; i < 3; ++i) {
        score += temp[i] * diff[i];
      }
      score = -score;

      if (score > best_score) {
        best_score = score;
        best_class = cls;
      }
    }
    
    img[idx].w = best_class;
  }
}

void invert_matrix(const double src[3][3], double dst[3][3]) {
  double det = src[0][0] * (src[1][1] * src[2][2] - src[1][2] * src[2][1])
             - src[0][1] * (src[1][0] * src[2][2] - src[1][2] * src[2][0])
             + src[0][2] * (src[1][0] * src[2][1] - src[1][1] * src[2][0]);

  dst[0][0] =  (src[1][1] * src[2][2] - src[1][2] * src[2][1]) / det;
  dst[0][1] = -(src[0][1] * src[2][2] - src[0][2] * src[2][1]) / det;
  dst[0][2] =  (src[0][1] * src[1][2] - src[0][2] * src[1][1]) / det;
  
  dst[1][0] = -(src[1][0] * src[2][2] - src[1][2] * src[2][0]) / det;
  dst[1][1] =  (src[0][0] * src[2][2] - src[0][2] * src[2][0]) / det;
  dst[1][2] = -(src[0][0] * src[1][2] - src[0][2] * src[1][0]) / det;
  
  dst[2][0] =  (src[1][0] * src[2][1] - src[1][1] * src[2][0]) / det;
  dst[2][1] = -(src[0][0] * src[2][1] - src[0][1] * src[2][0]) / det;
  dst[2][2] =  (src[0][0] * src[1][1] - src[0][1] * src[1][0]) / det;
}

int main() {
  std::string input_file, output_file;
  if (!std::getline(std::cin, input_file)) {
    fprintf(stderr, "ERROR:\nНе удалось прочитать путь входного файла\n");
    return 0;
  }
  if (!std::getline(std::cin, output_file)) {
    fprintf(stderr, "ERROR:\nНе удалось прочитать путь выходного файла\n");
    return 0;
  }

  std::ifstream fin(input_file, std::ios::binary);
  if (!fin) {
    fprintf(stderr, "ERROR:\nНе удалось открыть файл: %s\n", input_file.c_str());
    return 0;
  }

  int width = 0, height = 0;
  if (!fin.read(reinterpret_cast<char*>(&width), sizeof(int))) {
    fprintf(stderr, "ERROR:\nНе удалось прочитать ширину изображения\n");
    return 0;
  }
  if (!fin.read(reinterpret_cast<char*>(&height), sizeof(int))) {
    fprintf(stderr, "ERROR:\nНе удалось прочитать высоту изображения\n");
    return 0;
  }

  size_t total_pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
  std::vector<uchar4> image_data(total_pixels);
  
  if (!fin.read(reinterpret_cast<char*>(image_data.data()), sizeof(uchar4) * total_pixels)) {
    fprintf(stderr, "ERROR:\nНе удалось прочитать данные изображения\n");
    return 0;
  }
  fin.close();

  int num_classes;
  if (!(std::cin >> num_classes)) {
    fprintf(stderr, "ERROR:\nНе удалось прочитать количество классов\n");
    return 0;
  }

  std::vector<std::vector<Pixel>> class_pixels(num_classes);
  for (int cls = 0; cls < num_classes; ++cls) {
    int num_points;
    if (!(std::cin >> num_points)) {
      fprintf(stderr, "ERROR:\nНе удалось прочитать количество точек для класса %d\n", cls);
      return 0;
    }
    
    class_pixels[cls].resize(num_points);
    for (int j = 0; j < num_points; ++j) {
      if (!(std::cin >> class_pixels[cls][j].x >> class_pixels[cls][j].y)) {
        fprintf(stderr, "ERROR:\nНе удалось прочитать координаты точки\n");
        return 0;
      }
    }
  }

  double3 means[MAXN_CLASSES];
  for (int cls = 0; cls < num_classes; ++cls) {
    double sum_r = 0.0, sum_g = 0.0, sum_b = 0.0;
    
    for (const auto& pt : class_pixels[cls]) {
      uchar4 px = image_data[pt.y * width + pt.x];
      sum_r += px.x;
      sum_g += px.y;
      sum_b += px.z;
    }
    
    int cnt = class_pixels[cls].size();
    means[cls] = make_double3(sum_r / cnt, sum_g / cnt, sum_b / cnt);
  }

  double covariance[MAXN_CLASSES][3][3];
  double inv_covariance[MAXN_CLASSES][3][3];
  
  for (int cls = 0; cls < num_classes; ++cls) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        covariance[cls][i][j] = 0.0;
      }
    }

    for (const auto& pt : class_pixels[cls]) {
      uchar4 px = image_data[pt.y * width + pt.x];
      double diff[3] = {
        px.x - means[cls].x,
        px.y - means[cls].y,
        px.z - means[cls].z
      };

      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          covariance[cls][i][j] += diff[i] * diff[j];
        }
      }
    }

    int cnt = class_pixels[cls].size();
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        covariance[cls][i][j] /= (cnt - 1);
      }
    }

    invert_matrix(covariance[cls], inv_covariance[cls]);
  }

  CUDA_CHECK(cudaMemcpyToSymbol(class_means, means, sizeof(double3) * MAXN_CLASSES));
  CUDA_CHECK(cudaMemcpyToSymbol(inverse_cov, inv_covariance, sizeof(double) * MAXN_CLASSES * 3 * 3));

  uchar4* d_image = nullptr;
  CUDA_CHECK(cudaMalloc(&d_image, sizeof(uchar4) * total_pixels));
  CUDA_CHECK(cudaMemcpy(d_image, image_data.data(), sizeof(uchar4) * total_pixels, cudaMemcpyHostToDevice));

  dim3 blocks(256);
  dim3 threads(256);

  // cudaEvent_t start, stop;
  // CUDA_CHECK(cudaEventCreate(&start));
  // CUDA_CHECK(cudaEventCreate(&stop));
  // CUDA_CHECK(cudaEventRecord(start));

  classify_kernel<<<blocks, threads>>>(d_image, num_classes, width, height);

  // CUDA_CHECK(cudaEventRecord(stop));
  // CUDA_CHECK(cudaEventSynchronize(stop));
  // float ms = 0.0f;
  // CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  // printf("GPU kernel time: %.6f ms\n", ms);
  // CUDA_CHECK(cudaEventDestroy(start));
  // CUDA_CHECK(cudaEventDestroy(stop));
  
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(image_data.data(), d_image, sizeof(uchar4) * total_pixels, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_image));

  std::ofstream fout(output_file, std::ios::binary);
  if (!fout) {
    fprintf(stderr, "ERROR:\nНе удалось открыть выходной файл: %s\n", output_file.c_str());
    return 0;
  }

  if (!fout.write(reinterpret_cast<const char*>(&width), sizeof(int)) ||
      !fout.write(reinterpret_cast<const char*>(&height), sizeof(int)) ||
      !fout.write(reinterpret_cast<const char*>(image_data.data()), sizeof(uchar4) * total_pixels)) {
    fprintf(stderr, "ERROR:\nНе удалось записать выходные данные\n");
    return 0;
  }
  fout.close();


  return 0;
}