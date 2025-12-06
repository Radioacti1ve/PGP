#include <iostream>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <cuda_runtime.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _err = (call);                                                 \
    if (_err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA ERROR at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(_err));                                       \
      exit(0);                                                                 \
    }                                                                          \
  } while (0)

const int TRIANGLE_COUNT = 50;

struct Vec3 {
  double x, y, z;

  __host__ __device__ Vec3() : x(0.0), y(0.0), z(0.0) {}
  __host__ __device__ Vec3(double xx, double yy, double zz) : x(xx), y(yy), z(zz) {}
};

__host__ __device__ Vec3 operator+(const Vec3 &a, const Vec3 &b) {
  return Vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ Vec3 operator-(const Vec3 &a, const Vec3 &b) {
  return Vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ Vec3 operator*(const Vec3 &a, double s) {
  return Vec3(a.x * s, a.y * s, a.z * s);
}

struct Triangle {
  Vec3 a, b, c;
  uchar4 color;

  __host__ __device__ Triangle() {}
  __host__ __device__ Triangle(const Vec3 &aa, const Vec3 &bb, const Vec3 &cc, uchar4 col)
      : a(aa), b(bb), c(cc), color(col) {}
};

struct Scene {
  Triangle *tris;
  int       triCount;
  Vec3      lightPos;
  uchar4    lightColor;
};

struct Camera {
  Vec3 pos;
  Vec3 lookAt;
  double fovDeg;
};

__host__ __device__ inline double dot3(const Vec3 &a, const Vec3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline Vec3 cross3(const Vec3 &a, const Vec3 &b) {
  return Vec3(
      a.y * b.z - a.z * b.y,
      a.z * b.x - a.x * b.z,
      a.x * b.y - a.y * b.x
  );
}

__host__ __device__ inline Vec3 normalize(const Vec3 &v) {
  double len = sqrt(dot3(v, v));
  return Vec3(v.x / len, v.y / len, v.z / len);
}

__host__ __device__ inline Vec3 basisMul(const Vec3 &ex, const Vec3 &ey, const Vec3 &ez,
                                         const Vec3 &v) {
  return Vec3(
      ex.x * v.x + ey.x * v.y + ez.x * v.z,
      ex.y * v.x + ey.y * v.y + ez.y * v.z,
      ex.z * v.x + ey.z * v.y + ez.z * v.z
  );
}

struct HitInfo {
  int   triIndex;
  double t;
};

__host__ __device__ bool intersectTriangle(const Vec3 &orig,
                                           const Vec3 &dir,
                                           const Triangle &tri,
                                           double &tOut) {
  const double EPS = 1e-10;
  Vec3 e1 = tri.b - tri.a;
  Vec3 e2 = tri.c - tri.a;
  Vec3 p = cross3(dir, e2);
  double det = dot3(p, e1);
  if (fabs(det) < EPS) return false;

  double invDet = 1.0 / det;
  Vec3 tvec = orig - tri.a;
  double u = dot3(p, tvec) * invDet;
  if (u < 0.0 || u > 1.0) return false;

  Vec3 q = cross3(tvec, e1);
  double v = dot3(q, dir) * invDet;
  if (v < 0.0 || u + v > 1.0) return false;

  double t = dot3(q, e2) * invDet;
  if (t < 0.0) return false;

  tOut = t;
  return true;
}

__host__ __device__ HitInfo tracePrimary(const Vec3 &orig,
                                         const Vec3 &dir,
                                         const Scene &scene) {
  HitInfo hit;
  hit.triIndex = -1;
  hit.t = 0.0;

  for (int i = 0; i < scene.triCount; ++i) {
    double tCandidate;
    if (!intersectTriangle(orig, dir, scene.tris[i], tCandidate)) continue;

    if (hit.triIndex < 0 || tCandidate < hit.t) {
      hit.triIndex = i;
      hit.t = tCandidate;
    }
  }
  return hit;
}

__host__ __device__ bool isShadowed(const Vec3 &point,
                                    const Vec3 &lightDir,
                                    double maxDist,
                                    const Scene &scene,
                                    int selfIndex) {
  for (int i = 0; i < scene.triCount; ++i) {
    if (i == selfIndex) continue;
    double tCandidate;
    if (!intersectTriangle(point, lightDir, scene.tris[i], tCandidate)) continue;
    if (tCandidate > 0.0 && tCandidate < maxDist) {
      return true;
    }
  }
  return false;
}

__host__ __device__ uchar4 shadeRay(const Vec3 &orig,
                                    const Vec3 &dir,
                                    const Scene &scene) {
  HitInfo h = tracePrimary(orig, dir, scene);
  if (h.triIndex < 0) {
    return make_uchar4(255, 255, 100, 255);
  }

  Vec3 hitPos = orig + dir * h.t;
  Vec3 toLight = scene.lightPos - hitPos;
  double dist = sqrt(dot3(toLight, toLight));
  Vec3 lightDir = normalize(toLight);

  if (isShadowed(hitPos, lightDir, dist, scene, h.triIndex)) {
    return make_uchar4(120, 120, 60, 255);
  }

  const Triangle &tri = scene.tris[h.triIndex];
  return make_uchar4(
      tri.color.x * scene.lightColor.x,
      tri.color.y * scene.lightColor.y,
      tri.color.z * scene.lightColor.z,
      255
  );
}

__host__ __device__ void buildCameraBasis(const Camera &cam,
                                          Vec3 &ex, Vec3 &ey, Vec3 &ez) {
  ez = normalize(cam.lookAt - cam.pos);
  Vec3 worldUp(0.0, 0.0, 1.0);
  ex = normalize(cross3(ez, worldUp));
  ey = normalize(cross3(ex, ez));
}

__host__ __device__ uchar4 renderPixel(int px, int py,
                                       int width, int height,
                                       int sqrtSamples,
                                       const Camera &cam,
                                       const Scene &scene) {
  Vec3 ex, ey, ez;
  buildCameraBasis(cam, ex, ey, ez);

  double aspect = static_cast<double>(height) / static_cast<double>(width);
  double z = 1.0 / tan(cam.fovDeg * M_PI / 360.0);

  uint4 acc = make_uint4(0, 0, 0, 0);
  int totalSamples = sqrtSamples * sqrtSamples;

  for (int sx = 0; sx < sqrtSamples; ++sx) {
    for (int sy = 0; sy < sqrtSamples; ++sy) {
      double u = ( (px + (sx + 0.5) / sqrtSamples) / (width  - 1.0) ) * 2.0 - 1.0;
      double v = ( (py + (sy + 0.5) / sqrtSamples) / (height - 1.0) ) * 2.0 - 1.0;
      v *= aspect;

      Vec3 viewDirCam(u, v, z);
      Vec3 worldDir = basisMul(ex, ey, ez, viewDirCam);
      worldDir = normalize(worldDir);

      uchar4 c = shadeRay(cam.pos, worldDir, scene);
      acc.x += c.x;
      acc.y += c.y;
      acc.z += c.z;
    }
  }

  return make_uchar4(acc.x / totalSamples,
                     acc.y / totalSamples,
                     acc.z / totalSamples,
                     255);
}

void renderFrameCPU(const Camera &cam,
                    int width, int height,
                    int sqrtSamples,
                    const Scene &scene,
                    uchar4 *outImage) {
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      int idx = (height - 1 - j) * width + i;
      outImage[idx] = renderPixel(i, j, width, height, sqrtSamples, cam, scene);
    }
  }
}

__global__ void renderKernel(const Camera cam,
                             int width, int height,
                             int sqrtSamples,
                             Scene scene,
                             uchar4 *outImage) {
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  if (ix >= width || iy >= height) return;

  int idx = (height - 1 - iy) * width + ix;
  outImage[idx] = renderPixel(ix, iy, width, height, sqrtSamples, cam, scene);
}

void buildFloor(const Vec3 &a, const Vec3 &b, const Vec3 &c, const Vec3 &d,
                uchar4 color, Triangle *tris, int offset) {
  tris[offset + 0] = Triangle(a, b, c, color);
  tris[offset + 1] = Triangle(a, c, d, color);
}

void buildTetrahedron(const Vec3 &center,
                      double r,
                      uchar4 color,
                      Triangle *tris,
                      int offset) {
  double edge = r * sqrt(3.0);
  double stretch = 2.0;

  Vec3 v0(center.x - edge / 2.0, center.y - r * 0.3 * stretch, center.z - edge / sqrt(12.0));
  Vec3 v1(center.x,               center.y + r * stretch,      center.z - edge / sqrt(12.0));
  Vec3 v2(center.x + edge / 2.0, center.y - r * 0.3 * stretch, center.z - edge / sqrt(12.0));
  Vec3 v3(center.x,               center.y,                    center.z + r);

  tris[offset + 0] = Triangle(v0, v1, v2, color);
  tris[offset + 1] = Triangle(v0, v1, v3, color);
  tris[offset + 2] = Triangle(v0, v2, v3, color);
  tris[offset + 3] = Triangle(v1, v2, v3, color);
}

void buildOctahedron(const Vec3 &center,
                     double r,
                     uchar4 color,
                     Triangle *tris,
                     int offset) {
  Vec3 v[6] = {
      Vec3(center.x + r, center.y,     center.z    ),
      Vec3(center.x - r, center.y,     center.z    ),
      Vec3(center.x,     center.y + r, center.z    ),
      Vec3(center.x,     center.y - r, center.z    ),
      Vec3(center.x,     center.y,     center.z + r),
      Vec3(center.x,     center.y,     center.z - r)
  };

  tris[offset + 0] = Triangle(v[0], v[2], v[4], color);
  tris[offset + 1] = Triangle(v[2], v[1], v[4], color);
  tris[offset + 2] = Triangle(v[1], v[3], v[4], color);
  tris[offset + 3] = Triangle(v[3], v[0], v[4], color);

  tris[offset + 4] = Triangle(v[2], v[0], v[5], color);
  tris[offset + 5] = Triangle(v[1], v[2], v[5], color);
  tris[offset + 6] = Triangle(v[3], v[1], v[5], color);
  tris[offset + 7] = Triangle(v[0], v[3], v[5], color);
}

void buildDodecahedron(const Vec3 &center,
                       double r,
                       uchar4 color,
                       Triangle *tris,
                       int offset) {
  double phi = (1.0 + sqrt(5.0)) / 2.0;
  double invPhi = 1.0 / phi;

  Vec3 v[20] = {
      Vec3(-invPhi, 0,       phi),
      Vec3( invPhi, 0,       phi),
      Vec3(-1,      1,       1),
      Vec3( 1,      1,       1),
      Vec3( 1,     -1,       1),
      Vec3(-1,     -1,       1),
      Vec3( 0,     -phi,     invPhi),
      Vec3( 0,      phi,     invPhi),
      Vec3(-phi,   -invPhi,  0),
      Vec3(-phi,    invPhi,  0),
      Vec3( phi,    invPhi,  0),
      Vec3( phi,   -invPhi,  0),
      Vec3( 0,     -phi,    -invPhi),
      Vec3( 0,      phi,    -invPhi),
      Vec3( 1,      1,      -1),
      Vec3( 1,     -1,      -1),
      Vec3(-1,     -1,      -1),
      Vec3(-1,      1,      -1),
      Vec3( invPhi, 0,      -phi),
      Vec3(-invPhi, 0,      -phi)
  };

  for (int i = 0; i < 20; ++i) {
    v[i].x = v[i].x * r / sqrt(3.0) + center.x;
    v[i].y = v[i].y * r / sqrt(3.0) + center.y;
    v[i].z = v[i].z * r / sqrt(3.0) + center.z;
  }

  int k = offset;
  tris[k++] = Triangle(v[4],  v[0],  v[6],  color);
  tris[k++] = Triangle(v[0],  v[5],  v[6],  color);
  tris[k++] = Triangle(v[0],  v[4],  v[1],  color);
  tris[k++] = Triangle(v[0],  v[3],  v[7],  color);
  tris[k++] = Triangle(v[2],  v[0],  v[7],  color);
  tris[k++] = Triangle(v[0],  v[1],  v[3],  color);
  tris[k++] = Triangle(v[10], v[1],  v[11], color);
  tris[k++] = Triangle(v[3],  v[1],  v[10], color);
  tris[k++] = Triangle(v[1],  v[4],  v[11], color);
  tris[k++] = Triangle(v[5],  v[0],  v[8],  color);
  tris[k++] = Triangle(v[0],  v[2],  v[9],  color);
  tris[k++] = Triangle(v[8],  v[0],  v[9],  color);
  tris[k++] = Triangle(v[5],  v[8],  v[16], color);
  tris[k++] = Triangle(v[6],  v[5],  v[12], color);
  tris[k++] = Triangle(v[12], v[5],  v[16], color);
  tris[k++] = Triangle(v[4],  v[12], v[15], color);
  tris[k++] = Triangle(v[4],  v[6],  v[12], color);
  tris[k++] = Triangle(v[11], v[4],  v[15], color);
  tris[k++] = Triangle(v[2],  v[13], v[17], color);
  tris[k++] = Triangle(v[2],  v[7],  v[13], color);
  tris[k++] = Triangle(v[9],  v[2],  v[17], color);
  tris[k++] = Triangle(v[13], v[3],  v[14], color);
  tris[k++] = Triangle(v[7],  v[3],  v[13], color);
  tris[k++] = Triangle(v[3],  v[10], v[14], color);
  tris[k++] = Triangle(v[8],  v[17], v[19], color);
  tris[k++] = Triangle(v[16], v[8],  v[19], color);
  tris[k++] = Triangle(v[8],  v[9],  v[17], color);
  tris[k++] = Triangle(v[14], v[11], v[18], color);
  tris[k++] = Triangle(v[11], v[15], v[18], color);
  tris[k++] = Triangle(v[10], v[11], v[14], color);
  tris[k++] = Triangle(v[12], v[19], v[18], color);
  tris[k++] = Triangle(v[15], v[12], v[18], color);
  tris[k++] = Triangle(v[12], v[16], v[19], color);
  tris[k++] = Triangle(v[19], v[13], v[18], color);
  tris[k++] = Triangle(v[17], v[13], v[19], color);
  tris[k++] = Triangle(v[13], v[14], v[18], color);
}

void values_for_report() {
  std::cout << "100\n";
  std::cout << "res/%d.data\n";
  std::cout << "600 600 120\n\n";

  std::cout << "10.0 3.0 0.0  4.0 1.0  2.0 6.0 1.0  0.0 0.0\n";
  std::cout << "2.0 0.0 0.0  0.5 0.1  1.0 4.0 1.0  0.0 0.0\n\n";

  std::cout << "3.0 3.0 0.5   1.0 0.2 0.2   2.0   0.0 0.0  0\n";
  std::cout << "0.0 0.0 0.7   0.2 0.9 0.2   1.75  0.0 0.0  0\n";
  std::cout << "-3.0 -3.0 0.0 0.2 0.4 1.0   1.2   0.0 0.0  0\n\n";

  std::cout << "-40.0 -40.0 -1.0  -40.0 40.0 -1.0  40.0 40.0 -1.0  40.0 -40.0 -1.0 "
               "~/floor.data  0.6 0.6 0.6  0.0\n\n";

  std::cout << "1\n";
  std::cout << "-10.0 0.0 15.0  0.9 0.9 0.9\n\n";

  std::cout << "1 4\n";
}

int main(int argc, char *argv[]) {
  if (argc >= 2 && std::strcmp(argv[1], "--default") == 0) {
    values_for_report();
    return 0;
  }

  bool use_gpu = true;
  if (argc >= 2 && std::strcmp(argv[1], "--cpu") == 0) {
    use_gpu = false;
  }

  int frames_number;
  char output_path[256];
  int w, h;
  double angle;

  double r0c, z0c, phi0c, Arc, Azc, wrc, wzc, wphic, prc, pzc;
  double r0n, z0n, phi0n, Arn, Azn, wrn, wzn, wphin, prn, pzn;

  double c1x, c1y, c1z, col1x, col1y, col1z, r1, refl1, transp1;
  int lights1;
  double c2x, c2y, c2z, col2x, col2y, col2z, r2, refl2, transp2;
  int lights2;
  double c3x, c3y, c3z, col3x, col3y, col3z, r3, refl3, transp3;
  int lights3;

  double f1x, f1y, f1z, f2x, f2y, f2z, f3x, f3y, f3z, f4x, f4y, f4z;
  char floor_tex_path[256];
  double floor_col_x, floor_col_y, floor_col_z, floor_reflect;

  int light_count;
  double light_pos_x, light_pos_y, light_pos_z;
  double light_col_x, light_col_y, light_col_z;
  double tmp_lx, tmp_ly, tmp_lz, tmp_lr, tmp_lg, tmp_lb;

  int max_depth;
  double sqrt_rays_per_pixel;

  std::cin >> frames_number;
  std::cin >> output_path;
  std::cin >> w >> h >> angle;

  std::cin >> r0c >> z0c >> phi0c >> Arc >> Azc >> wrc >> wzc >> wphic >> prc >> pzc;
  std::cin >> r0n >> z0n >> phi0n >> Arn >> Azn >> wrn >> wzn >> wphin >> prn >> pzn;

  std::cin >> c1x >> c1y >> c1z >> col1x >> col1y >> col1z >> r1 >> refl1 >> transp1 >> lights1;
  std::cin >> c2x >> c2y >> c2z >> col2x >> col2y >> col2z >> r2 >> refl2 >> transp2 >> lights2;
  std::cin >> c3x >> c3y >> c3z >> col3x >> col3y >> col3z >> r3 >> refl3 >> transp3 >> lights3;

  std::cin >> f1x >> f1y >> f1z
           >> f2x >> f2y >> f2z
           >> f3x >> f3y >> f3z
           >> f4x >> f4y >> f4z
           >> floor_tex_path
           >> floor_col_x >> floor_col_y >> floor_col_z
           >> floor_reflect;

  std::cin >> light_count;
  if (light_count > 0) {
    std::cin >> light_pos_x >> light_pos_y >> light_pos_z
             >> light_col_x >> light_col_y >> light_col_z;
    for (int i = 1; i < light_count; ++i) {
      std::cin >> tmp_lx >> tmp_ly >> tmp_lz >> tmp_lr >> tmp_lg >> tmp_lb;
    }
  } else {
    light_pos_x = -10.0;
    light_pos_y = 0.0;
    light_pos_z = 10.0;
    light_col_x = light_col_y = light_col_z = 1.0;
  }

  std::cin >> max_depth >> sqrt_rays_per_pixel;
  int sr = static_cast<int>(sqrt_rays_per_pixel);

  Triangle tris[TRIANGLE_COUNT];
  buildFloor(Vec3(f1x, f1y, f1z),
             Vec3(f2x, f2y, f2z),
             Vec3(f3x, f3y, f3z),
             Vec3(f4x, f4y, f4z),
             make_uchar4(floor_col_x * 255, floor_col_y * 255, floor_col_z * 255, 255),
             tris, 0);

  buildTetrahedron(Vec3(c1x + 2.0, c1y + 2.0, c1z + 0.8),
                   r1 * 1.7,
                   make_uchar4(col1x * 255, col1y * 255, col1z * 255, 255),
                   tris, 2);

  buildOctahedron(Vec3(c2x, c2y - 3.0, c2z + 1.2),
                  r2 * 1.5,
                  make_uchar4(col2x * 255, col2y * 255, col2z * 255, 255),
                  tris, 6);

  buildDodecahedron(Vec3(c3x - 3.0, c3y - 3.0, c3z + 1.0),
                    r3 * 1.6,
                    make_uchar4(col3x * 255, col3y * 255, col3z * 255, 255),
                    tris, 14);

  Scene hostScene;
  hostScene.tris = tris;
  hostScene.triCount = TRIANGLE_COUNT;
  hostScene.lightPos = Vec3(light_pos_x, light_pos_y, light_pos_z);
  hostScene.lightColor = make_uchar4(light_col_x * 255,
                                     light_col_y * 255,
                                     light_col_z * 255,
                                     255);

  uchar4 *imageCPU = (uchar4*)std::malloc(sizeof(uchar4) * w * h);
  uchar4 *imageGPU = nullptr;
  Triangle *devTris = nullptr;
  Scene devScene;

  if (use_gpu) {
    CUDA_CHECK(cudaMalloc(&imageGPU, sizeof(uchar4) * w * h));
    CUDA_CHECK(cudaMalloc(&devTris, sizeof(Triangle) * TRIANGLE_COUNT));
    CUDA_CHECK(cudaMemcpy(devTris, tris, sizeof(Triangle) * TRIANGLE_COUNT,
                          cudaMemcpyHostToDevice));
    devScene.tris = devTris;
    devScene.triCount = TRIANGLE_COUNT;
    devScene.lightPos = hostScene.lightPos;
    devScene.lightColor = hostScene.lightColor;
  }

  for (int frame = 0; frame < frames_number; ++frame) {
    double t = 2.0 * M_PI * frame / frames_number;

    double rc   = r0c + Arc * std::sin(wrc * t + prc);
    double zc   = z0c + Azc * std::sin(wzc * t + pzc);
    double phic = phi0c + wphic * t;

    double rn   = r0n + Arn * std::sin(wrn * t + prn);
    double zn   = z0n + Azn * std::sin(wzn * t + pzn);
    double phin = phi0n + wphin * t;

    Camera cam;
    cam.pos    = Vec3(rc * std::cos(phic), rc * std::sin(phic), zc);
    cam.lookAt = Vec3(rn * std::cos(phin), rn * std::sin(phin), zn);
    cam.fovDeg = angle;

    float time_ms = 0.0f;

    if (use_gpu) {
      cudaEvent_t start, stop;
      CUDA_CHECK(cudaEventCreate(&start));
      CUDA_CHECK(cudaEventCreate(&stop));
      CUDA_CHECK(cudaEventRecord(start));

      Camera camDev = cam;

      dim3 block(16, 16);
      dim3 grid(8, 8);

      renderKernel<<<grid, block>>>(camDev, w, h, sr, devScene, imageGPU);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaMemcpy(imageCPU, imageGPU, sizeof(uchar4) * w * h,
                            cudaMemcpyDeviceToHost));

      CUDA_CHECK(cudaEventRecord(stop));
      CUDA_CHECK(cudaEventSynchronize(stop));
      CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));
      CUDA_CHECK(cudaEventDestroy(start));
      CUDA_CHECK(cudaEventDestroy(stop));
    } else {
      clock_t c0 = clock();
      renderFrameCPU(cam, w, h, sr, hostScene, imageCPU);
      clock_t c1 = clock();
      time_ms = 1000.0f * float(c1 - c0) / float(CLOCKS_PER_SEC);
    }

    char filename[256];
    std::sprintf(filename, output_path, frame);
    FILE *f = std::fopen(filename, "wb");
    std::fwrite(&w, sizeof(int), 1, f);
    std::fwrite(&h, sizeof(int), 1, f);
    std::fwrite(imageCPU, sizeof(uchar4), w * h, f);
    std::fclose(f);

    std::cout << (frame + 1) << "\t" << time_ms
              << "\t" << (w * h * sr * sr) << std::endl;
  }

  if (use_gpu) {
    CUDA_CHECK(cudaFree(imageGPU));
    CUDA_CHECK(cudaFree(devTris));
  }
  std::free(imageCPU);

  return 0;
}
