#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>

const float K_COULOMB = 8.987551785972e9f;

struct ParticlesData {
    float* m = nullptr; // 质量
    float* q = nullptr; // 电荷
    float3* r = nullptr; // 位置
    float3* v = nullptr; // 速度
};

// 设备端辅助函数
__device__ __forceinline__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 operator*(float s, float3 a) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __forceinline__ float3& operator+=(float3& a, float3 b) {
    a.x += b.x, a.y += b.y, a.z += b.z;
    return a;
}

__device__ __forceinline__ float dot(float3 a, float3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__device__ __forceinline__ float norm(float3 a) { return sqrtf(dot(a, a)); }

__device__ __forceinline__ float3 cross(float3 a, float3 b) {
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

float3 d_mag_field;

// Boris 推进器
__device__ __forceinline__ void borisPush(float m, float q, float3& r, float3& v, float3 E, float dt, float3 d_mag_field) {
    // 半步电场加速
    v += (q * dt / (2.0f * m)) * E;
    // 磁场旋转
    float3 t = (q * dt / (2.0f * m)) * d_mag_field;
    float3 v1 = v + cross(v, t);
    float t_norm_sq = dot(t, t);
    v += cross(v1, (2.0f / (1.0f + t_norm_sq)) * t);
    // 后半步电场加速
    v += (q * dt / (2.0f * m)) * E;
    // 位置更新
    r += dt * v;
}

// 其他粒子在粒子 i 处产生的电场
__device__ float3 computeElectricField(int i, int n, const ParticlesData& p) {
    float3 E = make_float3(0.0f, 0.0f, 0.0f);
    float3 ri = p.r[i];
    for (int j = 0; j < n; j++) {
        float3 d = ri - p.r[j];
        float dist = fmaxf(norm(d), 1e-4); // max 的作用是避免除以 0
        float inv_dist3 = 1.0f / (dist * dist * dist);
        E += (p.q[j] * inv_dist3 * (j != i)) * d;
    }
    return K_COULOMB * E;
}

// 主核
__global__ void updateParticlesKernel(ParticlesData p, int n, float dt, float3 d_mag_field) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // 计算电场
    float3 E = computeElectricField(i, n, p);
    // Boris 推进
    borisPush(p.m[i], p.q[i], p.r[i], p.v[i], E, dt, d_mag_field);
}

// 启动主核
void updateParticlesLaunch(ParticlesData& d_p, int n, float dt, float3 d_mag_field) {
    int threadsPerBlock = 128;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    updateParticlesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_p, n, dt, d_mag_field);
    cudaDeviceSynchronize();
}

// 粒子信息内存管理
ParticlesData allocParticlesGPU(std::size_t n) {
    ParticlesData d_p;
    cudaMalloc(&d_p.m, n * sizeof(float));
    cudaMalloc(&d_p.q, n * sizeof(float));
    cudaMalloc(&d_p.r, n * sizeof(float3));
    cudaMalloc(&d_p.v, n * sizeof(float3));
    return d_p;
}

void freeParticlesGPU(const ParticlesData& d_p) {
    cudaFree(d_p.m);
    cudaFree(d_p.q);
    cudaFree(d_p.r);
    cudaFree(d_p.v);
}

void copyParticlesToGPU(const ParticlesData& d_p, const ParticlesData& h_p, int n) {
    cudaMemcpy(d_p.m, h_p.m, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p.q, h_p.q, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p.r, h_p.r, n * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p.v, h_p.v, n * sizeof(float3), cudaMemcpyHostToDevice);
}

void copyParticlesFromGPU(const ParticlesData& h_p, const ParticlesData& d_p, int n) {
    //cudaMemcpy(h_p.m, d_p.m, n * sizeof(float), cudaMemcpyDeviceToHost);
    //cudaMemcpy(h_p.q, d_p.q, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_p.r, d_p.r, n * sizeof(float3), cudaMemcpyDeviceToHost);
    //cudaMemcpy(h_p.v, d_p.v, n * sizeof(float3), cudaMemcpyDeviceToHost);
}

// 粒子管理
class ParticlesManager {
    std::vector<float> h_m, h_q;
    std::vector<float3> h_r, h_v;
    ParticlesData h, d;
    std::size_t n;

public:
    ParticlesManager() {}
    ~ParticlesManager() { freeParticlesGPU(d); }
    std::size_t input(std::istream& in, ParticlesData& h_p, ParticlesData& d_p) {
        while (true) {
            float x, y, z, vx, vy, vz, q, m;
            if (!(in >> x >> y >> z >> vx >> vy >> vz >> q >> m)) break;
            h_m.push_back(m);
            h_q.push_back(q);
            h_r.push_back({x, y, z});
            h_v.push_back({vx, vy, vz});
        }
        h_p = h = {h_m.data(), h_q.data(), h_r.data(), h_v.data()};
        n = h_m.size();
        d_p = d = allocParticlesGPU(n);
        copyParticlesToGPU(d, h, n);
        return n;
    }
    inline void fetch() { copyParticlesFromGPU(h, d, n); }
};

int main() {
    std::fstream fin_mag("mag.txt", std::ios::in);
    fin_mag >> d_mag_field.x >> d_mag_field.y >> d_mag_field.z;
    std::fstream fin_param("param.txt", std::ios::in);
    std::string str1, str2;
    float dt;
    int steps, interval;
    fin_param >> str1 >> str2 >> dt;
    fin_param >> str1 >> str2 >> steps;
    fin_param >> str1 >> str2 >> interval;
    std::fstream fin_particles("particles.txt", std::ios::in);
    if (!fin_particles.is_open()) {
        std::cerr << "错误：无法打开粒子初始参数文件！\n";
        return 1;
    }
    ParticlesData h_p, d_p;
    ParticlesManager manager;
    int n = int(manager.input(fin_particles, h_p, d_p));
#ifdef TEXT_FORMAT
    std::fstream fout("trajectories.txt", std::ios::out);
    int R = steps / interval + 1;
    fout << n << ' ' << R << '\n';
    for (int i = 0; i < n; ++i) {
        fout << h_p.r[i].x << ' ' << h_p.r[i].y << ' ' << h_p.r[i].z << ' ';
    }
    fout << '\n';
    for (int t = 1; t <= steps; ++t) {
        updateParticlesLaunch(d_p, n, dt, d_mag_field);
        if (t % interval == 0) {
            manager.fetch();
            for (int i = 0; i < n; ++i) {
                fout << h_p.r[i].x << ' ' << h_p.r[i].y << ' ' << h_p.r[i].z << ' ';
            }
            fout << '\n';
            std::cout << "时间：" << t * dt << " s\n";
        }
    }
#else
    std::fstream fout("trajectories.bin", std::ios::out | std::ios::binary);
    int R = steps / interval + 1;
    fout.write((const char*)&n, sizeof n);
    fout.write((const char*)&R, sizeof R);
    for (int i = 0; i < n; ++i) {
        fout.write((const char*)&h_p.r[i].x, sizeof h_p.r[i].x);
        fout.write((const char*)&h_p.r[i].y, sizeof h_p.r[i].y);
        fout.write((const char*)&h_p.r[i].z, sizeof h_p.r[i].z);
    }
    for (int t = 1; t <= steps; ++t) {
        updateParticlesLaunch(d_p, n, dt, d_mag_field);
        if (t % interval == 0) {
            manager.fetch();
            for (int i = 0; i < n; ++i) {
                fout.write((const char*)&h_p.r[i].x, sizeof h_p.r[i].x);
                fout.write((const char*)&h_p.r[i].y, sizeof h_p.r[i].y);
                fout.write((const char*)&h_p.r[i].z, sizeof h_p.r[i].z);
            }
            std::cout << "时间：" << t * dt << " s\n";
        }
    }
#endif
    fin_particles.close();
    fin_mag.close();
    fin_param.close();
    fout.close();
    return 0;
}