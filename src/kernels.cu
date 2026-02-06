#include <cuda_fp16.h>
#include <cassert>
#include <vector>

#include "../tester/utils.h"

template <typename T>
__device__ T warpReduce(T val, int warp_size) {
  #pragma unroll
  for (int stride = warp_size >> 1; stride > 0; stride >>= 1) {
    val += __shfl_down_sync((1LL << warp_size) - 1, val, stride);
  }
  return val;
}

template <typename T>
__global__ void traceKernel(T* A, T* result, int n, int cols,
    int warp_size, int num_warps) {
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  int warp_id = threadIdx.x / warp_size;
  int lane_id = threadIdx.x % warp_size;
  extern __shared__ char pool[];
  T* smem = (T*)pool;
  T val = idx < n ? A[idx*cols + idx] : T(0);
  T warp_sum = warpReduce(val, warp_size);
  if (lane_id == 0) {
    smem[warp_id] = warp_sum;
  }
  __syncthreads();
  if (warp_id == 0) {
    T block_sum = lane_id < num_warps ? smem[lane_id] : T(0);
    block_sum = warpReduce(block_sum, warp_size);
    if (lane_id == 0) {
      atomicAdd(result, block_sum);
    }
  }
}

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  // Get warp size
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  const int warp_size = prop.warpSize;

  // Prepare
  T* d_input;
  T* d_result;
  cudaMalloc((void**)&d_input, rows * cols * sizeof(T));
  cudaMalloc((void**)&d_result, sizeof(T));
  cudaMemcpy(d_input, h_input.data(), rows * cols * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemset(d_result, 0, sizeof(T));

  // Launch kernel
  int block_size = 256;
  int n = std::min(rows, cols);
  int num_blocks = (n + block_size - 1) / block_size;
  int num_warps = (block_size + warp_size - 1) / warp_size;
  traceKernel<<<num_blocks, block_size, num_warps * sizeof(T)>>>(
    d_input, d_result, n, int(cols), warp_size, num_warps
  );
  cudaDeviceSynchronize();

  // Copy result, wrap up
  T result;
  cudaMemcpy(&result, d_result, sizeof(T), cudaMemcpyDeviceToHost);
  cudaFree(d_input);
  cudaFree(d_result);
  return result;
}

constexpr int BS = 64;
constexpr int MAXD = 64;

__device__ __forceinline__ int swizzle(int r, int c) {
  return r << 6 | (c ^ r);
}

template <typename T>
__global__ void flashAttentionKernel(const T* Q, const T* K, const T* V,
    int N, int M, int d, int Tc, int Tr, float softmax_scale,
    int n_kv_h, int num, T* O, bool is_causal) {
  // Batch, head, sequence tile, index in tile
  int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z, tx = threadIdx.x;
  // Global index
  int tgt_idx = bz*BS + tx;

  // Addresses of Q, O concerned; offset for K and V
  const T* Q_concerned = Q + ((bx*N + tgt_idx) * gridDim.y + by) * d;
  T* O_concerned = O + ((bx*N + tgt_idx) * gridDim.y + by) * d;
  int kv_offset = (bx*M*n_kv_h + by/num) * d;

  // Define SRAM for Q, K, V
  extern __shared__ float sram[];
  float* s_q = sram;
  float* s_k = s_q + BS*MAXD;
  float* s_v = s_k + BS*MAXD;

  // Load Q to SRAM
  if (tgt_idx < N) {
    for (int x = 0; x < d; ++x) {
      s_q[swizzle(tx, x)] = float(Q_concerned[x]);
    }
  } else {
    for (int x = 0; x < d; ++x) {
      s_q[swizzle(tx, x)] = 0.f;
    }
  }
  __syncthreads();

  // Define local arrays out, S
  float out[MAXD] = {};
  float S[BS];

  // Prepare
  float m = -INFINITY, l = 0.f;
  int j_limit = Tc;
  if (is_causal == true && bz + 1 < j_limit) {
    j_limit = bz + 1;
  }

  // Main loop
  for (int j = 0; j < j_limit; ++j) {
    int src_idx = j*BS + tx;
    if (src_idx < M) {
      for (int x = 0; x < d; ++x) {
        int from = kv_offset + src_idx*n_kv_h*d + x, to = swizzle(tx, x);
        s_k[to] = float(K[from]), s_v[to] = float(V[from]);
      }
    } else {
      for (int x = 0; x < d; ++x) {
        int to = swizzle(tx, x);
        s_k[to] = s_v[to] = 0.f;
      }
    }
    __syncthreads();
    // Compute m_new, S
    float m_new = m;
    int y_limit = M - j*BS;
    if (BS < y_limit) y_limit = BS;
    if (is_causal) {
      int causal_limit = tgt_idx - j*BS + 1;
      if (causal_limit < y_limit) y_limit = causal_limit;
    }
    for (int y = 0; y < y_limit; y++) {
      float sum = 0;
      for (int x = 0; x < d; x++) {
        sum = fmaf(s_q[swizzle(tx, x)], s_k[swizzle(y, x)], sum);
      }
      sum *= softmax_scale;
      if (sum > m_new) {
        m_new = sum;
      }
      S[y] = sum;
    }
    // Compute l, O
    float alpha = __expf(m - m_new);
    l *= alpha;
    for (int x = 0; x < d; ++x) {
      out[x] *= alpha;
    }
    for (int y = 0; y < y_limit; y++) {
      float p = __expf(S[y] - m_new);
      l += p;
      for (int x = 0; x < d; ++x) {
        out[x] = fmaf(p, s_v[swizzle(y, x)], out[x]);
      }
    }
    __syncthreads();
    // Update m
    m = m_new;
  }

  // Normalize
  if (tgt_idx < N) {
    for (int x = 0; x < d; ++x) {
      O_concerned[x] = T(out[x] / l);
    }
  }
  __syncthreads();
}

template <typename T>
void flashAttentionLaunch(const T* Q, const T* K, const T* V, T* O,
    int B, int nqh, int nkvh, int N, int M, int d, bool is_causal) {
  const int num = nqh / nkvh;  // GQA grouping factor
  const int Tc = int(ceil(double(M) / BS));
  const int Tr = int(ceil(double(N) / BS));
  const float softmax_scale = float(1. / sqrt(d));

  // Initialize O
  cudaMemset(O, 0, B * nqh * N * d * sizeof(T));

  // Launch kernel
  const int sram_size = 3 * BS * MAXD * sizeof(float);
  dim3 grid_dim(B, nqh, Tr);
  dim3 block_dim(BS);
  flashAttentionKernel<<<grid_dim, block_dim, sram_size>>>(
    Q, K, V, N, M, d, Tc, Tr, softmax_scale, nkvh, num, O, is_causal);
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim,
                    bool is_causal) {
  T* d_q;
  T* d_k;
  T* d_v;
  T* d_o;
  cudaMalloc((void**)&d_q, h_q.size() * sizeof(T));
  cudaMalloc((void**)&d_k, h_k.size() * sizeof(T));
  cudaMalloc((void**)&d_v, h_v.size() * sizeof(T));
  cudaMalloc((void**)&d_o, h_o.size() * sizeof(T));
  cudaMemcpy(d_q, h_q.data(), h_q.size() * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(d_k, h_k.data(), h_k.size() * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, h_v.data(), h_v.size() * sizeof(T), cudaMemcpyHostToDevice);
  flashAttentionLaunch(d_q, d_k, d_v, d_o, batch_size,
      query_heads, kv_heads, target_seq_len, src_seq_len, head_dim, is_causal);
  T* h_o2 = (T*)malloc(h_q.size() * sizeof(T));
  cudaMemcpy(h_o2, d_o, h_q.size() * sizeof(T), cudaMemcpyDeviceToHost);
  h_o = std::vector<T>(h_o2, h_o2 + h_q.size());
  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);
  cudaFree(d_o);
  free(h_o2);
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
