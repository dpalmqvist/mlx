// Copyright © 2026 Apple Inc.

#include <metal_simdgroup>

using namespace metal;

// Kernel for SDPA on a single-token query against a quantized KV cache.
//
// K and V are stored using the same packing scheme as mx.quantize() with
// group-wise affine quantization along the last (head_dim) axis:
//   q_keys / q_values : packed uint32 buffer, last dim = head_dim * bits / 32.
//   k_scales/biases, v_scales/biases : same dtype as queries; one scale and
//     one bias per (batch, head, token, group).
//
// v1 supports bits=4 only. group_size and head_dim must satisfy
// (group_size % qk_per_thread == 0) where qk_per_thread = head_dim / 32, and
// (head_dim % 64 == 0) so each lane reads an integral number of bytes.
// (Empirically: head_dim ∈ {64, 128, 256} and group_size ∈ {32, 64, 128}.)
//
// Reuses function_constants 20..25 declared in sdpa_vector.h (has_mask,
// query_transposed, do_causal, bool_mask, float_mask, has_sinks). Both
// headers are included into the same .metal file, so reusing the symbols
// keeps the dispatch glue uniform.

template <typename T, int D, int V, int group_size, int bits>
[[kernel]] void sdpa_vector_quantized(
    const device T* queries [[buffer(0)]],
    const device uint32_t* q_keys [[buffer(1)]],
    const device T* k_scales [[buffer(2)]],
    const device T* k_biases [[buffer(3)]],
    const device uint32_t* q_values [[buffer(4)]],
    const device T* v_scales [[buffer(5)]],
    const device T* v_biases [[buffer(6)]],
    device T* out [[buffer(7)]],
    const constant int& gqa_factor [[buffer(8)]],
    const constant int& N [[buffer(9)]],
    // Strides for packed K/V are in uint32 elements.
    const constant size_t& k_pack_head_stride [[buffer(10)]],
    const constant size_t& k_pack_seq_stride [[buffer(11)]],
    const constant size_t& v_pack_head_stride [[buffer(12)]],
    const constant size_t& v_pack_seq_stride [[buffer(13)]],
    // Strides for scales/biases are in T elements (scales and biases share
    // the same shape, so one set of strides suffices).
    const constant size_t& k_meta_head_stride [[buffer(14)]],
    const constant size_t& k_meta_seq_stride [[buffer(15)]],
    const constant size_t& v_meta_head_stride [[buffer(16)]],
    const constant size_t& v_meta_seq_stride [[buffer(17)]],
    const constant float& scale [[buffer(18)]],
    const device bool* bmask
    [[buffer(19), function_constant(bool_mask)]],
    const device T* fmask
    [[buffer(20), function_constant(float_mask)]],
    const constant int& mask_kv_seq_stride
    [[buffer(21), function_constant(has_mask)]],
    const constant int& mask_q_seq_stride
    [[buffer(22), function_constant(has_mask)]],
    const constant int& mask_head_stride
    [[buffer(23), function_constant(has_mask)]],
    const device T* sinks [[buffer(24), function_constant(has_sinks)]],
    const constant int& num_q_heads
    [[buffer(25), function_constant(has_sinks)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(bits == 4, "v1 sdpa_vector_quantized supports bits=4 only");
  static_assert((D % 64) == 0, "head_dim must be divisible by 64");
  static_assert((V % 64) == 0, "value_dim must be divisible by 64");
  static_assert(
      (group_size % (D / 32)) == 0,
      "group_size must be a multiple of head_dim/32");
  static_assert(
      (group_size % (V / 32)) == 0,
      "group_size must be a multiple of value_dim/32");

  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  constexpr int qk_scale_step = group_size / qk_per_thread;
  constexpr int v_scale_step = group_size / v_per_thread;

  typedef float U;

  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U vals[v_per_thread];
  thread U o[v_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  const int q_batch_head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int kv_head_idx = q_batch_head_idx / gqa_factor;
  const int o_offset = q_batch_head_idx * tpg.y + q_seq_idx;
  const int q_offset = query_transposed
      ? tpg.x * q_seq_idx + q_batch_head_idx
      : o_offset;

  queries += q_offset * D + simd_lid * qk_per_thread;
  out += o_offset * V + simd_gid * v_per_thread;

  q_keys += kv_head_idx * k_pack_head_stride + simd_gid * k_pack_seq_stride;
  k_scales += kv_head_idx * k_meta_head_stride + simd_gid * k_meta_seq_stride;
  k_biases += kv_head_idx * k_meta_head_stride + simd_gid * k_meta_seq_stride;
  q_values += kv_head_idx * v_pack_head_stride + simd_gid * v_pack_seq_stride;
  v_scales += kv_head_idx * v_meta_head_stride + simd_gid * v_meta_seq_stride;
  v_biases += kv_head_idx * v_meta_head_stride + simd_gid * v_meta_seq_stride;

  if (bool_mask) {
    bmask += q_batch_head_idx * mask_head_stride +
        simd_gid * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  if (float_mask) {
    fmask += q_batch_head_idx * mask_head_stride +
        simd_gid * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }

  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  // Per-thread group lookup: lane simd_lid covers head_dim positions
  // [simd_lid * qk_per_thread, simd_lid * qk_per_thread + qk_per_thread).
  // group_size is required to be a multiple of qk_per_thread, so all
  // qk_per_thread positions for one lane fall in a single group.
  const int qk_group_idx = simd_lid / qk_scale_step;
  const int v_group_idx = simd_lid / v_scale_step;

  // Each lane reads qk_per_thread nibbles = qk_per_thread/2 bytes (bits=4).
  // For D ∈ {64, 128, 256} → qk_per_thread ∈ {2, 4, 8} → 1, 2, 4 bytes per
  // lane. We dispatch on qk_per_thread to use the widest load that's
  // representable by a built-in integer type for cleaner codegen.
  static_assert(qk_per_thread == 2 || qk_per_thread == 4 || qk_per_thread == 8,
                "qk_per_thread must be 2, 4, or 8 (D in {64, 128, 256})");
  static_assert(v_per_thread == 2 || v_per_thread == 4 || v_per_thread == 8,
                "v_per_thread must be 2, 4, or 8 (V in {64, 128, 256})");

  U max_score = -INFINITY;
  U sum_exp_score = 0;
  if (has_sinks && simd_gid == 0) {
    max_score = static_cast<U>(sinks[q_batch_head_idx % num_q_heads]);
    sum_exp_score = 1;
  }

  for (int i = simd_gid; i < N; i += BN) {
    bool use_key = true;
    if (do_causal) {
      use_key = i <= (N - int(tpg.y) + int(q_seq_idx));
    } else if (bool_mask) {
      use_key = bmask[0];
    } else if (float_mask) {
      use_key = (fmask[0] >= -INFINITY);
    }

    if (use_key) {
      // Dequantize K nibbles for this lane. We pick the load type by
      // qk_per_thread so each lane issues exactly one packed load.
      {
        U ks = static_cast<U>(k_scales[qk_group_idx]);
        U kb = static_cast<U>(k_biases[qk_group_idx]);
        if (qk_per_thread == 2) {
          uint8_t p = ((const device uint8_t*)q_keys)[simd_lid];
          k[0] = U(p & 0xF) * ks + kb;
          k[1] = U((p >> 4) & 0xF) * ks + kb;
        } else if (qk_per_thread == 4) {
          uint16_t p = ((const device uint16_t*)q_keys)[simd_lid];
          k[0] = U(p & 0xF) * ks + kb;
          k[1] = U((p >> 4) & 0xF) * ks + kb;
          k[2] = U((p >> 8) & 0xF) * ks + kb;
          k[3] = U((p >> 12) & 0xF) * ks + kb;
        } else { // qk_per_thread == 8
          uint32_t p = q_keys[simd_lid];
          k[0] = U(p & 0xF) * ks + kb;
          k[1] = U((p >> 4) & 0xF) * ks + kb;
          k[2] = U((p >> 8) & 0xF) * ks + kb;
          k[3] = U((p >> 12) & 0xF) * ks + kb;
          k[4] = U((p >> 16) & 0xF) * ks + kb;
          k[5] = U((p >> 20) & 0xF) * ks + kb;
          k[6] = U((p >> 24) & 0xF) * ks + kb;
          k[7] = U((p >> 28) & 0xF) * ks + kb;
        }
      }

      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * k[j];
      }
      score = simd_sum(score);
      if (float_mask) {
        score += static_cast<U>(fmask[0]);
      }

      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);
      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Dequantize V nibbles for this lane.
      {
        U vs = static_cast<U>(v_scales[v_group_idx]);
        U vb = static_cast<U>(v_biases[v_group_idx]);
        if (v_per_thread == 2) {
          uint8_t p = ((const device uint8_t*)q_values)[simd_lid];
          vals[0] = U(p & 0xF) * vs + vb;
          vals[1] = U((p >> 4) & 0xF) * vs + vb;
        } else if (v_per_thread == 4) {
          uint16_t p = ((const device uint16_t*)q_values)[simd_lid];
          vals[0] = U(p & 0xF) * vs + vb;
          vals[1] = U((p >> 4) & 0xF) * vs + vb;
          vals[2] = U((p >> 8) & 0xF) * vs + vb;
          vals[3] = U((p >> 12) & 0xF) * vs + vb;
        } else { // v_per_thread == 8
          uint32_t p = q_values[simd_lid];
          vals[0] = U(p & 0xF) * vs + vb;
          vals[1] = U((p >> 4) & 0xF) * vs + vb;
          vals[2] = U((p >> 8) & 0xF) * vs + vb;
          vals[3] = U((p >> 12) & 0xF) * vs + vb;
          vals[4] = U((p >> 16) & 0xF) * vs + vb;
          vals[5] = U((p >> 20) & 0xF) * vs + vb;
          vals[6] = U((p >> 24) & 0xF) * vs + vb;
          vals[7] = U((p >> 28) & 0xF) * vs + vb;
        }
      }

      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * vals[j];
      }
    }

    q_keys += BN * k_pack_seq_stride;
    k_scales += BN * k_meta_seq_stride;
    k_biases += BN * k_meta_seq_stride;
    q_values += BN * v_pack_seq_stride;
    v_scales += BN * v_meta_seq_stride;
    v_biases += BN * v_meta_seq_stride;
    if (bool_mask) {
      bmask += BN * mask_kv_seq_stride;
    }
    if (float_mask) {
      fmask += BN * mask_kv_seq_stride;
    }
  }

  // Combine partials across simdgroups (matches sdpa_vector exactly).
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  for (int i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (simd_lid == 0) {
    for (int i = 0; i < v_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

// 2-pass variant: each threadgroup handles one (kv_head, batch, block) and
// produces partial outputs + per-block max + per-block sum. The aggregation
// across blocks is handled by the existing (type-agnostic) sdpa_vector_2pass_2
// kernel — partials, sums, and maxs are written in the same layout.
//
// Reuses function constants 20..25 from sdpa_vector.h, plus 26 (`blocks`).
template <typename T, int D, int V, int group_size, int bits>
[[kernel]] void sdpa_vector_quantized_2pass_1(
    const device T* queries [[buffer(0)]],
    const device uint32_t* q_keys [[buffer(1)]],
    const device T* k_scales [[buffer(2)]],
    const device T* k_biases [[buffer(3)]],
    const device uint32_t* q_values [[buffer(4)]],
    const device T* v_scales [[buffer(5)]],
    const device T* v_biases [[buffer(6)]],
    device T* out [[buffer(7)]],
    device float* sums [[buffer(8)]],
    device float* maxs [[buffer(9)]],
    const constant int& N [[buffer(10)]],
    const constant size_t& k_pack_head_stride [[buffer(11)]],
    const constant size_t& k_pack_seq_stride [[buffer(12)]],
    const constant size_t& v_pack_head_stride [[buffer(13)]],
    const constant size_t& v_pack_seq_stride [[buffer(14)]],
    const constant size_t& k_meta_head_stride [[buffer(15)]],
    const constant size_t& k_meta_seq_stride [[buffer(16)]],
    const constant size_t& v_meta_head_stride [[buffer(17)]],
    const constant size_t& v_meta_seq_stride [[buffer(18)]],
    const constant float& scale [[buffer(19)]],
    const device bool* bmask
    [[buffer(20), function_constant(bool_mask)]],
    const device T* fmask
    [[buffer(21), function_constant(float_mask)]],
    const constant int& mask_kv_seq_stride
    [[buffer(22), function_constant(has_mask)]],
    const constant int& mask_q_seq_stride
    [[buffer(23), function_constant(has_mask)]],
    const constant int& mask_head_stride
    [[buffer(24), function_constant(has_mask)]],
    const device T* sinks [[buffer(25), function_constant(has_sinks)]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(bits == 4, "v2 sdpa_vector_quantized_2pass_1 supports bits=4 only");
  static_assert((D % 64) == 0, "head_dim must be divisible by 64");
  static_assert((V % 64) == 0, "value_dim must be divisible by 64");

  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  constexpr int qk_scale_step = group_size / qk_per_thread;
  constexpr int v_scale_step = group_size / v_per_thread;
  static_assert(qk_per_thread == 2 || qk_per_thread == 4 || qk_per_thread == 8,
                "qk_per_thread must be 2, 4, or 8");
  static_assert(v_per_thread == 2 || v_per_thread == 4 || v_per_thread == 8,
                "v_per_thread must be 2, 4, or 8");

  typedef float U;

  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U vals[v_per_thread];
  thread U o[v_per_thread] = {0};

  const int kv_head_idx = tid.x;
  const int batch_idx = tid.y;
  const int block_idx = tid.z;
  const int gqa_factor = tptg.y;
  const int q_seq_len = tptg.z;
  const int q_seq_idx = tidtg.z;
  const int q_head_idx = gqa_factor * kv_head_idx + tidtg.y;
  const int num_kv_heads = tpg.x;
  const int num_q_heads = num_kv_heads * gqa_factor;
  const int q_batch_head_idx = batch_idx * num_q_heads + q_head_idx;
  const int o_offset = q_batch_head_idx * q_seq_len + q_seq_idx;
  const int q_offset = query_transposed
      ? num_q_heads * q_seq_idx + q_batch_head_idx
      : o_offset;

  queries += q_offset * D + simd_lid * qk_per_thread;

  const int kv_batch_head_idx = batch_idx * num_kv_heads + kv_head_idx;
  q_keys += kv_batch_head_idx * k_pack_head_stride + block_idx * k_pack_seq_stride;
  k_scales += kv_batch_head_idx * k_meta_head_stride + block_idx * k_meta_seq_stride;
  k_biases += kv_batch_head_idx * k_meta_head_stride + block_idx * k_meta_seq_stride;
  q_values += kv_batch_head_idx * v_pack_head_stride + block_idx * v_pack_seq_stride;
  v_scales += kv_batch_head_idx * v_meta_head_stride + block_idx * v_meta_seq_stride;
  v_biases += kv_batch_head_idx * v_meta_head_stride + block_idx * v_meta_seq_stride;
  out += o_offset * blocks * V + block_idx * V + simd_lid * v_per_thread;
  if (bool_mask) {
    bmask += q_batch_head_idx * mask_head_stride +
        block_idx * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  if (float_mask) {
    fmask += q_batch_head_idx * mask_head_stride +
        block_idx * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  sums += o_offset * blocks + block_idx;
  maxs += o_offset * blocks + block_idx;

  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }

  const int qk_group_idx = simd_lid / qk_scale_step;
  const int v_group_idx = simd_lid / v_scale_step;

  U max_score = -INFINITY;
  U sum_exp_score = 0;
  if (has_sinks && block_idx == 0) {
    max_score = static_cast<U>(sinks[q_head_idx]);
    sum_exp_score = 1;
  }

  for (int i = block_idx; i < N; i += blocks) {
    bool use_key = true;
    if (do_causal) {
      use_key = i <= (N - q_seq_len + int(q_seq_idx));
    } else if (bool_mask) {
      use_key = bmask[0];
    } else if (float_mask) {
      use_key = (fmask[0] >= -INFINITY);
    }
    if (use_key) {
      // Dequantize K (one packed load per lane).
      {
        U ks = static_cast<U>(k_scales[qk_group_idx]);
        U kb = static_cast<U>(k_biases[qk_group_idx]);
        if (qk_per_thread == 2) {
          uint8_t p = ((const device uint8_t*)q_keys)[simd_lid];
          k[0] = U(p & 0xF) * ks + kb;
          k[1] = U((p >> 4) & 0xF) * ks + kb;
        } else if (qk_per_thread == 4) {
          uint16_t p = ((const device uint16_t*)q_keys)[simd_lid];
          k[0] = U(p & 0xF) * ks + kb;
          k[1] = U((p >> 4) & 0xF) * ks + kb;
          k[2] = U((p >> 8) & 0xF) * ks + kb;
          k[3] = U((p >> 12) & 0xF) * ks + kb;
        } else { // qk_per_thread == 8
          uint32_t p = q_keys[simd_lid];
          k[0] = U(p & 0xF) * ks + kb;
          k[1] = U((p >> 4) & 0xF) * ks + kb;
          k[2] = U((p >> 8) & 0xF) * ks + kb;
          k[3] = U((p >> 12) & 0xF) * ks + kb;
          k[4] = U((p >> 16) & 0xF) * ks + kb;
          k[5] = U((p >> 20) & 0xF) * ks + kb;
          k[6] = U((p >> 24) & 0xF) * ks + kb;
          k[7] = U((p >> 28) & 0xF) * ks + kb;
        }
      }

      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * k[j];
      }
      score = simd_sum(score);
      if (float_mask) {
        score += static_cast<U>(fmask[0]);
      }

      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);
      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Dequantize V.
      {
        U vs = static_cast<U>(v_scales[v_group_idx]);
        U vb = static_cast<U>(v_biases[v_group_idx]);
        if (v_per_thread == 2) {
          uint8_t p = ((const device uint8_t*)q_values)[simd_lid];
          vals[0] = U(p & 0xF) * vs + vb;
          vals[1] = U((p >> 4) & 0xF) * vs + vb;
        } else if (v_per_thread == 4) {
          uint16_t p = ((const device uint16_t*)q_values)[simd_lid];
          vals[0] = U(p & 0xF) * vs + vb;
          vals[1] = U((p >> 4) & 0xF) * vs + vb;
          vals[2] = U((p >> 8) & 0xF) * vs + vb;
          vals[3] = U((p >> 12) & 0xF) * vs + vb;
        } else {
          uint32_t p = q_values[simd_lid];
          vals[0] = U(p & 0xF) * vs + vb;
          vals[1] = U((p >> 4) & 0xF) * vs + vb;
          vals[2] = U((p >> 8) & 0xF) * vs + vb;
          vals[3] = U((p >> 12) & 0xF) * vs + vb;
          vals[4] = U((p >> 16) & 0xF) * vs + vb;
          vals[5] = U((p >> 20) & 0xF) * vs + vb;
          vals[6] = U((p >> 24) & 0xF) * vs + vb;
          vals[7] = U((p >> 28) & 0xF) * vs + vb;
        }
      }

      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * vals[j];
      }
    }

    q_keys += blocks * k_pack_seq_stride;
    k_scales += blocks * k_meta_seq_stride;
    k_biases += blocks * k_meta_seq_stride;
    q_values += blocks * v_pack_seq_stride;
    v_scales += blocks * v_meta_seq_stride;
    v_biases += blocks * v_meta_seq_stride;
    if (bool_mask) {
      bmask += blocks * mask_kv_seq_stride;
    }
    if (float_mask) {
      fmask += blocks * mask_kv_seq_stride;
    }
  }

  if (simd_lid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = max_score;
  }
  for (int i = 0; i < v_per_thread; i++) {
    out[i] = static_cast<T>(o[i]);
  }
}
