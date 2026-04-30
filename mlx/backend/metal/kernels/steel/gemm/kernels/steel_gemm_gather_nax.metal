// Copyright © 2024 Apple Inc.

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

#include "mlx/backend/metal/kernels/steel/gemm/gemm_nax.h"
#include "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_gather_nax.h"

// clang-format off
#define instantiate_gather_mm_rhs(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn) \
  instantiate_kernel(                                                             \
      "steel_gather_mm_rhs_nax_" #tname "_" #iname "_" #oname "_bm" #bm "_bn" #bn \
      "_bk" #bk "_wm" #wm "_wn" #wn,                                              \
      gather_mm_rhs_nax,                                                          \
      itype,                                                                      \
      bm,                                                                         \
      bn,                                                                         \
      bk,                                                                         \
      wm,                                                                         \
      wn,                                                                         \
      trans_a,                                                                    \
      trans_b,                                                                    \
      float)

#define instantiate_gather_mm_rhs_g16(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn) \
  instantiate_kernel(                                                             \
      "steel_gather_mm_rhs_nax_" #tname "_" #iname "_" #oname "_bm" #bm "_bn" #bn \
      "_bk" #bk "_wm" #wm "_wn" #wn "_g16",                                       \
      gather_mm_rhs_nax,                                                          \
      itype,                                                                      \
      bm,                                                                         \
      bn,                                                                         \
      bk,                                                                         \
      wm,                                                                         \
      wn,                                                                         \
      trans_a,                                                                    \
      trans_b,                                                                    \
      float,                                                                      \
      mlx::steel::NAXFrag32)

#define instantiate_gather_mm_rhs_transpose_helper(iname, itype, oname, otype, bm, bn, bk, wm, wn) \
  instantiate_gather_mm_rhs(nn, false, false, iname, itype, oname, otype, bm, bn, bk, wm, wn)  \
  instantiate_gather_mm_rhs(nt, false,  true, iname, itype, oname, otype, bm, bn, bk, wm, wn)

#define instantiate_gather_mm_rhs_transpose_helper_g16(iname, itype, oname, otype, bm, bn, bk, wm, wn) \
  instantiate_gather_mm_rhs_g16(nn, false, false, iname, itype, oname, otype, bm, bn, bk, wm, wn) \
  instantiate_gather_mm_rhs_g16(nt, false,  true, iname, itype, oname, otype, bm, bn, bk, wm, wn)

#define instantiate_gather_mm_shapes_helper(iname, itype, oname, otype)                      \
  instantiate_gather_mm_rhs_transpose_helper(iname, itype, oname, otype, 16, 128, 128, 1, 4) \
  instantiate_gather_mm_rhs_transpose_helper(iname, itype, oname, otype, 32, 128, 128, 1, 4) \
  instantiate_gather_mm_rhs_transpose_helper(iname, itype, oname, otype, 64, 128, 128, 2, 4)

#define instantiate_gather_mm_shapes_helper_g16(iname, itype, oname, otype)                      \
  instantiate_gather_mm_rhs_transpose_helper_g16(iname, itype, oname, otype, 32, 128, 128, 1, 4) \
  instantiate_gather_mm_rhs_transpose_helper_g16(iname, itype, oname, otype, 64, 128, 128, 2, 4)
// clang-format on

instantiate_gather_mm_shapes_helper(float16, half, float16, half);
instantiate_gather_mm_shapes_helper(bfloat16, bfloat, bfloat16, bfloat);
instantiate_gather_mm_shapes_helper(float32, float, float32, float);
instantiate_gather_mm_shapes_helper_g16(float16, half, float16, half);
instantiate_gather_mm_shapes_helper_g16(bfloat16, bfloat, bfloat16, bfloat);
instantiate_gather_mm_shapes_helper_g16(float32, float, float32, float);
