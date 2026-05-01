// Copyright © 2024-25 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"

#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h"

#define instantiate_attn(tname, dtype, bq, bk, bd, wm, wn, mname, mtype) \
  instantiate_kernel(                                                    \
      "steel_attention_" #tname "_bq" #bq "_bk" #bk "_bd" #bd            \
      "_wm" #wm "_wn" #wn "_mask" #mname,                                \
  attention_nax, dtype, bq, bk, bd, wm, wn, mtype, float)

#define instantiate_attn_g16(tname, dtype, bq, bk, bd, wm, wn, mname, mtype) \
  instantiate_kernel(                                                        \
      "steel_attention_" #tname "_bq" #bq "_bk" #bk "_bd" #bd                \
      "_wm" #wm "_wn" #wn "_mask" #mname "_g16",                             \
  attention_nax, dtype, bq, bk, bd, wm, wn, mtype, float, mlx::steel::NAXFrag32)

#define instantiate_attn_shapes_helper(iname, itype, mname, mtype)  \
    instantiate_attn(iname, itype, 64, 32, 128, 4, 1, mname, mtype) \
    instantiate_attn(iname, itype, 64, 32,  64, 4, 1, mname, mtype) \
    instantiate_attn(iname, itype, 64, 64, 128, 4, 1, mname, mtype) \
    instantiate_attn(iname, itype, 64, 64,  64, 4, 1, mname, mtype)

#define instantiate_attn_shapes_helper_g16(iname, itype, mname, mtype)  \
    instantiate_attn_g16(iname, itype, 64, 32, 128, 2, 1, mname, mtype) \
    instantiate_attn_g16(iname, itype, 64, 32,  64, 2, 1, mname, mtype) \
    instantiate_attn_g16(iname, itype, 64, 64, 128, 2, 1, mname, mtype) \
    instantiate_attn_g16(iname, itype, 64, 64,  64, 2, 1, mname, mtype)

#define instantiate_attn_mask_helper(iname, itype) \
    instantiate_attn_shapes_helper(iname, itype, iname, itype) \
    instantiate_attn_shapes_helper(iname, itype, bool_, bool)

// Note: bool mask is intentionally omitted for g16. NAXFrag32 stages mask
// loads through MPP cooperative tensors, which reject bool element types.
// The typed-mask (same dtype as query/key) variant covers the common case;
// bool-mask g16 support requires a separate per-element load path in
// steel_attention_nax.h and is deferred to a follow-up task.
#define instantiate_attn_mask_helper_g16(iname, itype) \
    instantiate_attn_shapes_helper_g16(iname, itype, iname, itype)

instantiate_attn_mask_helper(float16, half);
instantiate_attn_mask_helper(bfloat16, bfloat);

instantiate_attn_mask_helper(float32, float);

instantiate_attn_mask_helper_g16(float16, half);
instantiate_attn_mask_helper_g16(bfloat16, bfloat);
instantiate_attn_mask_helper_g16(float32, float);
// clang-format on
