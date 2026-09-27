#pragma once

// GGML_CUDA_NVTX=1: an NVTX range around every graph evaluation ("graph") and every node that launches work, so an
// Nsight Systems capture (nsys profile -t cuda,nvtx) attributes each kernel to the node that launched it. A node's range
// is named
//     <op> <node name> r=<bytes> w=<bytes>[ w0=<src0 name>]
// r is the most bytes the node reads: every source's extent, except a GET_ROWS (the rows it gathers and its indices), a
// CPY (its source; the destination is written), a MUL_MAT_ID (the experts its tokens can select, all taken as distinct)
// and a MUL_MAT hinted as a Hadamard product, named FWHT (its activation: ggml_cuda_mul_mat computes the transform,
// ggml_cuda_op_fwht, and never reads the matrix); w is the bytes it writes, a SET_ROWS only the rows it scatters. w0 names
// a matmul's weights. Inside a fused launch an intermediate is counted as written and read though it never leaves the
// kernel, so a fused group's bytes are an upper bound; a matmul's weights, the bulk of a decode token, are exact. A launch that fuses the nodes after its own marks each one it consumed, inside
// its range: "fused " and the same name. rig's roofline probe reads these names; the format is its contract.
// Off (the default) a node costs one test of a static flag. The ranges are the host's: a CUDA graph replay has none, so a
// capture for them runs with graphs off.

#include "common.cuh"

#include <string>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define GGML_CUDA_NVTX_AVAILABLE
#endif

static bool ggml_cuda_nvtx_enabled() {
    static const bool enabled = ggml_env_switch("GGML_CUDA_NVTX");
    return enabled;
}

static std::string ggml_cuda_nvtx_node_name(const ggml_tensor * node) {
    const bool fwht = node->op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(node, 1) == GGML_HINT_SRC0_IS_HADAMARD;
    int64_t r = 0;
    int64_t w = ggml_nbytes(node);
    switch (node->op) {
        case GGML_OP_CPY:
            r = ggml_nbytes(node->src[0]);
            break;
        case GGML_OP_GET_ROWS:
            r = ggml_nbytes(node->src[1]) + ggml_nrows(node) * (int64_t) node->src[0]->nb[1];
            break;
        case GGML_OP_SET_ROWS:
            r = ggml_nbytes(node->src[0]) + ggml_nbytes(node->src[1]);
            w = ggml_nrows(node->src[0]) * (int64_t) node->nb[1];
            break;
        case GGML_OP_MUL_MAT_ID: {
            const ggml_tensor * ids      = node->src[2];
            const int64_t       n_expert = node->src[0]->ne[2];
            const int64_t       selected = std::min(n_expert, ids->ne[0] * ids->ne[1]);
            r = ggml_nbytes(node->src[0]) / n_expert * selected + ggml_nbytes(node->src[1]) + ggml_nbytes(ids);
            break;
        }
        case GGML_OP_MUL_MAT:
            if (fwht) {
                r = ggml_nbytes(node->src[1]);
                break;
            }
            [[fallthrough]];
        default:
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                if (node->src[s] != nullptr) {
                    r += ggml_nbytes(node->src[s]);
                }
            }
    }
    std::string name = std::string(fwht ? "FWHT" : ggml_op_desc(node)) + " " + node->name + " r=" + std::to_string(r) +
        " w=" + std::to_string(w);
    if (!fwht && (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID) && node->src[0]->name[0] != '\0') {
        name += std::string(" w0=") + node->src[0]->name;
    }
    return name;
}

// a range for the scope: a graph evaluation, or one node's launches
struct ggml_cuda_nvtx_range {
    bool pushed = false;

    explicit ggml_cuda_nvtx_range(const char * name) {
#ifdef GGML_CUDA_NVTX_AVAILABLE
        if (ggml_cuda_nvtx_enabled()) {
            nvtxRangePushA(name);
            pushed = true;
        }
#else
        GGML_UNUSED(name);
#endif
    }

    explicit ggml_cuda_nvtx_range(const ggml_tensor * node) {
#ifdef GGML_CUDA_NVTX_AVAILABLE
        if (ggml_cuda_nvtx_enabled()) {
            nvtxRangePushA(ggml_cuda_nvtx_node_name(node).c_str());
            pushed = true;
        }
#else
        GGML_UNUSED(node);
#endif
    }

    ~ggml_cuda_nvtx_range() {
#ifdef GGML_CUDA_NVTX_AVAILABLE
        if (pushed) {
            nvtxRangePop();
        }
#endif
    }

    ggml_cuda_nvtx_range(const ggml_cuda_nvtx_range &)             = delete;
    ggml_cuda_nvtx_range & operator=(const ggml_cuda_nvtx_range &) = delete;
};

// a node the current node's launch consumed
static void ggml_cuda_nvtx_mark_fused(const ggml_tensor * node) {
#ifdef GGML_CUDA_NVTX_AVAILABLE
    if (ggml_cuda_nvtx_enabled()) {
        nvtxMarkA(("fused " + ggml_cuda_nvtx_node_name(node)).c_str());
    }
#else
    GGML_UNUSED(node);
#endif
}
