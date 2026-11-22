#include "loop-control.cuh"
#include "convert.cuh"

// A looped layer's controller (Inkling 2 Loop): for token t, with r = rms_norm(cur) ++ rms_norm(message),
// z = down r (rank values), and out = cur - tanh(forget z)*cur + tanh(gain z)*message.
//
// At decode the op is a chain of two short kernels, so it is built for latency: whatever no earlier kernel writes (the
// weights) is requested before the kernel waits on the one before it (PDL), and each kernel then needs one round of
// loads.
//
// loop_control_down: a block a token and `rows` rows of down; thread i owns columns i, i + 256, ... of cur and of
// message (nv of each). Before the wait it loads its columns of its rows of down into registers; after it, its columns
// of cur and message, and it sums at once the two squares and each row's dots with cur and with message, which one
// block reduction totals: z = rsqrt(mean cur^2 + eps)*(dot with cur) + rsqrt(mean message^2 + eps)*(dot with message).
// Consecutive threads read consecutive columns, so a warp reads a whole Q8_0 block and its scale once.
//
// loop_control_gates: a block a warp, 32 output columns and a token. Before the wait it requests its gain and
// forget rows (contiguous in the weights) into L2. After it, lane l reads values l, l + 32, ... of each
// of the 32 rows (so the warp reads each row whole, coalesced), multiplies them by z's, and a transposing reduction
// leaves lane l the dots of row l, which it mixes into column l.

#define LOOP_CONTROL_DOWN_THREADS 256

static bool loop_control_type(const ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16 || type == GGML_TYPE_Q8_0;
}

// value k of a weight row
template <ggml_type type>
static __device__ __forceinline__ float loop_control_w(const char * row, const int k) {
    if constexpr (type == GGML_TYPE_F32) {
        return ((const float *) row)[k];
    } else if constexpr (type == GGML_TYPE_F16) {
        return __half2float(((const half *) row)[k]);
    } else if constexpr (type == GGML_TYPE_BF16) {
        return ggml_cuda_cast<float>(((const nv_bfloat16 *) row)[k]);
    } else {
        static_assert(type == GGML_TYPE_Q8_0, "loop_control: weight type");
        const block_q8_0 * b = (const block_q8_0 *) row + k/QK8_0;
        return __half2float(b->d)*b->qs[k % QK8_0];
    }
}

template <ggml_type type, int rows, int nv>
static __global__ void __launch_bounds__(LOOP_CONTROL_DOWN_THREADS) loop_control_down(
        const float * cur, const float * msg, const int64_t s_cur, const int64_t s_msg, const char * down,
        const int64_t nb_down, float * z, const int n_embd, const int rank, const float eps, const bool weights_first) {
    constexpr int n_warps = LOOP_CONTROL_DOWN_THREADS/WARP_SIZE;
    constexpr int n_sums  = 2 + 2*rows; // cur's squares, message's, then each row's dot with cur and with message
    __shared__ float part[n_warps][n_sums];

    const int t    = blockIdx.y;
    const int r0   = blockIdx.x*rows;
    const int warp = threadIdx.x/WARP_SIZE;
    const int lane = threadIdx.x%WARP_SIZE;

    if (!weights_first) {
        ggml_cuda_pdl_sync(); // down is an output of the graph: it is read after its kernel
    }
    float wc[rows][nv];
    float wm[rows][nv];
#pragma unroll
    for (int r = 0; r < rows; ++r) {
        const char * row = down + min(r0 + r, rank - 1)*nb_down;
#pragma unroll
        for (int i = 0; i < nv; ++i) {
            const int k = threadIdx.x + i*LOOP_CONTROL_DOWN_THREADS;
            wc[r][i] = k < n_embd ? loop_control_w<type>(row, k)          : 0.0f;
            wm[r][i] = k < n_embd ? loop_control_w<type>(row, n_embd + k) : 0.0f;
        }
    }
    ggml_cuda_pdl_sync();

    cur += t*s_cur;
    msg += t*s_msg;
    float sums[n_sums] = {};
#pragma unroll
    for (int i = 0; i < nv; ++i) {
        const int k = threadIdx.x + i*LOOP_CONTROL_DOWN_THREADS;
        const float a = k < n_embd ? cur[k] : 0.0f;
        const float b = k < n_embd ? msg[k] : 0.0f;
        sums[0] += a*a;
        sums[1] += b*b;
#pragma unroll
        for (int r = 0; r < rows; ++r) {
            sums[2 + 2*r] += wc[r][i]*a;
            sums[3 + 2*r] += wm[r][i]*b;
        }
    }
#pragma unroll
    for (int s = 0; s < n_sums; ++s) {
        sums[s] = warp_reduce_sum(sums[s]);
    }
    if (lane == 0) {
#pragma unroll
        for (int s = 0; s < n_sums; ++s) {
            part[warp][s] = sums[s];
        }
    }
    // the gates launch now, not at this kernel's start: their blocks would wait on the device beside this kernel's (on
    // the 5090, 8 chained controllers at 4 tokens took 73 us with the trigger at the start, 45 us here)
    ggml_cuda_pdl_lc();
    __syncthreads();
    const int r = threadIdx.x;
    if (r < rows && r0 + r < rank) {
        float sc = 0.0f, sm = 0.0f, dc = 0.0f, dm = 0.0f;
#pragma unroll
        for (int w = 0; w < n_warps; ++w) {
            sc += part[w][0];
            sm += part[w][1];
            dc += part[w][2 + 2*r];
            dm += part[w][3 + 2*r];
        }
        z[t*rank + r0 + r] = rsqrtf(sc/n_embd + eps)*dc + rsqrtf(sm/n_embd + eps)*dm;
    }
}

// 32 values in each lane of a warp to their sums over the warp, the sum of value i landing in lane i (v[0]): each step
// a lane keeps the half of its values its partner does not, and adds the partner's copy of that half
static __device__ __forceinline__ void loop_control_transpose_sum(float v[WARP_SIZE], const int lane) {
#pragma unroll
    for (int s = WARP_SIZE/2; s >= 1; s >>= 1) {
        const bool upper = lane & s;
#pragma unroll
        for (int i = 0; i < s; ++i) {
            const float send = upper ? v[i]     : v[i + s];
            const float keep = upper ? v[i + s] : v[i];
            v[i] = keep + __shfl_xor_sync(0xFFFFFFFF, send, s, WARP_SIZE);
        }
    }
}

template <ggml_type type, int n_m>
static __global__ void __launch_bounds__(WARP_SIZE) loop_control_gates(
        const float * cur, const float * msg, const int64_t s_cur, const int64_t s_msg, const char * gain,
        const int64_t nb_gain, const char * forget, const int64_t nb_forget, const float * z, float * dst,
        const int64_t s_dst, const int n_embd, const int rank, const bool gates_first) {
    const int lane = threadIdx.x;
    const int j0   = blockIdx.x*WARP_SIZE;
    const int nj   = min(WARP_SIZE, n_embd - j0);
    if (gates_first && blockIdx.y == 0 && lane == 0) {
        ggml_cuda_prefetch_l2(gain   + j0*nb_gain,   nj*nb_gain);
        ggml_cuda_prefetch_l2(forget + j0*nb_forget, nj*nb_forget);
    }
    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();

    const int t = blockIdx.y;
    float zl[n_m];
#pragma unroll
    for (int m = 0; m < n_m; ++m) {
        const int k = lane + m*WARP_SIZE;
        zl[m] = k < rank ? z[t*rank + k] : 0.0f;
    }
    // lane l takes values l, l + 32, ... of each of the warp's 32 rows: the warp reads the rows whole
    float ag[WARP_SIZE];
    float af[WARP_SIZE];
#pragma unroll
    for (int r = 0; r < WARP_SIZE; ++r) {
        const int j = min(j0 + r, n_embd - 1);
        const char * gr = gain   + j*nb_gain;
        const char * fr = forget + j*nb_forget;
        float sg = 0.0f, sf = 0.0f;
#pragma unroll
        for (int m = 0; m < n_m; ++m) {
            const int k = lane + m*WARP_SIZE;
            if (k < rank) {
                sg += loop_control_w<type>(gr, k)*zl[m];
                sf += loop_control_w<type>(fr, k)*zl[m];
            }
        }
        ag[r] = sg;
        af[r] = sf;
    }
    loop_control_transpose_sum(ag, lane);
    loop_control_transpose_sum(af, lane);
    if (lane < nj) {
        const int j = j0 + lane;
        const float g = tanhf(ag[0]);
        const float f = tanhf(af[0]);
        const float c = cur[t*s_cur + j];
        const float m = msg[t*s_msg + j];
        dst[t*s_dst + j] = (c - f*c) + g*m;
    }
}

struct loop_control_args {
    const float * cur;
    const float * msg;
    int64_t s_cur;
    int64_t s_msg;
    float * z;
    int n_embd;
    int n_tokens;
    int rank;
    bool weights_first; // down is written by no node of the graph, so no kernel before this one writes it
    bool gates_first;   // and gain and forget
};

static bool loop_control_leaf(const ggml_tensor * t) {
    while (t->view_src) {
        t = t->view_src;
    }
    return t->op == GGML_OP_NONE;
}

template <ggml_type type, int rows, int nv>
static void loop_control_down_launch(const loop_control_args & a, const ggml_tensor * down, const float eps, cudaStream_t stream) {
    const dim3 grid((a.rank + rows - 1)/rows, a.n_tokens);
    const ggml_cuda_kernel_launch_params params = {grid, dim3(LOOP_CONTROL_DOWN_THREADS), 0, stream};
    ggml_cuda_kernel_launch(loop_control_down<type, rows, nv>, params, a.cur, a.msg, a.s_cur, a.s_msg,
            (const char *) down->data, (int64_t) down->nb[1], a.z, a.n_embd, a.rank, eps, a.weights_first);
}

template <ggml_type type, int rows>
static void loop_control_down_cuda(const loop_control_args & a, const ggml_tensor * down, const float eps, cudaStream_t stream) {
    const int nv = (a.n_embd + LOOP_CONTROL_DOWN_THREADS - 1)/LOOP_CONTROL_DOWN_THREADS;
    static_assert(LOOP_CONTROL_MAX_EMBD <= 16*LOOP_CONTROL_DOWN_THREADS, "loop_control: columns a thread");
    if (nv <= 4) {
        loop_control_down_launch<type, rows, 4>(a, down, eps, stream);
    } else if (nv <= 9) {
        loop_control_down_launch<type, rows, 9>(a, down, eps, stream); // the served width, 2304
    } else {
        loop_control_down_launch<type, rows, 16>(a, down, eps, stream);
    }
}

template <ggml_type type>
static void loop_control_down_type(const loop_control_args & a, const ggml_tensor * down, const float eps, cudaStream_t stream) {
    // a block a token: at up to 4 tokens a row of down a block, the most blocks; more, 4 rows, so that each token's
    // loads of cur and message serve 4 rows (on the 5090, 8 chained controllers of F32 weights at 23 tokens: 135 us
    // a row a block, 127 us 4 rows a block)
    if (a.n_tokens <= 4) {
        loop_control_down_cuda<type, 1>(a, down, eps, stream);
    } else {
        loop_control_down_cuda<type, 4>(a, down, eps, stream);
    }
}

template <ggml_type type>
static void loop_control_gates_cuda(const loop_control_args & a, const ggml_tensor * gain, const ggml_tensor * forget,
        float * dst, const int64_t s_dst, cudaStream_t stream) {
    // a block a warp, 32 columns and a token
    const dim3 grid((a.n_embd + WARP_SIZE - 1)/WARP_SIZE, a.n_tokens);
    const ggml_cuda_kernel_launch_params params = {grid, dim3(WARP_SIZE), 0, stream};
    const int n_m = (a.rank + WARP_SIZE - 1)/WARP_SIZE;
    const auto kernel = n_m == 1 ? loop_control_gates<type, 1> : n_m == 2 ? loop_control_gates<type, 2> :
                                   loop_control_gates<type, 4>;
    ggml_cuda_kernel_launch(kernel, params, a.cur, a.msg, a.s_cur, a.s_msg, (const char *) gain->data,
            (int64_t) gain->nb[1], (const char *) forget->data, (int64_t) forget->nb[1], (const float *) a.z, dst, s_dst,
            a.n_embd, a.rank, a.gates_first);
}

bool ggml_cuda_loop_control_supported(const ggml_tensor * op) {
    const ggml_tensor * down   = op->src[2];
    const ggml_tensor * gain   = op->src[3];
    const ggml_tensor * forget = op->src[4];
    if (op->ne[1] > LOOP_CONTROL_MAX_TOKENS || op->ne[0] > LOOP_CONTROL_MAX_EMBD || !loop_control_type(down->type) ||
        !loop_control_type(gain->type) || gain->type != forget->type || down->nb[0] != ggml_type_size(down->type) ||
        gain->nb[0] != ggml_type_size(gain->type) || forget->nb[0] != ggml_type_size(forget->type)) {
        return false;
    }
    // a lane of the gates holds 4 values of a row at most
    return gain->ne[0] <= 4*WARP_SIZE;
}

void ggml_cuda_op_loop_control(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * cur    = dst->src[0];
    const ggml_tensor * msg    = dst->src[1];
    const ggml_tensor * down   = dst->src[2];
    const ggml_tensor * gain   = dst->src[3];
    const ggml_tensor * forget = dst->src[4];
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_cuda_loop_control_supported(dst));
    if (dst->ne[1] == 0) {
        return;
    }
    const float  eps    = ggml_get_op_params_f32(dst, 0);
    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<float> z(ctx.pool(), (size_t) dst->ne[1]*down->ne[1]);
    const loop_control_args a = {
        (const float *) cur->data, (const float *) msg->data, (int64_t) (cur->nb[1]/sizeof(float)),
        (int64_t) (msg->nb[1]/sizeof(float)), z.get(), (int) dst->ne[0], (int) dst->ne[1], (int) down->ne[1],
        loop_control_leaf(down), loop_control_leaf(gain) && loop_control_leaf(forget),
    };

    switch (down->type) {
        case GGML_TYPE_F32:  loop_control_down_type<GGML_TYPE_F32> (a, down, eps, stream); break;
        case GGML_TYPE_F16:  loop_control_down_type<GGML_TYPE_F16> (a, down, eps, stream); break;
        case GGML_TYPE_BF16: loop_control_down_type<GGML_TYPE_BF16>(a, down, eps, stream); break;
        case GGML_TYPE_Q8_0: loop_control_down_type<GGML_TYPE_Q8_0>(a, down, eps, stream); break;
        default: GGML_ABORT("loop_control: down of type %s", ggml_type_name(down->type));
    }

    float * d = (float *) dst->data;
    const int64_t s_dst = dst->nb[1]/sizeof(float);
    switch (gain->type) {
        case GGML_TYPE_F32:  loop_control_gates_cuda<GGML_TYPE_F32> (a, gain, forget, d, s_dst, stream); break;
        case GGML_TYPE_F16:  loop_control_gates_cuda<GGML_TYPE_F16> (a, gain, forget, d, s_dst, stream); break;
        case GGML_TYPE_BF16: loop_control_gates_cuda<GGML_TYPE_BF16>(a, gain, forget, d, s_dst, stream); break;
        case GGML_TYPE_Q8_0: loop_control_gates_cuda<GGML_TYPE_Q8_0>(a, gain, forget, d, s_dst, stream); break;
        default: GGML_ABORT("loop_control: gain of type %s", ggml_type_name(gain->type));
    }
}
