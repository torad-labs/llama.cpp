#include "q8-layout.cuh"

static __global__ void q8_copy_bytes(const char * src, ggml_cuda_q8_layout sl,
                                    char * dst, ggml_cuda_q8_layout dl, size_t size) {
    ggml_cuda_pdl_sync();
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < size; i += (size_t) gridDim.x * blockDim.x) {
        const char value = sl.base ? sl.base[ggml_cuda_q8_physical_byte(sl, i)] : src[i];
        if (dl.base) { dl.base[ggml_cuda_q8_physical_byte(dl, i)] = value; }
        else         { dst[i] = value; }
    }
}

void ggml_cuda_q8_copy_bytes(const void * src, ggml_cuda_q8_layout sl,
                           void * dst, ggml_cuda_q8_layout dl, size_t size, cudaStream_t stream) {
    if (size == 0) { return; }
    GGML_ASSERT(!sl.base || (sl.offset <= sl.bytes && size <= sl.bytes - sl.offset));
    GGML_ASSERT(!dl.base || (dl.offset <= dl.bytes && size <= dl.bytes - dl.offset));
    const unsigned blocks = (unsigned) std::min<size_t>((size + 255) / 256, 65535);
    q8_copy_bytes<<<blocks, 256, 0, stream>>>((const char *) src, sl, (char *) dst, dl, size);
    CUDA_CHECK(cudaGetLastError());
}

static __global__ void q8_memset_bytes(ggml_cuda_q8_layout layout, uint8_t value, size_t size) {
    ggml_cuda_pdl_sync();
    for (size_t i = (size_t) blockIdx.x*blockDim.x + threadIdx.x; i < size; i += (size_t) gridDim.x*blockDim.x) {
        layout.base[ggml_cuda_q8_physical_byte(layout, i)] = value;
    }
}

void ggml_cuda_q8_memset_bytes(ggml_cuda_q8_layout layout, uint8_t value, size_t size, cudaStream_t stream) {
    GGML_ASSERT(layout.base && layout.offset <= layout.bytes && size <= layout.bytes - layout.offset);
    if (!size) { return; }
    const unsigned blocks = (unsigned) std::min<size_t>((size + 255)/256, 65535);
    q8_memset_bytes<<<blocks, 256, 0, stream>>>(layout, value, size);
    CUDA_CHECK(cudaGetLastError());
}

struct q8_copy_shape {
    int64_t ne[4];
    size_t nb[4];

    static q8_copy_shape of(const ggml_tensor * t) {
        q8_copy_shape shape;
        for (int i = 0; i < 4; ++i) { shape.ne[i] = t->ne[i]; shape.nb[i] = t->nb[i]; }
        return shape;
    }

    __device__ size_t byte(int64_t element, int block_size) const {
        const int64_t c = element % ne[0];
        const int64_t r = element / ne[0];
        return (c / block_size)*nb[0] + (r % ne[1])*nb[1] + (r / ne[1] % ne[2])*nb[2] + (r / (ne[1]*ne[2]))*nb[3];
    }
};

static __global__ void q8_copy_tensor(const char * src, ggml_cuda_q8_layout sl, q8_copy_shape ss,
                                     char * dst, ggml_cuda_q8_layout dl, q8_copy_shape ds, size_t nbytes) {
    ggml_cuda_pdl_sync();
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < nbytes; i += (size_t) gridDim.x * blockDim.x) {
        const int64_t element = (i / sizeof(block_q8_0)) * QK8_0;
        const size_t byte = i % sizeof(block_q8_0);
        const size_t so = ss.byte(element, QK8_0) + byte;
        const size_t d = ds.byte(element, QK8_0) + byte;
        const char value = sl.base ? sl.base[ggml_cuda_q8_physical_byte(sl, so)] : src[so];
        if (dl.base) { dl.base[ggml_cuda_q8_physical_byte(dl, d)] = value; }
        else         { dst[d] = value; }
    }
}

static __global__ void q8_copy_f32(ggml_cuda_q8_layout sl, q8_copy_shape ss, char * dst, q8_copy_shape ds, int64_t n) {
    ggml_cuda_pdl_sync();
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const size_t block = ss.byte(i, QK8_0) / sizeof(block_q8_0);
        *(float *) (dst + ds.byte(i, 1)) = float(ggml_cuda_q8_scale(sl, block)) * ggml_cuda_q8_codes(sl, block)[i % QK8_0];
    }
}

bool ggml_cuda_q8_copy_tensor(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    const auto sl = ggml_cuda_q8_layout_of(src);
    const auto dl = ggml_cuda_q8_layout_of(dst);
    if (!sl.base && !dl.base) { return false; }
    const int64_t n = ggml_nelements(src);
    GGML_ASSERT(n == ggml_nelements(dst));
    if (n == 0) { return true; }
    if (src->type == GGML_TYPE_Q8_0 && dst->type == GGML_TYPE_Q8_0) {
        const size_t bytes = (n / QK8_0) * sizeof(block_q8_0);
        const unsigned blocks = (unsigned) std::min<size_t>((bytes + 255) / 256, 65535);
        q8_copy_tensor<<<blocks, 256, 0, ctx.stream()>>>((const char *) src->data, sl, q8_copy_shape::of(src),
                (char *) dst->data, dl, q8_copy_shape::of(dst), bytes);
    } else if (src->type == GGML_TYPE_Q8_0 && dst->type == GGML_TYPE_F32 && sl.base) {
        const unsigned blocks = (unsigned) std::min<int64_t>((n + 255) / 256, 65535);
        q8_copy_f32<<<blocks, 256, 0, ctx.stream()>>>(sl, q8_copy_shape::of(src), (char *) dst->data, q8_copy_shape::of(dst), n);
    } else {
        return false;
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}
