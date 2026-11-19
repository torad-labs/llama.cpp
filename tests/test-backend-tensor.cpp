#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include "../ggml/src/ggml-backend-impl.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

static size_t align_up(size_t x, size_t align) {
    return (x + align - 1) / align * align;
}

// allocate 1D F32 tensors of the given sizes into a buffer of the given buft
static ggml_backend_buffer_ptr alloc_tensors(ggml_context * ctx, ggml_backend_buffer_type_t buft, const std::vector<size_t> & ne, std::vector<ggml_tensor *> & tensors) {
    const size_t align = ggml_backend_buft_get_alignment(buft);

    tensors.clear();
    std::vector<size_t> sizes(ne.size());
    std::vector<size_t> offsets(ne.size());

    size_t total = 0;
    for (size_t i = 0; i < ne.size(); i++) {
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne[i]);
        tensors.push_back(t);
        sizes[i]    = ggml_backend_buft_get_alloc_size(buft, t);
        offsets[i]  = align_up(total, align);
        total       = offsets[i] + sizes[i];
    }

    ggml_backend_buffer_ptr buf(ggml_backend_buft_alloc_buffer(buft, total));
    if (!buf) {
        return nullptr;
    }

    char * base = (char *) ggml_backend_buffer_get_base(buf.get());
    for (size_t i = 0; i < tensors.size(); i++) {
        if (ggml_backend_tensor_alloc(buf.get(), tensors[i], base + offsets[i]) != GGML_STATUS_SUCCESS) {
            return nullptr;
        }
    }

    return buf;
}

static bool run_tensor_n_case(
        std::vector<ggml_tensor *> & tensors,
        const std::vector<size_t> & offsets,
        const std::vector<size_t> & sizes,
        const std::vector<size_t> & data_offsets) {
    const size_t n = tensors.size();

    std::vector<uint8_t> data(256, 0);
    std::vector<uint8_t> out(256, 0);

    for (size_t i = 0; i < n; i++) {
        if (sizes[i] == 0) {
            continue;
        }
        memset(data.data() + data_offsets[i], (int) (0x40 + i*0x10), sizes[i]);
    }

    ggml_backend_tensor_set_n(tensors.data(), data.data(), offsets.data(), sizes.data(), data_offsets.data(), n);

    ggml_backend_tensor_get_n(tensors.data(), out.data(), offsets.data(), sizes.data(), data_offsets.data(), n);

    for (size_t i = 0; i < n; i++) {
        if (sizes[i] == 0) {
            continue;
        }
        if (memcmp(data.data() + data_offsets[i], out.data() + data_offsets[i], sizes[i]) != 0) {
            return false;
        }
    }

    std::vector<uint8_t> slice(256, 0);
    for (size_t i = 0; i < n; i++) {
        if (sizes[i] == 0) {
            continue;
        }
        memset(slice.data(), 0, slice.size());
        ggml_backend_tensor_get(tensors[i], slice.data(), offsets[i], sizes[i]);
        if (memcmp(slice.data(), data.data() + data_offsets[i], sizes[i]) != 0) {
            return false;
        }
    }

    return true;
}

// verify batched tensor set/get on a single buffer type
static bool test_tensor_n(ggml_backend_buffer_type_t buft) {
    ggml_init_params params = {};
    params.mem_size = ggml_tensor_overhead()*32 + 1024;
    params.no_alloc = true;

    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return false;
    }

    std::vector<ggml_tensor *> tensors;
    ggml_backend_buffer_ptr buf = alloc_tensors(ctx.get(), buft, { 3, 2, 4, 3, 6 }, tensors);
    if (!buf) {
        return false;
    }

    ggml_tensor * t0    = tensors[0];
    ggml_tensor * t1    = tensors[1];
    ggml_tensor * t2    = tensors[2];
    ggml_tensor * t3    = tensors[3];
    ggml_tensor * t_big = tensors[4];

    {
        std::vector<ggml_tensor *> batch = { t0, t1, t2, t0 };
        std::vector<size_t> offsets = { 0, 0, 0, 0 };
        std::vector<size_t> sizes = { 3*4, 2*4, 4*4, 0 };
        std::vector<size_t> data_offsets = { 0, 3*4, 5*4, 64 };

        if (!run_tensor_n_case(batch, offsets, sizes, data_offsets)) {
            return false;
        }
    }

    {
        std::vector<ggml_tensor *> batch = { t_big, t_big, t_big };
        std::vector<size_t> offsets = { 0, 8, 16 };
        std::vector<size_t> sizes = { 8, 8, 8 };
        std::vector<size_t> data_offsets = { 0, 8, 16 };

        if (!run_tensor_n_case(batch, offsets, sizes, data_offsets)) {
            return false;
        }
    }

    {
        std::vector<ggml_tensor *> batch = { t0, t3 };
        std::vector<size_t> offsets = { 0, 0 };
        std::vector<size_t> sizes = { 3*4, 3*4 };
        std::vector<size_t> data_offsets = { 0, 3*4 };

        if (!run_tensor_n_case(batch, offsets, sizes, data_offsets)) {
            return false;
        }
    }

    {
        uint8_t input[24], expected[128], out[128] = {};
        for (size_t i = 0; i < sizeof(input); i++) {
            input[i] = (uint8_t) (i + 1);
        }
        ggml_backend_tensor_set(t_big, input, 0, sizeof(input));

        // Verify scattered byte reads, including repeated source offsets.
        std::vector<ggml_tensor *> tensors(128, t_big);
        std::vector<size_t> offsets(128), sizes(128, 1), data_offsets(128);
        for (size_t i = 0; i < tensors.size(); i++) {
            offsets[i] = (i * 5) % sizeof(input);
            data_offsets[i] = i;
            expected[i] = input[offsets[i]];
        }

        ggml_backend_tensor_get_n(tensors.data(), out, offsets.data(), sizes.data(), data_offsets.data(), tensors.size());
        if (memcmp(out, expected, sizeof(out)) != 0) {
            return false;
        }
    }

    return true;
}

// verify batched tensor set/get and copy across two buffer types
static bool test_tensor_n_mixed_buft(ggml_backend_buffer_type_t buft) {
    ggml_init_params params = {};
    params.mem_size = ggml_tensor_overhead()*16 + 2048;
    params.no_alloc = true;

    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return false;
    }

    ggml_tensor * t0 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 3);
    ggml_tensor * t1 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 2);
    ggml_tensor * t2 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
    ggml_tensor * s0 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 3);
    ggml_tensor * s1 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 2);
    ggml_tensor * s2 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);

    const size_t align0 = ggml_backend_buft_get_alignment(buft);
    const size_t sz_t0 = ggml_backend_buft_get_alloc_size(buft, t0);
    const size_t sz_t2 = ggml_backend_buft_get_alloc_size(buft, t2);
    const size_t buf0_size = align_up(sz_t0, align0) + sz_t2;

    ggml_backend_buffer_ptr buf0(ggml_backend_buft_alloc_buffer(buft, buf0_size));
    if (!buf0) {
        return false;
    }

    char * base0 = (char *) ggml_backend_buffer_get_base(buf0.get());
    if (ggml_backend_tensor_alloc(buf0.get(), t0, base0) != GGML_STATUS_SUCCESS) {
        return false;
    }
    char * p2 = base0 + align_up(sz_t0, align0);
    if (ggml_backend_tensor_alloc(buf0.get(), t2, p2) != GGML_STATUS_SUCCESS) {
        return false;
    }

    const size_t host_align = ggml_backend_buft_get_alignment(ggml_backend_cpu_buffer_type());
    std::vector<uint8_t> host(1024 + host_align, 0);
    uint8_t * host_ptr = (uint8_t *)(((uintptr_t)host.data() + host_align - 1) / host_align * host_align);
    ggml_backend_buffer_ptr buf1(ggml_backend_cpu_buffer_from_ptr(host_ptr, 1024));
    if (!buf1) {
        return false;
    }

    ggml_backend_buffer_type_t cpu_buft = ggml_backend_buffer_get_type(buf1.get());
    const size_t align1 = ggml_backend_buft_get_alignment(cpu_buft);
    const size_t sz_t1 = ggml_backend_buft_get_alloc_size(cpu_buft, t1);
    const size_t sz_s0 = ggml_backend_buft_get_alloc_size(cpu_buft, s0);
    const size_t sz_s1 = ggml_backend_buft_get_alloc_size(cpu_buft, s1);

    char * base1 = (char *) ggml_backend_buffer_get_base(buf1.get());
    char * p = base1;
    if (ggml_backend_tensor_alloc(buf1.get(), t1, p) != GGML_STATUS_SUCCESS) {
        return false;
    }
    p += align_up(sz_t1, align1);
    if (ggml_backend_tensor_alloc(buf1.get(), s0, p) != GGML_STATUS_SUCCESS) {
        return false;
    }
    p += align_up(sz_s0, align1);
    if (ggml_backend_tensor_alloc(buf1.get(), s1, p) != GGML_STATUS_SUCCESS) {
        return false;
    }
    p += align_up(sz_s1, align1);
    if (ggml_backend_tensor_alloc(buf1.get(), s2, p) != GGML_STATUS_SUCCESS) {
        return false;
    }

    uint8_t src0[12], src1[8], src2[16];
    memset(src0, 0xA1, sizeof(src0));
    memset(src1, 0xB2, sizeof(src1));
    memset(src2, 0xC3, sizeof(src2));

    ggml_backend_tensor_set(s0, src0, 0, sizeof(src0));
    ggml_backend_tensor_set(s1, src1, 0, sizeof(src1));
    ggml_backend_tensor_set(s2, src2, 0, sizeof(src2));

    {
        std::vector<ggml_tensor *> batch = { t0, t1, t2 };
        std::vector<size_t> offsets = { 0, 0, 0 };
        std::vector<size_t> sizes = { sizeof(src0), sizeof(src1), sizeof(src2) };
        std::vector<size_t> data_offsets = { 0, sizeof(src0), sizeof(src0) + sizeof(src1) };

        if (!run_tensor_n_case(batch, offsets, sizes, data_offsets)) {
            return false;
        }
    }

    {
        std::vector<const ggml_tensor *> srcs = { s0, s1, s2 };
        std::vector<ggml_tensor *> dsts = { t0, t1, t2 };
        ggml_backend_tensor_copy_n(srcs.data(), dsts.data(), srcs.size());

        const uint8_t * expected[3] = { src0, src1, src2 };
        const size_t sizes[3] = { sizeof(src0), sizeof(src1), sizeof(src2) };
        std::vector<uint8_t> out(64, 0);
        for (size_t i = 0; i < 3; i++) {
            memset(out.data(), 0, out.size());
            ggml_backend_tensor_get(dsts[i], out.data(), 0, sizes[i]);
            if (memcmp(out.data(), expected[i], sizes[i]) != 0) {
                return false;
            }
        }
    }

    return true;
}

// Batched set/get of many ranges, as a saved sequence's rows are: odd sizes and offsets, holes in the tensor and in host memory, ranges out of tensor order, a second tensor between them, more ranges and more bytes than a device's scratch holds at once, and one range larger than it.
// Only the ranges change, and ranges that overlap are copied in their order.
static bool test_tensor_n_ranges(ggml_backend_buffer_type_t buft) {
    const size_t n_bytes = 12*1024*1024;

    ggml_init_params params = {};
    params.mem_size = ggml_tensor_overhead()*4;
    params.no_alloc = true;

    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return false;
    }

    std::vector<ggml_tensor *> tensors;
    ggml_backend_buffer_ptr buf = alloc_tensors(ctx.get(), buft, { n_bytes/4, 256 }, tensors);
    if (!buf) {
        return false;
    }
    ggml_tensor * big   = tensors[0];
    ggml_tensor * small = tensors[1];

    uint32_t state = 12345;
    const auto next = [&]() {
        state = state*1664525u + 1013904223u;
        return state >> 8;
    };

    std::vector<ggml_tensor *> ts;
    std::vector<size_t> offsets, sizes, data_offsets;
    size_t at = 0, host = 0;
    const auto add = [&](size_t size, size_t gap, size_t hole) {
        at   += gap;
        host += hole;
        ts.push_back(big);
        offsets.push_back(at);
        sizes.push_back(size);
        data_offsets.push_back(host);
        at   += size;
        host += size;
    };
    for (int i = 0; i < 20000; i++) {
        add(1 + next() % 97, next() % 8, i % 16 == 15 ? 1 + next() % 5 : 0);
    }
    for (int i = 0; i < 40; i++) {
        add(100000 + i, 3, 0);
    }
    add(3*1024*1024, 5, 0);
    if (at > n_bytes) {
        return false;
    }
    // the first 5,000 ranges in the opposite order, and the other tensor's range after them
    for (size_t i = 0; i < 2500; i++) {
        std::swap(offsets[i], offsets[4999 - i]);
        std::swap(sizes[i], sizes[4999 - i]);
        std::swap(data_offsets[i], data_offsets[4999 - i]);
    }
    ts.insert(ts.begin() + 5000, small);
    offsets.insert(offsets.begin() + 5000, 8);
    sizes.insert(sizes.begin() + 5000, 100);
    data_offsets.insert(data_offsets.begin() + 5000, host);
    host += 100;

    std::vector<uint8_t> init(n_bytes), init_small(256*4), src(host), expected, now(n_bytes), now_small(256*4);
    for (size_t i = 0; i < n_bytes; i++) {
        init[i] = (uint8_t) (i*7 + 3);
    }
    for (size_t i = 0; i < init_small.size(); i++) {
        init_small[i] = (uint8_t) (i*5 + 1);
    }
    for (size_t i = 0; i < host; i++) {
        src[i] = (uint8_t) next();
    }
    ggml_backend_tensor_set(big, init.data(), 0, n_bytes);
    ggml_backend_tensor_set(small, init_small.data(), 0, init_small.size());

    ggml_backend_tensor_set_n(ts.data(), src.data(), offsets.data(), sizes.data(), data_offsets.data(), ts.size());

    expected = init;
    for (size_t i = 0; i < ts.size(); i++) {
        memcpy((ts[i] == big ? expected.data() : init_small.data()) + offsets[i], src.data() + data_offsets[i], sizes[i]);
    }
    ggml_backend_tensor_get(big, now.data(), 0, n_bytes);
    ggml_backend_tensor_get(small, now_small.data(), 0, now_small.size());
    if (now != expected || now_small != init_small) {
        return false;
    }

    // a tensor mirrored over the devices of a tensor split holds the ranges on every device
    std::vector<ggml_tensor *> copies;
    ggml_backend_meta_tensor_ranges(big, 0, n_bytes, /*read =*/ false,
        [](ggml_tensor * tensor, size_t offset, size_t size, size_t data_offset, void * userdata) {
            if (offset == 0 && data_offset == 0 && size == ggml_nbytes(tensor)) {
                ((std::vector<ggml_tensor *> *) userdata)->push_back(tensor);
            }
        }, &copies);
    for (ggml_tensor * copy : copies) {
        ggml_backend_tensor_get(copy, now.data(), 0, n_bytes);
        if (now != expected) {
            return false;
        }
    }

    std::vector<uint8_t> out(host, 0xA5), out_expected(host, 0xA5);
    for (size_t i = 0; i < ts.size(); i++) {
        memcpy(out_expected.data() + data_offsets[i], src.data() + data_offsets[i], sizes[i]);
    }
    ggml_backend_tensor_get_n(ts.data(), out.data(), offsets.data(), sizes.data(), data_offsets.data(), ts.size());
    if (out != out_expected) {
        return false;
    }

    // a batch of few ranges, which a device does not read as it reads many
    {
        const size_t n_few = 100;
        std::vector<uint8_t> few(host, 0xA5), few_expected(host, 0xA5);
        for (size_t i = 0; i < n_few; i++) {
            memcpy(few_expected.data() + data_offsets[i], src.data() + data_offsets[i], sizes[i]);
        }
        ggml_backend_tensor_get_n(ts.data(), few.data(), offsets.data(), sizes.data(), data_offsets.data(), n_few);
        if (few != few_expected) {
            return false;
        }
    }

    // one host range written to two tensors, one after the other, as a mirrored tensor's range is
    {
        std::vector<ggml_tensor *> two;
        std::vector<size_t> to, n, from;
        for (size_t i = 0; i < 8; i++) {
            for (ggml_tensor * t : { big, small }) {
                two.push_back(t);
                to.push_back((t == big ? 4096 : 0) + 100*i);
                n.push_back(48);
                from.push_back(200*i);
            }
        }
        ggml_backend_tensor_set_n(two.data(), src.data(), to.data(), n.data(), from.data(), two.size());
        ggml_backend_tensor_get(big, now.data(), 0, n_bytes);
        ggml_backend_tensor_get(small, now_small.data(), 0, now_small.size());
        for (size_t i = 0; i < 8; i++) {
            if (memcmp(now.data() + 4096 + 100*i, src.data() + 200*i, 48) != 0 || memcmp(now_small.data() + 100*i, src.data() + 200*i, 48) != 0) {
                return false;
            }
        }
    }

    // eight writes of one tensor range leave the last, and eight reads into one host range leave the last
    {
        std::vector<ggml_tensor *> same(8, big);
        std::vector<size_t> to(8, 64), n(8, 48), from(8), spread(8), one(8, 0);
        for (size_t i = 0; i < 8; i++) {
            from[i]   = 100*i;
            spread[i] = 1000*i;
        }
        ggml_backend_tensor_set_n(same.data(), src.data(), to.data(), n.data(), from.data(), same.size());
        uint8_t last[48];
        ggml_backend_tensor_get(big, last, 64, sizeof(last));
        if (memcmp(last, src.data() + 700, sizeof(last)) != 0) {
            return false;
        }
        ggml_backend_tensor_get(big, now.data(), 0, n_bytes);
        ggml_backend_tensor_get_n(same.data(), last, spread.data(), n.data(), one.data(), same.size());
        if (memcmp(last, now.data() + 7000, sizeof(last)) != 0) {
            return false;
        }
    }

    return true;
}

// unified test set run on a single buffer type
static bool test_buft(ggml_backend_buffer_type_t buft) {
    return test_tensor_n(buft) && test_tensor_n_mixed_buft(buft) && test_tensor_n_ranges(buft);
}

//
// Mock buffer type
//
// Ad-hoc host-backed buffer type used to exercise the dispatch logic in
// ggml_backend_tensor_{set,get,copy}_n without relying on real backend hooks.
//

struct mock_config {
    bool set_n;
    bool get_n;
    bool cpy_n;
    bool cpy_n_result;
    int  set_n_calls;
    int  get_n_calls;
    int  cpy_n_calls;
    int  copy_calls; // every set and get a buffer is asked for: one range, the rows of a 2D copy, or a batch
};

static mock_config g_mock = {};

static void mock_config_reset(bool set_n, bool get_n, bool cpy_n, bool cpy_n_result) {
    g_mock.set_n = set_n;
    g_mock.get_n = get_n;
    g_mock.cpy_n = cpy_n;
    g_mock.cpy_n_result = cpy_n_result;
    g_mock.set_n_calls = 0;
    g_mock.get_n_calls = 0;
    g_mock.cpy_n_calls = 0;
    g_mock.copy_calls = 0;
}

static void mock_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    free(buffer->context);
}

static void * mock_buffer_get_base(ggml_backend_buffer_t buffer) {
    return buffer->context;
}

static void mock_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    memset((char *) tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void mock_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    g_mock.copy_calls++;
    memcpy((char *) tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

static void mock_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    g_mock.copy_calls++;
    memcpy(data, (const char *) tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static void mock_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
        size_t n_copies, size_t stride_tensor, size_t stride_data) {
    g_mock.copy_calls++;
    for (size_t i = 0; i < n_copies; i++) {
        memcpy((char *) tensor->data + offset + i*stride_tensor, (const char *) data + i*stride_data, size);
    }

    GGML_UNUSED(buffer);
}

static void mock_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
        size_t n_copies, size_t stride_tensor, size_t stride_data) {
    g_mock.copy_calls++;
    for (size_t i = 0; i < n_copies; i++) {
        memcpy((char *) data + i*stride_data, (const char *) tensor->data + offset + i*stride_tensor, size);
    }

    GGML_UNUSED(buffer);
}

static bool mock_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    memcpy(dst->data, src->data, ggml_nbytes(src));
    return true;

    GGML_UNUSED(buffer);
}

static void mock_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(buffer->context, value, buffer->size);
}

static void mock_buffer_set_tensor_n(ggml_backend_buffer_t buffer, struct ggml_tensor * const * tensors, const void * const * data, const size_t * offsets, const size_t * sizes, size_t n_tensors) {
    g_mock.set_n_calls++;
    g_mock.copy_calls++;
    for (size_t i = 0; i < n_tensors; i++) {
        memcpy((char *) tensors[i]->data + offsets[i], data[i], sizes[i]);
    }

    GGML_UNUSED(buffer);
}

static void mock_buffer_get_tensor_n(ggml_backend_buffer_t buffer, const struct ggml_tensor * const * tensors, void * const * data, const size_t * offsets, const size_t * sizes, size_t n_tensors) {
    g_mock.get_n_calls++;
    g_mock.copy_calls++;
    for (size_t i = 0; i < n_tensors; i++) {
        memcpy(data[i], (const char *) tensors[i]->data + offsets[i], sizes[i]);
    }

    GGML_UNUSED(buffer);
}

static bool mock_buffer_cpy_tensor_n(ggml_backend_buffer_t buffer, const struct ggml_tensor * const * srcs, struct ggml_tensor * const * dsts, size_t n_tensors) {
    g_mock.cpy_n_calls++;
    if (!g_mock.cpy_n_result) {
        return false;
    }
    for (size_t i = 0; i < n_tensors; i++) {
        memcpy(dsts[i]->data, srcs[i]->data, ggml_nbytes(srcs[i]));
    }
    return true;

    GGML_UNUSED(buffer);
}

static const char * mock_buft_get_name(ggml_backend_buffer_type_t buft) {
    return "MOCK";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t mock_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = malloc(size);
    if (data == NULL) {
        return NULL;
    }

    struct ggml_backend_buffer_i iface = {
        /* .free_buffer     = */ mock_buffer_free_buffer,
        /* .get_base        = */ mock_buffer_get_base,
        /* .init_tensor     = */ NULL,
        /* .memset_tensor   = */ mock_buffer_memset_tensor,
        /* .set_tensor      = */ mock_buffer_set_tensor,
        /* .get_tensor      = */ mock_buffer_get_tensor,
        /* .set_tensor_2d   = */ mock_buffer_set_tensor_2d,
        /* .get_tensor_2d   = */ mock_buffer_get_tensor_2d,
        /* .cpy_tensor      = */ mock_buffer_cpy_tensor,
        /* .clear           = */ mock_buffer_clear,
        /* .reset           = */ NULL,
        /* .set_tensor_n    = */ NULL,
        /* .get_tensor_n    = */ NULL,
        /* .cpy_tensor_n    = */ NULL,
    };
    if (g_mock.set_n) iface.set_tensor_n = mock_buffer_set_tensor_n;
    if (g_mock.get_n) iface.get_tensor_n = mock_buffer_get_tensor_n;
    if (g_mock.cpy_n) iface.cpy_tensor_n = mock_buffer_cpy_tensor_n;

    return ggml_backend_buffer_init(buft, iface, data, size);
}

static size_t mock_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    return alignof(max_align_t); // what malloc gives: the tensors of an exact fit start at the buffer's base

    GGML_UNUSED(buft);
}

static bool mock_buft_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t mock_buft(void) {
    static struct ggml_backend_buffer_type buft = {
        /* .iface = */ {
            /* .get_name         = */ mock_buft_get_name,
            /* .alloc_buffer     = */ mock_buft_alloc_buffer,
            /* .alloc_buffer_n   = */ NULL,
            /* .get_alignment    = */ mock_buft_get_alignment,
            /* .get_max_size     = */ NULL,
            /* .get_alloc_size   = */ NULL,
            /* .get_alloc_size_n = */ NULL,
            /* .is_host          = */ mock_buft_is_host,
        },
        /* .device  = */ NULL,
        /* .context = */ NULL,
    };
    return &buft;
}

// verify that set_n/get_n route to the _n hook when installed and to the fallback otherwise
static bool test_mock_set_get_n(bool with_hooks) {
    mock_config_reset(with_hooks, with_hooks, false, true);

    ggml_init_params params = {};
    params.mem_size = ggml_tensor_overhead()*32 + 2048;
    params.no_alloc = true;

    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return false;
    }

    std::vector<ggml_tensor *> tensors;
    ggml_backend_buffer_ptr buf = alloc_tensors(ctx.get(), mock_buft(), { 3, 2, 4 }, tensors);
    if (!buf) {
        return false;
    }

    const size_t sizes[3]     = { 3*4, 2*4, 4*4 };
    const size_t data_offs[3] = { 0, 3*4, 5*4 };
    size_t offsets[3]         = { 0, 0, 0 };
    std::vector<uint8_t> data(64, 0), out(64, 0);

    for (size_t i = 0; i < 3; i++) {
        memset(data.data() + data_offs[i], (int) (0x40 + i*0x10), sizes[i]);
    }

    ggml_backend_tensor_set_n(tensors.data(), data.data(), offsets, sizes, data_offs, 3);
    ggml_backend_tensor_get_n(tensors.data(), out.data(), offsets, sizes, data_offs, 3);

    for (size_t i = 0; i < 3; i++) {
        if (memcmp(data.data() + data_offs[i], out.data() + data_offs[i], sizes[i]) != 0) {
            return false;
        }
    }

    const int expected_calls = with_hooks ? 1 : 0;
    if (g_mock.set_n_calls != expected_calls || g_mock.get_n_calls != expected_calls) {
        return false;
    }

    return true;
}

// verify that copy_n respects the all-or-nothing contract of cpy_tensor_n
static bool test_mock_copy_n(bool hook_result) {
    mock_config_reset(false, false, true, hook_result);

    ggml_init_params params = {};
    params.mem_size = ggml_tensor_overhead()*32 + 2048;
    params.no_alloc = true;

    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return false;
    }

    std::vector<ggml_tensor *> src_tensors;
    std::vector<ggml_tensor *> dst_tensors;
    ggml_backend_buffer_ptr sbuf = alloc_tensors(ctx.get(), ggml_backend_cpu_buffer_type(), { 3, 2, 4 }, src_tensors);
    ggml_backend_buffer_ptr dbuf = alloc_tensors(ctx.get(), mock_buft(), { 3, 2, 4 }, dst_tensors);
    if (!sbuf || !dbuf) {
        return false;
    }

    uint8_t sdata[3][16] = {};
    memset(sdata[0], 0xA1, 3*4);
    memset(sdata[1], 0xB2, 2*4);
    memset(sdata[2], 0xC3, 4*4);

    uint8_t junk[16];
    memset(junk, 0xEE, sizeof(junk));

    for (size_t i = 0; i < 3; i++) {
        ggml_backend_tensor_set(src_tensors[i], sdata[i], 0, ggml_nbytes(src_tensors[i]));
        ggml_backend_tensor_set(dst_tensors[i], junk, 0, ggml_nbytes(dst_tensors[i]));
    }

    std::vector<const ggml_tensor *> srcs;
    for (const auto & t : src_tensors) {
        srcs.push_back(t);
    }

    ggml_backend_tensor_copy_n(srcs.data(), dst_tensors.data(), 3);

    for (size_t i = 0; i < 3; i++) {
        std::vector<uint8_t> out(16, 0);
        ggml_backend_tensor_get(dst_tensors[i], out.data(), 0, ggml_nbytes(dst_tensors[i]));
        if (memcmp(out.data(), sdata[i], ggml_nbytes(dst_tensors[i])) != 0) {
            return false;
        }
    }

    if (g_mock.cpy_n_calls != 1) {
        return false;
    }

    return true;
}

// verify that a batch spanning multiple buffer types is grouped by buft
static bool test_mock_grouping() {
    mock_config_reset(true, true, false, true);

    ggml_init_params params = {};
    params.mem_size = ggml_tensor_overhead()*32 + 2048;
    params.no_alloc = true;

    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return false;
    }

    std::vector<ggml_tensor *> mock_tensors;
    std::vector<ggml_tensor *> cpu_tensors;
    ggml_backend_buffer_ptr mbuf = alloc_tensors(ctx.get(), mock_buft(), { 3, 4 }, mock_tensors);
    ggml_backend_buffer_ptr cbuf = alloc_tensors(ctx.get(), ggml_backend_cpu_buffer_type(), { 2 }, cpu_tensors);
    if (!mbuf || !cbuf) {
        return false;
    }

    // interleave mock/cpu/mock to exercise grouping
    std::vector<ggml_tensor *> tensors = { mock_tensors[0], cpu_tensors[0], mock_tensors[1] };
    std::vector<size_t> sizes       = { 3*4, 2*4, 4*4 };
    std::vector<size_t> data_offs   = { 0, 3*4, 5*4 };
    std::vector<size_t> offsets     = { 0, 0, 0 };
    std::vector<uint8_t> data(64, 0), out(64, 0);

    for (size_t i = 0; i < tensors.size(); i++) {
        memset(data.data() + data_offs[i], (int) (0x40 + i*0x10), sizes[i]);
    }

    ggml_backend_tensor_set_n(tensors.data(), data.data(), offsets.data(), sizes.data(), data_offs.data(), tensors.size());
    ggml_backend_tensor_get_n(tensors.data(), out.data(), offsets.data(), sizes.data(), data_offs.data(), tensors.size());

    for (size_t i = 0; i < tensors.size(); i++) {
        if (memcmp(data.data() + data_offs[i], out.data() + data_offs[i], sizes[i]) != 0) {
            return false;
        }
    }

    // mock group used the hook once, the cpu group did not
    if (g_mock.set_n_calls != 1 || g_mock.get_n_calls != 1) {
        return false;
    }

    return true;
}

static bool test_mock() {
    return
        test_mock_set_get_n(true)  &&
        test_mock_set_get_n(false) &&
        test_mock_copy_n(true)     &&
        test_mock_copy_n(false)    &&
        test_mock_grouping();
}

// A device of the mock buffer type: a tensor split over it twice counts the copies each device is asked for.
static const char * mock_dev_get_name(ggml_backend_dev_t dev) {
    return "MOCK";

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_type_t mock_dev_get_buffer_type(ggml_backend_dev_t dev) {
    return mock_buft();

    GGML_UNUSED(dev);
}

static ggml_backend_dev_t mock_dev(void) {
    static struct ggml_backend_device dev = {};
    dev.iface.get_name        = mock_dev_get_name;
    dev.iface.get_description = mock_dev_get_name;
    dev.iface.get_buffer_type = mock_dev_get_buffer_type;
    return &dev;
}

// "rows" has each row in two uneven parts, "segments" has each row as two segments of two repeats each.
static ggml_backend_meta_split_state split_rows(const ggml_tensor * tensor, void * userdata) {
    if (strcmp(tensor->name, "rows") == 0) {
        return { GGML_BACKEND_SPLIT_AXIS_0, { 96, 160 }, { 1 }, 1 };
    }
    if (strcmp(tensor->name, "segments") == 0) {
        return { GGML_BACKEND_SPLIT_AXIS_0, { 32, 32, 16, 48 }, { 2, 2 }, 2 };
    }
    return { GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1 };

    GGML_UNUSED(userdata);
}

// A state copy of tensors split along their rows over a device twice: runs of a few rows with gaps between them, as a sequence's cells are in a cache it shares, a stretch of them out of order and a few long ones.
// Only those rows change, and they read back as written. With count (the mock device), the copies the devices are asked for must not follow the number of runs.
static bool test_split_rows(ggml_backend_dev_t dev, bool count) {
    const int64_t n_embd = 256;
    const int64_t n_rows = 16384;

    ggml_backend_dev_t devs[] = { dev, dev };
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(ggml_backend_meta_device(devs, 2, split_rows, nullptr));

    ggml_init_params params = {};
    params.mem_size = ggml_tensor_overhead()*2;
    params.no_alloc = true;

    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return false;
    }

    ggml_tensor * tensors[2] = {
        ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, n_embd, n_rows),
        ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, n_embd, n_rows),
    };
    ggml_set_name(tensors[0], "rows");
    ggml_set_name(tensors[1], "segments");

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));
    if (!buf) {
        return false;
    }

    const size_t row     = ggml_row_size(GGML_TYPE_F16, n_embd);
    const size_t n_bytes = row*n_rows;

    uint32_t state = 4321;
    const auto next = [&]() {
        state = state*1664525u + 1013904223u;
        return state >> 8;
    };

    std::vector<ggml_tensor *> ts;
    std::vector<size_t> offsets, sizes, data_offsets;
    size_t host = 0;
    for (ggml_tensor * t : tensors) {
        size_t at = 0;
        for (size_t i = 0; ; i++) {
            // one to three rows, and now and then eight, nine or forty
            const size_t len = i % 1500 == 700 ? 40 : i % 1009 == 500 ? 9 : i % 97 == 48 ? 8 : 1 + next() % 3;
            at += 1 + next() % 3;
            if (at + len > (size_t) n_rows) {
                break;
            }
            ts.push_back(t);
            offsets.push_back(at*row);
            sizes.push_back(len*row);
            data_offsets.push_back(host);
            at   += len;
            host += len*row;
        }
    }
    const size_t n_runs = ts.size();
    if (n_runs < 4000) {
        return false;
    }
    // a stretch of the runs in the opposite order
    for (size_t i = 0; i < 500; i++) {
        std::swap(ts[1000 + i], ts[1999 - i]);
        std::swap(offsets[1000 + i], offsets[1999 - i]);
        std::swap(sizes[1000 + i], sizes[1999 - i]);
        std::swap(data_offsets[1000 + i], data_offsets[1999 - i]);
    }

    std::vector<uint8_t> expected[2], now(n_bytes), src(host), out(host, 0);
    for (size_t k = 0; k < 2; k++) {
        expected[k].resize(n_bytes);
        for (size_t i = 0; i < n_bytes; i++) {
            expected[k][i] = (uint8_t) (i*7 + 3 + k);
        }
        ggml_backend_tensor_set(tensors[k], expected[k].data(), 0, n_bytes);
    }
    for (size_t i = 0; i < host; i++) {
        src[i] = (uint8_t) next();
    }

    g_mock.copy_calls = 0;
    ggml_backend_tensor_set_n(ts.data(), src.data(), offsets.data(), sizes.data(), data_offsets.data(), n_runs);
    const size_t set_calls = g_mock.copy_calls;

    for (size_t i = 0; i < n_runs; i++) {
        memcpy(expected[ts[i] == tensors[1]].data() + offsets[i], src.data() + data_offsets[i], sizes[i]);
    }
    for (size_t k = 0; k < 2; k++) {
        ggml_backend_tensor_get(tensors[k], now.data(), 0, n_bytes);
        if (memcmp(now.data(), expected[k].data(), n_bytes) != 0) {
            return false;
        }
    }

    g_mock.copy_calls = 0;
    ggml_backend_tensor_get_n(ts.data(), out.data(), offsets.data(), sizes.data(), data_offsets.data(), n_runs);
    const size_t get_calls = g_mock.copy_calls;

    if (memcmp(out.data(), src.data(), host) != 0) {
        return false;
    }

    if (count) {
        printf("split rows: %zu runs, %zu copies to set them, %zu to get them\n", n_runs, set_calls, get_calls);
        if (set_calls*20 > n_runs || get_calls*20 > n_runs) {
            return false;
        }
    }

    return true;
}

int main() {
    ggml_backend_load_all();

    // run the unified test set on each backend buffer type
    const size_t n_devs = ggml_backend_dev_count();
    size_t n_ok = 0;

    for (size_t i = 0; i < n_devs; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char * name = ggml_backend_dev_name(dev);
        ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);

        const bool ok = test_buft(buft);
        printf("%s %s\n", name, ok ? "PASSED" : "FAILED");
        if (ok) {
            n_ok++;
        }
    }

    printf("%zu/%zu backends passed\n", n_ok, n_devs);

    // the batched set/get through a tensor split over a GPU twice, every tensor mirrored as a latent cache is
    bool meta_ok = true;
    for (size_t i = 0; i < n_devs; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        ggml_backend_dev_t devs[] = { dev, dev };
        ggml_backend_dev_t meta = ggml_backend_meta_device(devs, 2,
            [](const ggml_tensor *, void *) -> ggml_backend_meta_split_state {
                return { GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1 };
            }, nullptr);
        ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(meta);

        const bool ok = test_tensor_n(buft) && test_tensor_n_ranges(buft);
        printf("%s split %s\n", ggml_backend_dev_name(dev), ok ? "PASSED" : "FAILED");
        meta_ok = meta_ok && ok;
    }

    // a state copy of row-split tensors, over each device twice
    bool rows_ok = true;
    for (size_t i = 0; i < n_devs; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const bool ok = test_split_rows(dev, false);
        printf("%s split rows %s\n", ggml_backend_dev_name(dev), ok ? "PASSED" : "FAILED");
        rows_ok = rows_ok && ok;
    }

    // run the unified test set and the dispatch tests on the mock buffer type
    mock_config_reset(true, true, true, true);
    const bool mock_ok = test_buft(mock_buft()) && test_mock();
    printf("MOCK %s\n", mock_ok ? "PASSED" : "FAILED");

    // the same state copy over the mock device twice, where the copies are counted
    mock_config_reset(true, true, false, true);
    const bool mock_rows_ok = test_split_rows(mock_dev(), true);
    printf("MOCK split rows %s\n", mock_rows_ok ? "PASSED" : "FAILED");

    return n_ok == n_devs && meta_ok && rows_ok && mock_ok && mock_rows_ok ? 0 : 1;
}
