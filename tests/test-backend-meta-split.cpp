// The meta backend (-sm tensor) gives each device its slice of a split tensor: a tensor with the split axis cut to the
// slice, and the strides of the dims outside the split axis scaled with it. The dims outside were those with a greater
// stride than the split axis. A tensor split along an axis of extent 1, as one head split two ways, has the same stride
// on that axis as on the dim after it, so the dim after it kept its full stride: the device with no slice had a tensor
// of 0 bytes that took a whole chunk of every write, and writing the tensor aborted (tensor write out of bounds).
//
// Each tensor below is written, read back and cleared through the meta buffer, split along axis 2 as its name says:
// heads_<a><b> has a heads on the first device and b on the second. The meta backend is over the first GPU twice, which
// splits over two devices as two GPUs do; with no GPU the test is skipped.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <vector>

static constexpr int64_t n_cols = 64;
static constexpr int64_t n_rows = 8;

static ggml_backend_meta_split_state split_by_name(const ggml_tensor * tensor, void * /*userdata*/) {
    GGML_ASSERT(strncmp(tensor->name, "heads_", 6) == 0);
    ggml_backend_meta_split_state split_state = {GGML_BACKEND_SPLIT_AXIS_2, {0}, {1}, 1};
    split_state.ne[0] = tensor->name[6] - '0';
    split_state.ne[1] = tensor->name[7] - '0';
    GGML_ASSERT(split_state.ne[0] + split_state.ne[1] == tensor->ne[2]);
    return split_state;
}

int main() {
    ggml_backend_dev_t gpu = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count() && gpu == nullptr; i++) {
        if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu = ggml_backend_dev_get(i);
        }
    }
    if (gpu == nullptr) {
        printf("no GPU for the meta backend, skipped\n");
        return 0;
    }
    ggml_backend_dev_t devs[2] = {gpu, gpu};
    ggml_backend_dev_t meta = ggml_backend_meta_device(devs, 2, split_by_name, nullptr);

    const char * names[] = {"heads_10", "heads_01", "heads_11"};
    const ggml_init_params params = {
        /*.mem_size   =*/ 3*ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    std::vector<ggml_tensor *> tensors;
    for (const char * name : names) {
        ggml_tensor * t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_cols, n_rows, (name[6] - '0') + (name[7] - '0'));
        ggml_set_name(t, name);
        tensors.push_back(t);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_dev_buffer_type(meta));
    GGML_ASSERT(buf != nullptr);

    bool ok = true;
    for (ggml_tensor * t : tensors) {
        std::vector<float> data(ggml_nelements(t));
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = 1.0f + i;
        }
        ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
        std::vector<float> back(data.size());
        ggml_backend_tensor_get(t, back.data(), 0, ggml_nbytes(t));
        bool ok_t = back == data;

        ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
        ggml_backend_tensor_get(t, back.data(), 0, ggml_nbytes(t));
        ok_t = ok_t && back == std::vector<float>(data.size(), 0.0f);

        printf("%s %s\n", ok_t ? "ok  " : "FAIL", t->name);
        ok = ok && ok_t;
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    printf("%s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
