#include "common.cuh"

// the most tokens GGML_OP_LOOP_CONTROL runs on the device (a token is a grid row)
#define LOOP_CONTROL_MAX_TOKENS 64
// the widest input: a token's cur and message are held in shared memory
#define LOOP_CONTROL_MAX_EMBD   4096

// GGML_OP_LOOP_CONTROL: a looped layer's controller in two kernels, the gate inputs then the gates and the mix
void ggml_cuda_op_loop_control(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_loop_control_supported(const ggml_tensor * op);
