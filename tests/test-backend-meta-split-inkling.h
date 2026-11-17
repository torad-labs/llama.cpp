#pragma once

#include "ggml-backend.h"

// every tensor, cache and convolution state of Inkling 2 split over ranks of gpu as its heads are (-sm tensor)
bool test_inkling_split_decisions(ggml_backend_dev_t gpu);
