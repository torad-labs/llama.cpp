#include "testing.h"

#include "gguf.h"
#include "mtmd-image.h"
#include "mtmd-internal.h"

#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// this test file contains:
// 1. test cases for mtmd helpers
// 2. test cases for internal mtmd components
// internal headers can be included here

struct test_registry {
    using fn_t = void (*)(testing &);

    struct entry {
        std::string name;
        fn_t fn;
    };

    static std::vector<entry> & all() {
        static std::vector<entry> entries;
        return entries;
    }

    test_registry(const char * name, fn_t fn) {
        all().push_back({ name, fn });
    }
};

#define MAKE_TEST(name)                                               \
    static void name(testing & t);                                    \
    static const test_registry test_registry_ ## name(#name, &name);  \
    static void name(testing & t)


//
// mtmd_image
//

MAKE_TEST(test_image_preprocessor_lfm2) {
    clip_hparams hparams;
    hparams.patch_size = 16;
    hparams.n_merge = 2;
    hparams.set_limit_image_tokens(64, 256);

    // { image size, expected tiling }
    const std::vector<std::pair<clip_image_size, bool>> cases = {
        { {  704, 704 }, false },
        // 720 / (patch_size * n_merge) is exactly 22.5, so this only matches HF
        // if round_by_factor rounds half to even (22) instead of away from zero (23)
        { {  720, 720 }, false },
        { {  736, 736 }, true  },
        { { 1024, 977 }, true  },
        { { 1056, 384 }, false },
    };

    for (const auto & [size, expected] : cases) {
        const bool actual = mtmd_image_preprocessor_lfm2::should_tile(hparams, size);

        t.assert_equal(
            "tiling for " + std::to_string(size.width) + "x" + std::to_string(size.height),
            std::string(expected ? "tiled" : "single"),
            std::string(actual   ? "tiled" : "single"));
    }
}

//
// mtmd temporal merge
//

MAKE_TEST(test_temporal_merge_grouping) {
    std::vector<mtmd::bitmap_ptr> pool; // keeps the bitmaps alive until the end of the test

    // spec chars:
    //   v = video frame, w = video frame of another size, a = audio, i = plain image, t = text
    auto make_parts = [&pool](const std::string & spec) {
        std::vector<mtmd_input_part> parts;
        for (char c : spec) {
            if (c == 't') {
                parts.push_back({ "hello", nullptr });
                continue;
            }
            mtmd_bitmap * bm = nullptr;
            switch (c) {
                case 'v': bm = mtmd_bitmap_init(100, 100, nullptr);   break;
                case 'w': bm = mtmd_bitmap_init(200, 200, nullptr);   break;
                case 'a': bm = mtmd_bitmap_init_from_audio(100, nullptr); break;
                case 'i': bm = mtmd_bitmap_init(100, 100, nullptr);   break;
                default: throw std::runtime_error(std::string("unknown spec char: ") + c);
            }
            mtmd_bitmap_set_mergeable(bm, c != 'i');
            pool.emplace_back(bm);
            parts.push_back({ "", bm });
        }
        return parts;
    };

    // { parts, n_merge, expected size of each group }
    const std::vector<std::tuple<std::string, int, std::string>> cases = {
        { "vv",   2, "2"    },
        { "vvv",  2, "21"   },
        { "vvvv", 2, "22"   },
        { "vvi",  2, "21"   },
        { "tvvt", 2, "2"    },
        { "vtv",  2, "11"   }, // text in between breaks the merge
        { "vw",   2, "11"   }, // different sizes cannot be merged
        { "aa",   2, "11"   }, // audio is never merged
        { "ii",   2, "11"   }, // two unrelated images must stay separated
        { "iv",   2, "11"   },
        { "vi",   2, "11"   },
        { "vv",   1, "11"   }, // model without temporal merge
    };

    for (const auto & [spec, n_merge, expected] : cases) {
        auto parts  = make_parts(spec);
        auto groups = mtmd_group_mergeable_bitmaps(parts, n_merge);

        std::string actual;
        for (const auto & group : groups) {
            actual += std::to_string(group.size());
        }

        const std::string name = "\"" + spec + "\" with n_merge=" + std::to_string(n_merge);
        t.assert_equal("groups for " + name, expected, actual);

        size_t n_bitmap_parts = 0;
        for (const auto & p : parts) {
            n_bitmap_parts += p.bitmap != nullptr ? 1 : 0;
        }
        t.assert_equal("remaining bitmap parts for " + name, groups.size(), n_bitmap_parts);
    }
}

//
// mmproj header
//

// a glm5next projector's header alone (no tensors), as the load reads it before it loads a tensor. Unsloth's
// GLM-5.3-Flash projector names the type glm5v and the SwiGLU limit swiglu_clamp
MAKE_TEST(test_header_check_glm5next_spellings) {
    // { projector type, the SwiGLU limit's key (empty: the header has none), accepted }
    const std::vector<std::tuple<std::string, std::string, bool>> cases = {
        { "glm5next", "clip.vision.swiglu_limit", true  },
        { "glm5v",    "clip.vision.swiglu_clamp", true  }, // Unsloth's spellings
        { "glm5v",    "",                         false }, // the limit is still required
        { "glm5w",    "clip.vision.swiglu_limit", false }, // no projector type of this name
    };

    const std::string path = (std::filesystem::temp_directory_path() /
        ("test-mtmd-impl-header-" + std::to_string(std::random_device{}()) + ".gguf")).string();
    const float image_mean[3] = { 0.48145466f, 0.4578275f,  0.40821073f };
    const float image_std[3]  = { 0.26862954f, 0.26130258f, 0.27577711f };

    for (const auto & [proj_type, swiglu_key, expected] : cases) {
        gguf_context * gguf = gguf_init_empty();
        gguf_set_val_bool(gguf, "clip.has_vision_encoder",                  true);
        gguf_set_val_str (gguf, "clip.projector_type",                      proj_type.c_str());
        gguf_set_val_bool(gguf, "clip.use_silu",                            true);
        gguf_set_val_u32 (gguf, "clip.vision.projection_dim",               4096);
        gguf_set_val_u32 (gguf, "clip.vision.image_size",                   448);
        gguf_set_val_u32 (gguf, "clip.vision.patch_size",                   14);
        gguf_set_val_u32 (gguf, "clip.vision.embedding_length",             1024);
        gguf_set_val_u32 (gguf, "clip.vision.feed_forward_length",          4096);
        gguf_set_val_u32 (gguf, "clip.vision.block_count",                  24);
        gguf_set_val_u32 (gguf, "clip.vision.attention.head_count",         16);
        gguf_set_val_f32 (gguf, "clip.vision.attention.layer_norm_epsilon", 1e-5f);
        gguf_set_val_u32 (gguf, "clip.vision.spatial_merge_size",           2);
        gguf_set_arr_data(gguf, "clip.vision.image_mean", GGUF_TYPE_FLOAT32, image_mean, 3);
        gguf_set_arr_data(gguf, "clip.vision.image_std",  GGUF_TYPE_FLOAT32, image_std,  3);
        if (!swiglu_key.empty()) {
            gguf_set_val_f32(gguf, swiglu_key.c_str(), 10.0f);
        }
        const bool written = gguf_write_to_file(gguf, path.c_str(), /* only_meta= */ true);
        gguf_free(gguf);
        if (!t.assert_true("the header of " + proj_type + " is written to " + path, written)) {
            continue;
        }

        const bool actual = mtmd_check_header_from_file(path.c_str(), mtmd_context_params_default());

        t.assert_equal(
            "projector type " + proj_type + " with " + (swiglu_key.empty() ? "no SwiGLU limit" : swiglu_key),
            std::string(expected ? "accepted" : "refused"),
            std::string(actual   ? "accepted" : "refused"));
    }

    std::filesystem::remove(path);
}

//
// main
//

int main(int argc, char ** argv) {
    testing t(std::cout);
    t.verbose = true;

    // usage: test-mtmd-impl [filter_regex]
    for (int i = 1; i < argc; i++) {
        t.set_filter(argv[i]);
    }

    for (const auto & e : test_registry::all()) {
        t.test(e.name, e.fn);
    }

    return t.summary();
}
