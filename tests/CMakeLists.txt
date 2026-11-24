find_package(Threads REQUIRED)

llama_add_compile_flags()

# Optional args:
# - LINK: libraries to link against (defaults to llama-common, which also brings llama)
function(llama_build source)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs LINK)
    cmake_parse_arguments(LLAMA_BUILD "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_BUILD_UNPARSED_ARGUMENTS})

    if (NOT LLAMA_BUILD_LINK)
        set(LLAMA_BUILD_LINK llama-common)
    endif()

    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
    target_link_libraries(${TEST_TARGET} PRIVATE ${LLAMA_BUILD_LINK})
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
endfunction()

function(llama_test target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    set(TEST_TARGET ${target})

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

function(llama_test_cmd target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND ${target}
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# Builds and runs a test source file.
# Optional args:
# - NAME: name of the executable & test target (defaults to the source file name without extension)
# - LABEL: label for the test (defaults to main)
# - ARGS: arguments to pass to the test executable
# - WORKING_DIRECTORY
# - LINK: libraries to link against (defaults to llama-common, which also brings llama)
function(llama_build_and_test source)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS LINK)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_TEST_UNPARSED_ARGUMENTS})

    if (NOT LLAMA_TEST_LINK)
        set(LLAMA_TEST_LINK llama-common)
    endif()

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
    target_link_libraries(${TEST_TARGET} PRIVATE ${LLAMA_TEST_LINK})

    add_test(
        NAME ${TEST_TARGET}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_TARGET} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# build test-tokenizer-0 target once and add many tests
llama_build(test-tokenizer-0.cpp)

llama_test(test-tokenizer-0 NAME test-tokenizer-0-bert-bge        LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-bert-bge.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-command-r       LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-command-r.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-coder  LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-coder.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-llm    LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-llm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-falcon          LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gemma-4         LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gemma-4.gguf --check-decode)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gpt-2           LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-bpe       LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-spm       LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-mpt             LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-phi-3           LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-phi-3.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen2           LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen35          LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen35.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-refact          LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-starcoder       LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

if (NOT WIN32)
    llama_test_cmd(
        ${CMAKE_CURRENT_SOURCE_DIR}/test-tokenizers-repo.sh
        NAME test-tokenizers-ggml-vocabs
        LABEL behavior
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        ARGS https://huggingface.co/ggml-org-dev/vocabs ${PROJECT_SOURCE_DIR}/models/ggml-vocabs
    )
endif()

if (LLAMA_LLGUIDANCE)
    llama_build_and_test(test-grammar-llguidance.cpp  LABEL "main behavior"  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
endif ()

# llama_context state tests: KV and recurrent state save, load, copy, rollback
set(LLAMA_CONTEXT_TEST_SRC
    test-llama-context.cpp
    llama-context/cases-kv.cpp
    llama-context/cases-rs.cpp
    llama-context/test-case.cpp
    llama-context/test-log.cpp
    llama-context/test-runner.cpp
    llama-context/test-util.cpp
)
llama_build(${LLAMA_CONTEXT_TEST_SRC})
llama_build(test-recurrent-state-rollback.cpp)
llama_build(test-recurrent-rollback-rounding.cpp)
llama_build(test-recurrent-state-async.cpp)
llama_build(test-recurrent-state-staged.cpp)

if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    # these tests are disabled on Windows because they use internal functions not exported with LLAMA_API (when building with shared libraries)
    llama_build_and_test(test-unicode.cpp  LABEL behavior  LINK llama)
    llama_build_and_test(test-sampling.cpp  LABEL behavior)
    llama_build_and_test(test-reasoning-budget.cpp  LABEL behavior)
    llama_build_and_test(test-grammar-parser.cpp  LABEL behavior  LINK llama)
    llama_build_and_test(test-grammar-integration.cpp  LABEL behavior)
    llama_build_and_test(test-llama-grammar.cpp  LABEL behavior  LINK llama)
    llama_build_and_test(test-batch-alloc.cpp  LABEL behavior  LINK llama)
    llama_build_and_test(test-kv-cache.cpp LABEL behavior)
    target_include_directories(test-kv-cache PRIVATE ${PROJECT_SOURCE_DIR}/src)
    # the long reader's bucket is read once a process: the bucket as served, and a narrower one, are their own runs
    llama_test(test-kv-cache NAME test-kv-cache-bucket LABEL behavior)
    set_tests_properties(test-kv-cache-bucket PROPERTIES ENVIRONMENT "LLAMA_INKLING2_BUCKET_PREFILL_KV=1")
    llama_test(test-kv-cache NAME test-kv-cache-bucket-2048 LABEL behavior)
    set_tests_properties(test-kv-cache-bucket-2048 PROPERTIES ENVIRONMENT "LLAMA_INKLING2_BUCKET_PREFILL_KV=1;LLAMA_INKLING2_BUCKET_CELLS=2048")
    llama_test(test-kv-cache NAME test-kv-cache-pad-256 LABEL behavior)
    set_tests_properties(test-kv-cache-pad-256 PROPERTIES ENVIRONMENT "LLAMA_KV_PAD_SHARED_LEGACY=1")
    llama_build_and_test(test-kv-mask.cpp)
    llama_build_and_test(test-recurrent-arena.cpp)
    llama_build_and_test(test-q8-backend-storage.cpp)
    llama_test(test-q8-backend-storage NAME test-q8-meta-fallback-cpu ARGS CPU --meta)
    llama_test(test-q8-backend-storage NAME test-q8-meta-cache-cpu ARGS CPU --meta-cache)
    llama_test(test-q8-backend-storage NAME test-q8-channel-views-cpu ARGS CPU --views)
    llama_build(test-glm53-profile.cpp)
    llama_test(test-glm53-profile NAME test-glm53-profile-router ARGS --router)
    llama_test(test-glm53-profile NAME test-glm53-profile-spellings ARGS --spellings)
    llama_test(test-glm53-profile NAME test-glm53-profile-persistence ARGS --persistence)
    llama_test(test-glm53-profile NAME test-glm53-profile-warps ARGS --warps)
    llama_test(test-glm53-profile NAME test-glm53-profile-explicit-sampling ARGS canonical --backend-sampling)
    foreach(profile canonical sm120-balanced sm120-wide sm120-direct)
        llama_test(test-glm53-profile NAME test-glm53-profile-${profile} ARGS ${profile})
    endforeach()
    llama_build_and_test(test-kpool-can-reuse.cpp)
    # the direct-index cutoff's row cap is read once a process: the rule as it was (8) is its own run
    llama_test(test-kpool-can-reuse NAME test-kpool-can-reuse-row-cap-8)
    set_tests_properties(test-kpool-can-reuse-row-cap-8 PROPERTIES ENVIRONMENT "LLAMA_KPOOL_DIRECT_ROW_CAP=8")
    llama_build_and_test(test-kpool-input.cpp)
    llama_build_and_test(test-host-input-threads.cpp)
    llama_build_and_test(test-kpool-pools.cpp)
    llama_build_and_test(test-topk-overlap.cpp)
    llama_build_and_test(test-chat.cpp  LABEL behavior  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    target_include_directories(test-chat PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-chat PRIVATE server-context)
    llama_build_and_test(test-utf8-replace.cpp)
    target_include_directories(test-utf8-replace PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-utf8-replace PRIVATE server-context)
    if (LLAMA_BUILD_TOOLS AND LLAMA_BUILD_SERVER)
        llama_build_and_test(test-server-speculation-stats.cpp)
        target_include_directories(test-server-speculation-stats PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
        target_link_libraries(test-server-speculation-stats PRIVATE server-context)
        llama_build_and_test(test-server-response.cpp)
        target_include_directories(test-server-response PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
        target_link_libraries(test-server-response PRIVATE server-context)
    endif()
    llama_build_and_test(test-server-component.cpp  LABEL behavior)
    target_include_directories(test-server-component PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-server-component PRIVATE server-context)
    llama_build_and_test(test-json-schema-to-grammar.cpp  LABEL behavior)

    if (NOT GGML_BACKEND_DL)
        llama_build(test-quantize-stats.cpp)
    endif()

    llama_build(test-gbnf-validator.cpp LINK llama)

    # build test-tokenizer-1-bpe target once and add many tests
    llama_build(test-tokenizer-1-bpe.cpp)

    # TODO: disabled due to slowness
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-aquila     LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-aquila.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-falcon     LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-2      LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-neox   LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-neox.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-llama-bpe  LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf --ignore-merges)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-mpt        LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-refact     LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-starcoder  LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

    # build test-tokenizer-1-spm target once and add many tests
    llama_build(test-tokenizer-1-spm.cpp)

    llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-llama-spm  LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
    #llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-baichuan  LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-baichuan.gguf)

    # llama_build_and_test(test-double-float.cpp) # SLOW

endif()

llama_build(test-llama-archs.cpp llama-archs-gguf.cpp test-llama-archs-inkling.cpp test-llama-archs-inkling-paths.cpp
    test-inkling-archs-data.cpp)
# section 3.5 skeleton (doc 18-global-kv-off-f16.md): the mixed recency store's pools and dense read merge
llama_build(test-llama-kv-mixed.cpp)
if (GGML_CPU AND (NOT WIN32 OR NOT BUILD_SHARED_LIBS))
    llama_build(test-fp8.cpp)
    set(FP8_MODEL_DIR "${CMAKE_CURRENT_BINARY_DIR}/test-models-fp8")
    file(MAKE_DIRECTORY "${FP8_MODEL_DIR}")
    llama_test(test-llama-archs NAME test-generate-fp8-models LABEL "main behavior backend"
        ARGS -a llama -s 1 -d 0.1 -o "${FP8_MODEL_DIR}")
    set_tests_properties(test-generate-fp8-models PROPERTIES FIXTURES_SETUP fp8-models)
    llama_test(test-fp8 NAME test-fp8-requantize LABEL "main behavior backend"
        ARGS --models "${FP8_MODEL_DIR}")
    llama_test(test-fp8 NAME test-fp8-kv LABEL "main behavior backend"
        ARGS --models "${FP8_MODEL_DIR}" --kv)
    set_tests_properties(test-fp8-requantize test-fp8-kv PROPERTIES FIXTURES_REQUIRED fp8-models)
endif()

set(MODEL_DIR "${CMAKE_CURRENT_BINARY_DIR}/test-models/")
file(MAKE_DIRECTORY "${MODEL_DIR}")

llama_test(
    test-llama-archs
    NAME test-generate-models
    LABEL "main behavior backend"
    ARGS -o "${MODEL_DIR}"
)
set_tests_properties(test-generate-models PROPERTIES
    FIXTURES_SETUP generate-models
)

if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    # each architecture's own run and the recurrent state tests: they use internal functions, as the block above
    llama_test(test-llama-archs)
    llama_test(test-llama-archs NAME test-glm5next ARGS -a glm5next -s 1234)
    llama_test(test-llama-archs NAME test-glm-nextn-store-cpu ARGS --glm-nextn-store-only --glm-dp-device CPU -s 1234)
    llama_test(test-llama-archs NAME test-qwen35 ARGS -a qwen35 -s 1234)
    # The all-architectures run and the GLM run are the suite's longest without a device, about 230 s and 180 s alone on
    # a 16-thread host. In a suite at -j 6 on that host at load 45 to 50 the first passed ctest's default 1,500 s and the
    # second took 1,363 s: slow, not hung. An hour holds that six times over
    set_tests_properties(test-llama-archs test-glm5next PROPERTIES TIMEOUT 3600)
    if (GGML_CUDA)
        # Inkling 2 under a two-rank tensor split on the first GPU against that GPU alone, sequence state included
        foreach(inkling_case prefill decode chunks seq-cp state rollback graph-churn compact-greedy layer-inp)
            llama_test(test-llama-archs NAME test-inkling2-split-${inkling_case}
                ARGS --inkling-split ${inkling_case} --inkling-split-device CUDA0,CUDA0 -s 1234)
        endforeach()
        # Inkling 2 under -sm data (sequence-parallel rows, the experts partitioned a rank) against that GPU alone: a
        # prefill of one sequence a rank, a decode step after it, and a chunk that ends mid-sequence, two and four ranks
        llama_test(test-llama-archs NAME test-inkling2-data-2ranks ARGS --inkling-data CUDA0,CUDA0 -s 1234)
        llama_test(test-llama-archs NAME test-inkling2-data-4ranks ARGS --inkling-data CUDA0,CUDA0,CUDA0,CUDA0 -s 1234)
        # the synthetic Inkling 2 against the host with its router and relative-bias table on the split-K and short-K
        # kernels
        llama_test(test-llama-archs NAME test-inkling-f32-model ARGS -a inkling2 -s 1234)
        set_tests_properties(test-inkling-f32-model PROPERTIES ENVIRONMENT "GGML_CUDA_F32_FEW=1")
        # Inkling 2's products of the Q input as one few-row launch against one launch each, bit for bit, on the first
        # GPU and under a two-rank tensor split on it (GGML_CUDA_MMQ_FEW_GROUP_LEGACY)
        llama_test(test-llama-archs NAME test-inkling-stack-cuda0 ARGS --inkling-stack CUDA0 -s 1234)
        llama_test(test-llama-archs NAME test-inkling-stack-split-cuda0 ARGS --inkling-stack CUDA0,CUDA0 -s 1234)
        # Inkling 2 dirty-range delta uploads + batched small H2D inputs verified bit-for-bit against legacy full writes
        llama_test(test-llama-archs NAME test-inkling-delta-upload        ARGS --inkling-delta-upload CUDA0 -s 1234)
        llama_test(test-llama-archs NAME test-inkling-delta-upload-split2 ARGS --inkling-delta-upload CUDA0,CUDA0 -s 1234)
        llama_test(test-llama-archs NAME test-inkling-delta-upload-split4 ARGS --inkling-delta-upload CUDA0,CUDA0,CUDA0,CUDA0 -s 1234)
    endif()
    # Inkling 2's optimized paths against the paths they replaced, on the CPU and the GPU: the relative bias inside the
    # attention kernel (rel), the layer's fused ops (fuse), its short convolutions (conv), and its quantized KV caches
    # against F16 (kv)
    set(inkling_path_devices CPU)
    if (GGML_CUDA)
        list(APPEND inkling_path_devices CUDA0)
    endif()
    foreach(inkling_path_device ${inkling_path_devices})
        string(TOLOWER ${inkling_path_device} inkling_path_name)
        # sessions that share one cache, a call of several micro-batches included, against each session alone
        llama_test(test-llama-archs NAME test-inkling-window-${inkling_path_name}
            ARGS --inkling-window ${inkling_path_device} -s 1234)
        # the run-time pass count of Inkling 2 Loop: counts, reuse, validation and slot sizing
        llama_test(test-llama-archs NAME test-inkling2-loop-${inkling_path_name}
            ARGS --inkling-loop ${inkling_path_device} -s 1234)
        foreach(inkling_case prompt decode unified streams)
            llama_test(test-llama-archs NAME test-inkling-rel-${inkling_path_name}-${inkling_case}
                ARGS --inkling-path rel/${inkling_case} --inkling-path-device ${inkling_path_device} -s 1234)
        endforeach()
        foreach(inkling_case decode churn streams)
            llama_test(test-llama-archs NAME test-inkling-rel-packed-${inkling_path_name}-${inkling_case}
                ARGS --inkling-path rel-packed/${inkling_case} --inkling-path-device ${inkling_path_device} -s 1234)
        endforeach()
        llama_test(test-llama-archs NAME test-inkling-group-${inkling_path_name}-drafted
            ARGS --inkling-path group/drafted --inkling-path-device ${inkling_path_device} -s 1234)
        llama_test(test-llama-archs NAME test-inkling-decode-rows-${inkling_path_name}-long
            ARGS --inkling-path decode-rows/long --inkling-path-device ${inkling_path_device} -s 1234)
        foreach(inkling_case prompt unified outputs)
            llama_test(test-llama-archs NAME test-inkling-fuse-${inkling_path_name}-${inkling_case}
                ARGS --inkling-path fuse/${inkling_case} --inkling-path-device ${inkling_path_device} -s 1234)
        endforeach()
        foreach(inkling_case decode unified streams)
            llama_test(test-llama-archs NAME test-inkling-conv-${inkling_path_name}-${inkling_case}
                ARGS --inkling-path conv/${inkling_case} --inkling-path-device ${inkling_path_device} -s 1234)
        endforeach()
        foreach(inkling_kv q8_0 q4_0 split)
            foreach(inkling_case prompt unified)
                llama_test(test-llama-archs NAME test-inkling-kv-${inkling_path_name}-${inkling_kv}-${inkling_case}
                    ARGS --inkling-path kv-${inkling_kv}/${inkling_case} --inkling-path-device ${inkling_path_device} -s 1234)
            endforeach()
        endforeach()
        # the global layers' keys and values at one byte a value (q8_0, f8_e4m3) against F16, the sliding layers' F16:
        # 23 sessions that share a prompt in low-mean decode steps, and one session saved and read back
        foreach(inkling_kv q8_0 e4m3)
            foreach(inkling_case served restore)
                set(inkling_kv1b_test test-inkling-kv1b-${inkling_path_name}-${inkling_kv}-${inkling_case})
                llama_test(test-llama-archs NAME ${inkling_kv1b_test}
                    ARGS --inkling-path kv1b-${inkling_kv}/${inkling_case} --inkling-path-device ${inkling_path_device} -s 1234)
                set_tests_properties(${inkling_kv1b_test} PROPERTIES ENVIRONMENT
                    "GGML_CUDA_FATTN_REL_DECODE_ONE_BYTE=1;LLAMA_INKLING2_MEAN_MASK_DECODE=1;LLAMA_INKLING2_MEAN_POOL_MIN=8192")
            endforeach()
        endforeach()
        # doc 18 section 3.2 (campaign row inkling-optimize-kv-gates): each fast path on a q8_0, q4_0 or f8_e4m3 global
        # cache against F16 on the same path, and on a GPU the attention kernel the path names on the quantized cache.
        # f8_e4m3 runs without GGML_CUDA_FATTN_REL_DECODE_ONE_BYTE where the path does not need it, so its by-element
        # decode is what takes the low-mean and drafted steps; q4_0 rides GGML_CUDA_FATTN_REL_DECODE_Q4
        if (NOT inkling_path_device STREQUAL "CPU")
            foreach(inkling_kv q8_0 q4_0 e4m3)
                set(inkling_kvg_type "")
                if (inkling_kv STREQUAL "q4_0")
                    set(inkling_kvg_type "GGML_CUDA_FATTN_REL_DECODE_Q4=1")
                endif()
                set(inkling_kvg_runs "GGML_CUDA_FATTN_REL_QUERY_FIRST=1;GGML_CUDA_FATTN_REL_DECODE_ONE_BYTE=1")
                if (inkling_kv STREQUAL "q4_0")
                    set(inkling_kvg_runs "GGML_CUDA_FATTN_REL_QUERY_FIRST=1;GGML_CUDA_FATTN_REL_DECODE_Q4=1")
                endif()
                foreach(inkling_kvg mean:served group:drafted short:unified runs:runs bucket:bucket)
                    string(REPLACE ":" ";" inkling_kvg_pair ${inkling_kvg})
                    list(GET inkling_kvg_pair 0 inkling_kvg_path)
                    list(GET inkling_kvg_pair 1 inkling_kvg_case)
                    set(inkling_kvg_test test-inkling-kvg-${inkling_kvg_path}-${inkling_kv})
                    llama_test(test-llama-archs NAME ${inkling_kvg_test}
                        ARGS --inkling-path kvg-${inkling_kvg_path}-${inkling_kv}/${inkling_kvg_case}
                             --inkling-path-device ${inkling_path_device} -s 1234)
                    if (inkling_kvg_path STREQUAL "mean")
                        set(inkling_kvg_env "LLAMA_INKLING2_MEAN_MASK_DECODE=1;LLAMA_INKLING2_MEAN_POOL_MIN=8192;${inkling_kvg_type}")
                    elseif (inkling_kvg_path STREQUAL "runs")
                        set(inkling_kvg_env "${inkling_kvg_runs}")
                    elseif (inkling_kvg_path STREQUAL "bucket")
                        set(inkling_kvg_env "LLAMA_INKLING2_BUCKET_PREFILL_KV=1")
                    else()
                        set(inkling_kvg_env "${inkling_kvg_type}")
                    endif()
                    set_tests_properties(${inkling_kvg_test} PROPERTIES LABELS "main" ENVIRONMENT "${inkling_kvg_env}")
                endforeach()
            endforeach()
        endif()
        # section 3.5 skeleton (doc 18-global-kv-off-f16.md): the mixed recency store's dual pools route stores by
        # recency and its dense read merge serves every cell from exactly one source - all-hot and all-cold fixtures
        # equal their single pool bit for bit, evictions fall back to cold exactly
        set(inkling_kvmix_test test-inkling-kvmix-${inkling_path_name})
        llama_test(test-llama-kv-mixed NAME ${inkling_kvmix_test} ARGS ${inkling_path_device})
        set_tests_properties(${inkling_kvmix_test} PROPERTIES ENVIRONMENT "LLAMA_INKLING2_KV_MIXED=1")
        # Doc 18 section 3.5 end-to-end: pre-store F16 hot ring on q4_0 global KV (all-near matches F16 < 1e-4;
        # mixed near/far matches CPU across prefill, kv_runs shared+private, drafted verify, and restore)
        foreach(inkling_case unified decode)
            set(inkling_kvmix_near_test test-inkling-kvmix-near-${inkling_path_name}-${inkling_case})
            llama_test(test-llama-archs NAME ${inkling_kvmix_near_test}
                ARGS --inkling-path kv-mixed-near/${inkling_case} --inkling-path-device ${inkling_path_device} -s 1234)
            set_tests_properties(${inkling_kvmix_near_test} PROPERTIES ENVIRONMENT
                "GGML_CUDA_FATTN_REL_DECODE_Q4=1;GGML_CUDA_FATTN_REL_PACKED_PREFILL=1")
        endforeach()
        foreach(inkling_case prompt served drafted restore)
            set(inkling_kvmix_live_test test-inkling-kvmix-live-${inkling_path_name}-${inkling_case})
            llama_test(test-llama-archs NAME ${inkling_kvmix_live_test}
                ARGS --inkling-path kv-mixed/${inkling_case} --inkling-path-device ${inkling_path_device} -s 1234)
            set_tests_properties(${inkling_kvmix_live_test} PROPERTIES ENVIRONMENT
                "GGML_CUDA_FATTN_REL_DECODE_Q4=1;GGML_CUDA_FATTN_REL_PACKED_PREFILL=1;GGML_CUDA_FATTN_REL_RUNS_SHARED=1;GGML_CUDA_FATTN_REL_RUNS_GROUP=3")
        endforeach()
        # the global layers' decode by the runs of cells each row reads against its mask (LLAMA_INKLING2_KV_RUNS), and on
        # a GPU with the shared runs read once for groups of 3 rows (GGML_CUDA_FATTN_REL_RUNS_SHARED)
        foreach(inkling_case runs served restore drafted)
            llama_test(test-llama-archs NAME test-inkling-far-${inkling_path_name}-${inkling_case}
                ARGS --inkling-path far/${inkling_case} --inkling-path-device ${inkling_path_device} -s 1234)
            if (NOT inkling_path_device STREQUAL "CPU")
                llama_test(test-llama-archs NAME test-inkling-prefix-${inkling_path_name}-${inkling_case}
                    ARGS --inkling-path far/${inkling_case} --inkling-path-device ${inkling_path_device} -s 1234)
                set_tests_properties(test-inkling-prefix-${inkling_path_name}-${inkling_case} PROPERTIES ENVIRONMENT
                    "GGML_CUDA_FATTN_REL_RUNS_SHARED=1;GGML_CUDA_FATTN_REL_RUNS_GROUP=3")
            endif()
        endforeach()
        # sliding-window direct ring runs on q8_0 SWA cache against mask legacy (including drafted verify groups)
        foreach(inkling_case runs served drafted)
            set(inkling_ring_test test-inkling-ring-${inkling_path_name}-${inkling_case})
            llama_test(test-llama-archs NAME ${inkling_ring_test}
                ARGS --inkling-path ring/${inkling_case} --inkling-path-device ${inkling_path_device} -s 1234)
            set_tests_properties(${inkling_ring_test} PROPERTIES ENVIRONMENT "GGML_CUDA_FATTN_REL_DECODE_ONE_BYTE=1")
        endforeach()
        # and on a q8_0 or f8_e4m3 global cache (GGML_CUDA_FATTN_REL_DECODE_ONE_BYTE)
        foreach(inkling_kv q8_0 e4m3)
            foreach(inkling_case runs served)
                set(inkling_far_test test-inkling-far-${inkling_path_name}-${inkling_kv}-${inkling_case})
                llama_test(test-llama-archs NAME ${inkling_far_test}
                    ARGS --inkling-path far-${inkling_kv}/${inkling_case} --inkling-path-device ${inkling_path_device} -s 1234)
                set_tests_properties(${inkling_far_test} PROPERTIES ENVIRONMENT "GGML_CUDA_FATTN_REL_DECODE_ONE_BYTE=1")
                if (NOT inkling_path_device STREQUAL "CPU")
                    set(inkling_prefix_test test-inkling-prefix-${inkling_path_name}-${inkling_kv}-${inkling_case})
                    llama_test(test-llama-archs NAME ${inkling_prefix_test}
                        ARGS --inkling-path far-${inkling_kv}/${inkling_case} --inkling-path-device ${inkling_path_device} -s 1234)
                    set_tests_properties(${inkling_prefix_test} PROPERTIES ENVIRONMENT
                        "GGML_CUDA_FATTN_REL_DECODE_ONE_BYTE=1;GGML_CUDA_FATTN_REL_RUNS_SHARED=1;GGML_CUDA_FATTN_REL_RUNS_GROUP=3")
                endif()
            endforeach()
        endforeach()
    endforeach()
    llama_build_and_test(test-speculative-mtp.cpp)
    llama_test(test-speculative-mtp NAME test-speculative-mtp-dp-cpu ARGS --dp-device CPU)

    llama_test(
        test-recurrent-state-rollback
        LABEL main
        ARGS -m "${MODEL_DIR}/qwen35-dense.gguf"
    )
    set_tests_properties(test-recurrent-state-rollback PROPERTIES
        FIXTURES_REQUIRED generate-models
    )

    llama_test(
        test-recurrent-state-rollback
        NAME test-recurrent-state-rollback-nemotron-h
        LABEL main
        ARGS -m "${MODEL_DIR}/nemotron_h-dense.gguf"
    )
    set_tests_properties(test-recurrent-state-rollback-nemotron-h PROPERTIES
        FIXTURES_REQUIRED generate-models
    )
    # --device none for the two fixtures that pick by score (experts, the pools of a sparse attention). The test holds
    # a rolled-back state to a context that decoded only the kept rows. On a card a token in a batch of another width
    # is computed by another kernel: the dense fixtures above part by its rounding, which the test bounds, and these
    # two can pick another expert or pool, which no bound holds. So on a card the suite does not hold a GLM or
    # DeepSeek V4 rollback to the kept prefix. Their copies of a state on a card (seq_cp, a shared cell's snapshot
    # planes) are run by the async and staged tests below (GLM) and by test-llama-archs, which compares two contexts
    # that did the same
    llama_test(
        test-recurrent-state-rollback
        NAME test-recurrent-state-rollback-glm5next
        LABEL main
        ARGS -m "${MODEL_DIR}/glm5next-moe.gguf" --device none
    )
    set_tests_properties(test-recurrent-state-rollback-glm5next PROPERTIES
        FIXTURES_REQUIRED generate-models
    )
    llama_test(
        test-recurrent-state-rollback
        NAME test-recurrent-state-rollback-dsv4
        LABEL main
        ARGS -m "${MODEL_DIR}/deepseek4-moe.gguf" --device none
    )
    set_tests_properties(test-recurrent-state-rollback-dsv4 PROPERTIES
        FIXTURES_REQUIRED generate-models
    )
    # a recurrent state saved and restored with the async calls; gathered: a decode takes the first rows for its
    # sequences, so the saved or restored sequence's row moves
    foreach(model glm5next-moe qwen35-dense)
        llama_test(
            test-recurrent-state-async
            NAME test-recurrent-state-async-${model}
            LABEL main
            ARGS -m "${MODEL_DIR}/${model}.gguf"
        )
        set_tests_properties(test-recurrent-state-async-${model} PROPERTIES
            FIXTURES_REQUIRED generate-models
        )
    endforeach()
    llama_test(
        test-recurrent-state-async
        NAME test-recurrent-state-async-glm5next-moe-gathered
        LABEL main
        ARGS -m "${MODEL_DIR}/glm5next-moe.gguf"
    )
    set_tests_properties(test-recurrent-state-async-glm5next-moe-gathered PROPERTIES
        FIXTURES_REQUIRED generate-models
        ENVIRONMENT "LLAMA_GRAPH_ANY_SEQS=1"
    )
    # a recurrent state copied through a spare row on the devices (LLAMA_RS_STAGE_ROWS); gathered: a decode takes the
    # first rows for its sequences, so the copied sequence's row moves
    foreach(model glm5next-moe qwen35-dense)
        llama_test(
            test-recurrent-state-staged
            NAME test-recurrent-state-staged-${model}
            LABEL main
            ARGS -m "${MODEL_DIR}/${model}.gguf"
        )
        set_tests_properties(test-recurrent-state-staged-${model} PROPERTIES
            FIXTURES_REQUIRED generate-models
        )
    endforeach()
    llama_test(
        test-recurrent-state-staged
        NAME test-recurrent-state-staged-glm5next-moe-gathered
        LABEL main
        ARGS -m "${MODEL_DIR}/glm5next-moe.gguf"
    )
    set_tests_properties(test-recurrent-state-staged-glm5next-moe-gathered PROPERTIES
        FIXTURES_REQUIRED generate-models
        ENVIRONMENT "LLAMA_GRAPH_ANY_SEQS=1"
    )
    # DeepSeek V4's memory serializes its own layout, which the test does not read
    foreach(model qwen35-dense nemotron_h-dense)
        llama_test(
            test-recurrent-rollback-rounding
            NAME test-recurrent-rollback-rounding-${model}
            LABEL main
            ARGS -m "${MODEL_DIR}/${model}.gguf"
        )
        set_tests_properties(test-recurrent-rollback-rounding-${model} PROPERTIES
            FIXTURES_REQUIRED generate-models
        )
    endforeach()
    # LLAMA_RS_FIXED_ROWS=1: a recurrent memory that keeps a row a sequence on one device, as several sequence devices
    # always do, for the architectures whose graphs write their states by row (llm_arch_rs_by_row). The tests above over
    # that memory: no decode moves another sequence's state, a copy has a row of its own at once, and a graph holds
    # neither the memory's head nor a row to clear. test-recurrent-arena holds the memory's own cases
    llama_test(test-llama-archs NAME test-glm5next-fixed-rows ARGS -a glm5next -s 1234)
    llama_test(test-llama-archs NAME test-qwen35-fixed-rows   ARGS -a qwen35 -s 1234)
    set_tests_properties(test-glm5next-fixed-rows test-qwen35-fixed-rows PROPERTIES
        ENVIRONMENT "LLAMA_RS_FIXED_ROWS=1"
    )
    # as test-glm5next above: 1,340 s beside it in a suite on a loaded host
    set_tests_properties(test-glm5next-fixed-rows PROPERTIES TIMEOUT 3600)
    llama_test(
        test-recurrent-state-rollback
        NAME test-recurrent-state-rollback-fixed-rows
        LABEL main
        ARGS -m "${MODEL_DIR}/qwen35-dense.gguf"
    )
    # --device none: as test-recurrent-state-rollback-glm5next above
    llama_test(
        test-recurrent-state-rollback
        NAME test-recurrent-state-rollback-glm5next-fixed-rows
        LABEL main
        ARGS -m "${MODEL_DIR}/glm5next-moe.gguf" --device none
    )
    set_tests_properties(test-recurrent-state-rollback-fixed-rows test-recurrent-state-rollback-glm5next-fixed-rows
        PROPERTIES
        FIXTURES_REQUIRED generate-models
        ENVIRONMENT "LLAMA_RS_FIXED_ROWS=1"
    )
    foreach(model glm5next-moe qwen35-dense)
        foreach(kind async staged)
            llama_test(
                test-recurrent-state-${kind}
                NAME test-recurrent-state-${kind}-${model}-fixed-rows
                LABEL main
                ARGS -m "${MODEL_DIR}/${model}.gguf"
            )
            set_tests_properties(test-recurrent-state-${kind}-${model}-fixed-rows PROPERTIES
                FIXTURES_REQUIRED generate-models
                ENVIRONMENT "LLAMA_RS_FIXED_ROWS=1"
            )
        endforeach()
    endforeach()
    llama_build_and_test(test-kv-mean-center.cpp)
endif()

# Target model for test-speculative-rejection-sampling
set(SPEC_TARGET_DIR "${CMAKE_CURRENT_BINARY_DIR}/test-speculative-models/target")
file(MAKE_DIRECTORY "${SPEC_TARGET_DIR}")
llama_test(
    test-llama-archs
    NAME test-generate-speculative-target
    LABEL main
    ARGS -a llama -s 999 -d 0.1 -o "${SPEC_TARGET_DIR}"
)
set_tests_properties(
    test-generate-speculative-target
    PROPERTIES FIXTURES_SETUP speculative-model
)

# To avoid recreating the model use the fixture exclude any option:
# -FA "^speculative-model$"
llama_build_and_test(test-speculative-rejection-sampling.cpp ARGS
    -m "${SPEC_TARGET_DIR}/llama-dense.gguf")
set_tests_properties(test-speculative-rejection-sampling PROPERTIES FIXTURES_REQUIRED speculative-model)

# sequences verifying their drafts side by side, over rows of a real vocabulary's size
llama_build_and_test(test-speculative-verify-batch.cpp  LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)

# Test context state handling across all architectures, using the generated dummy models
llama_test(
    test-llama-context
    LABEL "main behavior backend"
    ARGS --models "${MODEL_DIR}"
)
# abort on graph reallocations that the initial reserve should have covered
set_tests_properties(test-llama-context PROPERTIES
    FIXTURES_REQUIRED generate-models
    ENVIRONMENT "GGML_SCHED_DEBUG_REALLOC=1"
)

# Inkling 2's sequence state (attention caches and the convolution state) through the context cases, on a fixture
# whose weights spread 0.1, where the next logits follow every earlier token. A case that skips fails: these must run
set(INKLING_STATE_DIR "${CMAKE_CURRENT_BINARY_DIR}/test-models-inkling-state")
file(MAKE_DIRECTORY "${INKLING_STATE_DIR}")
llama_test(test-llama-archs NAME test-inkling-state-model LABEL main ARGS -a inkling2 -s 1 -d 0.1 -o "${INKLING_STATE_DIR}")
set_tests_properties(test-inkling-state-model PROPERTIES FIXTURES_SETUP inkling-state-model)
# a session saved before its last token and restored: the same next logits as the run that never stopped
set(INKLING_STATE_CASES_restore           "^(baseline|state_load|rt)$")
# the same session restored into another sequence, over the host and the device paths, and into scattered cells
set(INKLING_STATE_CASES_restore-other-seq "^(baseline|cp_h|cp_d|cp_h_s|cp_d_s|frag)$")
# prefix reuse: a cut the convolution snapshots cannot serve is refused, an accepted one serves the kept tokens' state
set(INKLING_STATE_CASES_prefix            "^(replay|rollback|rollback_bound|shared|seq_rm)$")
foreach(name restore restore-other-seq prefix)
    llama_test(test-llama-context NAME test-inkling-state-${name} LABEL main
        ARGS -m "${INKLING_STATE_DIR}/inkling2-moe.gguf" --test "${INKLING_STATE_CASES_${name}}")
    set_tests_properties(test-inkling-state-${name} PROPERTIES
        FIXTURES_REQUIRED inkling-state-model
        FAIL_REGULAR_EXPRESSION "skipped  *: *[1-9]"
        ENVIRONMENT "GGML_SCHED_DEBUG_REALLOC=1"
    )
endforeach()

llama_build(test-fusion.cpp)

llama_build_and_test(test-chat-peg-parser.cpp peg-parser/simple-tokenize.cpp  LABEL behavior)
llama_build_and_test(test-jinja.cpp  LABEL behavior)
llama_test(test-jinja  NAME test-jinja-py  ARGS -py  LABEL "python behavior")
llama_test_cmd(python3
    NAME test-conversion-fp8
    LABEL "python behavior"
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    ARGS -m unittest discover -s tests -p test_conversion_fp8.py -v
)
llama_test_cmd(python3
    NAME test-kv-cache-scales
    LABEL "python behavior"
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    ARGS -m unittest discover -s tests -p test_kv_cache_scales.py -v
)
llama_build_and_test(test-json-schema.cpp  LABEL behavior)
llama_build_and_test(test-chat-auto-parser.cpp  LABEL behavior  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
llama_build_and_test(test-chat-template.cpp  LABEL behavior)
# debug tool for chat template differential analysis (not registered as a test, run it manually)
llama_build(test-chat-analysis.cpp)
llama_build_and_test(test-log.cpp  LABEL behavior)
llama_build_and_test(
    test-peg-parser.cpp
    peg-parser/simple-tokenize.cpp
    peg-parser/test-basic.cpp
    peg-parser/test-gbnf-generation.cpp
    peg-parser/test-json-parser.cpp
    peg-parser/test-python-dict-parser.cpp
    peg-parser/test-unicode.cpp
    peg-parser/tests.h
    LABEL behavior
)

if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "s390x")
    set(MODEL_NAME "stories15M-q4_0.gguf")
    set(MODEL_HASH "SHA256=66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739")
else()
    set(MODEL_NAME "stories15M-be.Q4_0.gguf")
    set(MODEL_HASH "SHA256=9aec857937849d976f30397e97eb1cabb53eb9dcb1ce4611ba8247fb5f44c65d")
endif()
set(MODEL_DEST "${CMAKE_BINARY_DIR}/${MODEL_NAME}")

add_test(NAME test-download-model COMMAND ${CMAKE_COMMAND}
    -DREPO=ggml-org-dev/stories15M
    -DDEST=${MODEL_DEST}
    -DNAME=${MODEL_NAME}
    -DHASH=${MODEL_HASH}
    -P ${PROJECT_SOURCE_DIR}/cmake/download-models.cmake
)
set_tests_properties(test-download-model PROPERTIES FIXTURES_SETUP test-download-model)

# the context state cases against a real model, the sweep above uses the generated dummy ones
llama_test(test-llama-context  NAME test-llama-context-model  LABEL "model behavior backend"  ARGS -m "${MODEL_DEST}")
set_tests_properties(test-llama-context-model PROPERTIES FIXTURES_REQUIRED test-download-model)

llama_build_and_test(test-thread-safety.cpp  LABEL "main backend"  ARGS -m "${MODEL_DEST}" -ngl 99 -p "The meaning of life is" -n 128 -c 256 -ub 32 -np 4 -t 2)
set_tests_properties(test-thread-safety PROPERTIES FIXTURES_REQUIRED test-download-model)

llama_build_and_test(test-arg-parser.cpp  LABEL behavior)
# the knob registry over every knob tunables.toml declares, and the list against the source
llama_build_and_test(test-tunables.cpp  LABEL behavior  LINK ggml)
find_package(Python3 3.11 COMPONENTS Interpreter)
if (Python3_FOUND)
    add_test(NAME test-tunables-list COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/tunables.py check)
    set_property(TEST test-tunables-list PROPERTY LABELS behavior)
endif()
llama_build_and_test(test-ngram-mod.cpp)
llama_build_and_test(test-model-resolution.cpp  LABEL behavior)
# the test serves its repos from an httplib server, and the library links it privately
target_link_libraries(test-model-resolution PRIVATE cpp-httplib)

if (NOT LLAMA_SANITIZE_ADDRESS AND NOT GGML_SCHED_NO_REALLOC)
  # TODO: repair known memory leaks
  llama_build_and_test(test-opt.cpp  LABEL "main behavior backend"  LINK ggml)
endif()
if (GGML_METAL AND NOT GGML_BACKEND_DL)
    # calls the Metal backend's graph optimizer and fusion table directly
    llama_build_and_test(test-metal-graph-optimize.cpp  LABEL "main behavior backend"  LINK ggml ggml-metal)
    target_include_directories(test-metal-graph-optimize PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src ${PROJECT_SOURCE_DIR}/ggml/src/ggml-metal)
endif()
llama_test(test-backend-ops)
# The test runs every case once a device and again on the host for each, and the host's side is nearly all of its time
# (the card sits near 0%): about 1,800 s a device for its 16,000 supported cases, read on one RTX 5080 beside a 16-thread
# host at load 12 to 19, where ctest's default 1,500 s cut it 500 cases short of its end. 7,200 s holds two devices
# with as much again to spare. Without a device it ends in under a second
set_tests_properties(test-backend-ops PROPERTIES TIMEOUT 7200)
llama_build_and_test(test-fattn-sparse-q8.cpp)
# This validates the canonical reference graphs. Packed readers run in the dedicated SM120 device matrix.
add_test(NAME test-backend-q8-readers-canonical-cpu COMMAND test-backend-ops test -b CPU -o Q8_WEIGHT_READERS -j 1)
set_tests_properties(test-backend-q8-readers-canonical-cpu PROPERTIES LABELS "main")
add_test(NAME test-backend-moe-prep-cpu COMMAND test-backend-ops test -b CPU -o GLM53_MOE_PREP,GLM53_MOE_ROUTES -p n_embd=256 -j 1)
set_tests_properties(test-backend-moe-prep-cpu PROPERTIES LABELS "main")
# Inkling 2's routed experts on the GPU against the host: the expert products at 384 experts, 8 used, hidden 4,608 and
# intermediate 2,304, and the whole MoE block with its router and SITU, merged and split as the graph builds it
if (GGML_CUDA)
    add_test(NAME test-inkling-moe-mul-mat-id-cuda0 COMMAND test-backend-ops test -b CUDA0 -o MUL_MAT_ID -p "n_mats=384,n_used=8,")
    add_test(NAME test-inkling-moe-block-cuda0 COMMAND test-backend-ops test -b CUDA0 -o MOE)
    add_test(NAME test-inkling-moe-routed-rows-cuda0 COMMAND test-backend-ops test -b CUDA0 -o MUL_MAT_ID
        -p "n_mats=32,n_used=8,.*m=256,n=(17|23|32),k=512,")
    set_tests_properties(test-inkling-moe-routed-rows-cuda0 PROPERTIES LABELS "main" ENVIRONMENT "GGML_CUDA_MMVQ_KQUANT_MMID_MAX=32")
    # the expert products of the split Q3_K_M pack with their routes sorted, at each token count the sort takes
    add_test(NAME test-inkling-moe-sorted-routes-cuda0 COMMAND test-backend-ops test -b CUDA0 -o MUL_MAT_ID
        -p "n_mats=384,n_used=8,b=[01],m=(1024|4608),n=(1|23|32),k=(4608|512),")
    set_tests_properties(test-inkling-moe-sorted-routes-cuda0 PROPERTIES LABELS "main" TIMEOUT 3600
        ENVIRONMENT "GGML_CUDA_MMVQ_EXPERT_SORT_23=1;GGML_CUDA_MMVQ_EXPERT_SORT_ROWS_MIN=1;GGML_CUDA_MMVQ_EXPERT_SORT_ROWS_MAX=32")
    # the same, split evenly (a down in blocks of 32: 576 of the hidden width a card) or by expert (96 whole a card), on
    # the row kernel up to 32 rows and MMQ at a drafted step's 92
    add_test(NAME test-inkling-moe-even-split-cuda0 COMMAND test-backend-ops test -b CUDA0 -o MUL_MAT_ID,MUL_MAT_ID_PARTITION
        -p "n_mats=384,n_used=8,b=[01],(m=1152,n=(1|23|92),k=4608,|m=4608,n=(1|23|92),k=576,|m=4608,n=23,k=(4608|2304),.*first=288,local=96,)")
    set_tests_properties(test-inkling-moe-even-split-cuda0 PROPERTIES LABELS "main" TIMEOUT 3600
        ENVIRONMENT "GGML_CUDA_MMVQ_EXPERT_SORT_23=1;GGML_CUDA_MMVQ_EXPERT_SORT_ROWS_MIN=1;GGML_CUDA_MMVQ_EXPERT_SORT_ROWS_MAX=32;GGML_CUDA_MMVQ_KQUANT_MMID_MAX=32")
    # and in the routes' own order, past 16 tokens in two halves of the row kernel's grid
    add_test(NAME test-inkling-moe-even-split-rows-cuda0 COMMAND test-backend-ops test -b CUDA0 -o MUL_MAT_ID
        -p "n_mats=384,n_used=8,b=0,m=4608,n=23,k=576,")
    set_tests_properties(test-inkling-moe-even-split-rows-cuda0 PROPERTIES LABELS "main" TIMEOUT 3600
        ENVIRONMENT "GGML_CUDA_MMVQ_KQUANT_MMID_MAX=32")
    # [experts] the expert MMQ's TMA tile path (mmq-tma-tiles.cu, GGML_CUDA_MMQ_TMA): Inkling 2's Q3_K gate/up and Q4_K
    # down expert products at the served tensor-4 by-expert shapes against the host, with the switch on over the expert
    # row kernel and with it off on that row kernel itself
    add_test(NAME test-inkling-mmq-tma-cuda0 COMMAND test-backend-ops test -b CUDA0 -o MUL_MAT_ID,MUL_MAT_ID_PARTITION
        -p "(type_a=q3_K,.*n_mats=384,n_used=8,b=1,m=(1024|1152|2304),n=(3|23),k=4608,|type_a=q4_K,.*n_mats=384,n_used=8,b=0,m=4608,n=12,k=2304,)")
    set_tests_properties(test-inkling-mmq-tma-cuda0 PROPERTIES LABELS "main" TIMEOUT 3600
        ENVIRONMENT "GGML_CUDA_MMVQ_KQUANT_MMID_MAX=23;GGML_CUDA_MMQ_TMA=1")
    add_test(NAME test-inkling-mmq-tma-row-kernel-cuda0 COMMAND test-backend-ops test -b CUDA0 -o MUL_MAT_ID,MUL_MAT_ID_PARTITION
        -p "(type_a=q3_K,.*n_mats=384,n_used=8,b=1,m=(1024|1152|2304),n=(3|23),k=4608,|type_a=q4_K,.*n_mats=384,n_used=8,b=0,m=4608,n=12,k=2304,)")
    set_tests_properties(test-inkling-mmq-tma-row-kernel-cuda0 PROPERTIES LABELS "main" TIMEOUT 3600
        ENVIRONMENT "GGML_CUDA_MMVQ_KQUANT_MMID_MAX=23")
    add_test(NAME test-inkling-norm-clip-cuda0 COMMAND test-backend-ops test -b CUDA0 -o NORM_CLIP)
    # the few-row K-quant group launch (Inkling 2's Q-input products) against the host, and against each matrix's own
    # launch bit for bit
    add_test(NAME test-inkling-stack-mul-mat-cuda0 COMMAND test-backend-ops test -b CUDA0 -o MUL_MAT_GROUP,MUL_MAT_GROUP_EXACT
        -p "type=q[3-6]_K")
    add_test(NAME test-inkling-short-conv-op-cuda0 COMMAND test-backend-ops test -b CUDA0 -o SHORT_CONV)
    # the global decode on a one-byte K/V cache (q8_0, f8_e4m3) on the tile kernel against the host: 23 sessions that share
    # a prompt, bounded rows, drafts on the one-cell kernel, 2 to 8 query heads a KV head, and query-first at 256 streams
    add_test(NAME test-inkling-kv1b-ops-cuda0 COMMAND test-backend-ops test -b CUDA0 -o FLASH_ATTN_EXT
        -p "hs=128,nh=2,.*type_KV=(q8_0|f8_e4m3)")
    # the global decode on a q4_0 K/V cache behind GGML_CUDA_FATTN_REL_DECODE_Q4: every relative q4 case against the
    # host with the switch on, so the tile staging and the one-cell value reader both stand where the dense deq-dense
    # reference stood (the same cases run switch-off elsewhere in the suite and hold the row order there)
    add_test(NAME test-inkling-kv-q4-decode-tile-cuda0 COMMAND test-backend-ops test -b CUDA0 -o FLASH_ATTN_EXT
        -p "hs=128,.*type_KV=q4_0")
    set_tests_properties(test-inkling-kv-q4-decode-tile-cuda0 PROPERTIES LABELS "main"
        ENVIRONMENT "GGML_CUDA_FATTN_REL_DECODE_Q4=1")
    add_test(NAME test-inkling-kv1b-ops-query-first-cuda0 COMMAND test-backend-ops test -b CUDA0 -o FLASH_ATTN_EXT
        -p "kv=262144,nb=23,.*type_KV=(q8_0|f8_e4m3).*shared=")
    set_tests_properties(test-inkling-kv1b-ops-cuda0 PROPERTIES LABELS "main" ENVIRONMENT "GGML_CUDA_FATTN_REL_DECODE_ONE_BYTE=1")
    # the global decode by runs of cells (ggml_flash_attn_ext_set_kv_runs) against the host: a shared prefix, chunks taken
    # in turn, decode cells, a session just forked, strides, 2 to 8 query heads a KV head, 32 sessions and the served
    # shape; with query-first, and with the shared runs read once for groups of 1, 5 and 32 rows
    # (the q8_0 and f8_e4m3 cases take the runs kernels with GGML_CUDA_FATTN_REL_DECODE_ONE_BYTE)
    add_test(NAME test-inkling-far-ops-cuda0 COMMAND test-backend-ops test -b CUDA0 -o FLASH_ATTN_EXT -p "kv_runs")
    set_tests_properties(test-inkling-far-ops-cuda0 PROPERTIES ENVIRONMENT "GGML_CUDA_FATTN_REL_DECODE_ONE_BYTE=1")
    add_test(NAME test-inkling-far-ops-query-first-cuda0 COMMAND test-backend-ops test -b CUDA0 -o FLASH_ATTN_EXT -p "kv_runs")
    set_tests_properties(test-inkling-far-ops-query-first-cuda0 PROPERTIES ENVIRONMENT
        "GGML_CUDA_FATTN_REL_DECODE_ONE_BYTE=1;GGML_CUDA_FATTN_REL_QUERY_FIRST=1")
    foreach(inkling_group 1 5 32)
        add_test(NAME test-inkling-prefix-ops-g${inkling_group}-cuda0 COMMAND test-backend-ops test -b CUDA0 -o FLASH_ATTN_EXT -p "kv_runs")
        set_tests_properties(test-inkling-prefix-ops-g${inkling_group}-cuda0 PROPERTIES ENVIRONMENT
            "GGML_CUDA_FATTN_REL_DECODE_ONE_BYTE=1;GGML_CUDA_FATTN_REL_RUNS_SHARED=1;GGML_CUDA_FATTN_REL_RUNS_GROUP=${inkling_group}")
    endforeach()
    set_tests_properties(test-inkling-prefix-ops-g5-cuda0 PROPERTIES ENVIRONMENT
        "GGML_CUDA_FATTN_REL_DECODE_ONE_BYTE=1;GGML_CUDA_FATTN_REL_RUNS_SHARED=1;GGML_CUDA_FATTN_REL_RUNS_GROUP=5;GGML_CUDA_FATTN_REL_QUERY_FIRST=1")
    set_tests_properties(test-inkling-kv1b-ops-query-first-cuda0 PROPERTIES LABELS "main"
        ENVIRONMENT "GGML_CUDA_FATTN_REL_DECODE_ONE_BYTE=1;GGML_CUDA_FATTN_REL_QUERY_FIRST=1")
    set_tests_properties(test-inkling-moe-mul-mat-id-cuda0 test-inkling-moe-block-cuda0 PROPERTIES LABELS "main" TIMEOUT 3600)
    # the router and the relative-bias table on their own kernels (GGML_CUDA_F32_FEW) against the host, then the top-k
    # fusion at the router's 384 experts and, with the tensors placed so the weights and ids lie over the logits, at each
    # row count that fuses in one wide block
    add_test(NAME test-inkling-f32-router-cuda0 COMMAND test-backend-ops test -b CUDA0 -o INKLING_F32 -p "m=384,")
    add_test(NAME test-inkling-f32-bias-table-cuda0 COMMAND test-backend-ops test -b CUDA0 -o INKLING_F32 -p "m=1024,")
    set_tests_properties(test-inkling-f32-router-cuda0 test-inkling-f32-bias-table-cuda0 PROPERTIES LABELS "main"
        ENVIRONMENT "GGML_CUDA_F32_FEW=1")
    add_test(NAME test-inkling-f32-topk-moe-cuda0 COMMAND test-backend-ops test -b CUDA0 -o TOPK_MOE -p "ne=\\[384,")
    llama_build(test-inkling-f32-topk.cpp LINK ggml)
    add_test(NAME test-inkling-f32-topk-alias-cuda0 COMMAND test-inkling-f32-topk)
    add_test(NAME test-inkling-f32-topk-alias-legacy-cuda0 COMMAND test-inkling-f32-topk)
    set_tests_properties(test-inkling-f32-topk-moe-cuda0 test-inkling-f32-topk-alias-cuda0
        test-inkling-f32-topk-alias-legacy-cuda0 PROPERTIES LABELS "main")
    # the gqa_packed relative prefill tile and its unpacked kernel on the same small shapes (two KV heads, groups of 3 or
    # 6 query heads a side, depths inside and past the f32 table's extent), each against the host reference: the packed
    # one runs when GGML_CUDA_FATTN_REL_PACKED_PREFILL is set beside the f32 prefill switch
    add_test(NAME test-inkling-rel-prefill-gqa-unpacked-cuda0 COMMAND test-backend-ops test -b CUDA0 -o FLASH_ATTN_EXT
        -p "^hs=(128,nh=2,gqa=6,kv=(256,nb=33|2048,nb=35)|64,nh=2,gqa=6,kv=512,nb=64|128,nh=4,gqa=3,kv=1040,nb=31),n_stream=1,n_dist=1025,type_KV=f16,sinks=1,mask_bits=1,query_sinks=1,n_kv_max=0")
    add_test(NAME test-inkling-rel-prefill-gqa-packed-cuda0 COMMAND test-backend-ops test -b CUDA0 -o FLASH_ATTN_EXT
        -p "^hs=(128,nh=2,gqa=6,kv=(256,nb=33|2048,nb=35)|64,nh=2,gqa=6,kv=512,nb=64|128,nh=4,gqa=3,kv=1040,nb=31),n_stream=1,n_dist=1025,type_KV=f16,sinks=1,mask_bits=1,query_sinks=1,n_kv_max=0")
    set_tests_properties(test-inkling-rel-prefill-gqa-unpacked-cuda0 PROPERTIES ENVIRONMENT "GGML_CUDA_FATTN_REL_PREFILL_F32=1")
    set_tests_properties(test-inkling-rel-prefill-gqa-packed-cuda0 PROPERTIES ENVIRONMENT
        "GGML_CUDA_FATTN_REL_PREFILL_F32=1;GGML_CUDA_FATTN_REL_PACKED_PREFILL=1")
    set_tests_properties(test-inkling-f32-topk-alias-legacy-cuda0 PROPERTIES ENVIRONMENT "GGML_CUDA_TOPK_MOE_WIDE_LEGACY=1")
    # Inkling 2's routed-expert chain as fused launches (fuse-moe.cu, plan 17 C3): the chain's one launch, the gate/up
    # product with the SITU in its epilogue and the down product with the weighted sum in its epilogue. The binary is run
    # once with GGML_CUDA_MOE_FUSE=0 and once =1 and the two results must be the same bytes; each run also checks its arm
    # against a double-precision CPU port (router logits, ids, weights, sampled SITU and one weighted-sum output) and sees
    # the tensors a fused launch leaves unwritten keep their sentinel. The expert-row limit is the served 23 so that the
    # reference arm runs the row kernels the fused launches copy.
    # guarded: test-inkling-moe-fuse.cpp is not in this commit (fusions lane still uncommitted)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test-inkling-moe-fuse.cpp)
    llama_build(test-inkling-moe-fuse.cpp LINK ggml Threads::Threads)
    add_test(NAME inkling-fuse-chain-ref-cuda0 COMMAND test-inkling-moe-fuse ${CMAKE_CURRENT_BINARY_DIR}/inkling-fuse-ref.bin)
    add_test(NAME inkling-fuse-chain-fused-cuda0 COMMAND test-inkling-moe-fuse ${CMAKE_CURRENT_BINARY_DIR}/inkling-fuse-fused.bin)
    add_test(NAME inkling-fuse-chain-exact COMMAND ${CMAKE_COMMAND} -E compare_files
        ${CMAKE_CURRENT_BINARY_DIR}/inkling-fuse-ref.bin ${CMAKE_CURRENT_BINARY_DIR}/inkling-fuse-fused.bin)
    set_tests_properties(inkling-fuse-chain-ref-cuda0 PROPERTIES LABELS "main" TIMEOUT 3600
        ENVIRONMENT "GGML_CUDA_MOE_FUSE=0;GGML_CUDA_MMVQ_KQUANT_MMID_MAX=23;GGML_CUDA_F32_FEW=1")
    set_tests_properties(inkling-fuse-chain-fused-cuda0 PROPERTIES LABELS "main" TIMEOUT 3600
        ENVIRONMENT "GGML_CUDA_MOE_FUSE=1;GGML_CUDA_MMVQ_KQUANT_MMID_MAX=23;GGML_CUDA_F32_FEW=1")
    set_tests_properties(inkling-fuse-chain-exact PROPERTIES LABELS "main"
        DEPENDS "inkling-fuse-chain-ref-cuda0;inkling-fuse-chain-fused-cuda0")
    add_test(NAME inkling-fuse-chain-tma-ref-cuda0 COMMAND test-inkling-moe-fuse ${CMAKE_CURRENT_BINARY_DIR}/inkling-fuse-tma-ref.bin)
    add_test(NAME inkling-fuse-chain-tma-fused-cuda0 COMMAND test-inkling-moe-fuse ${CMAKE_CURRENT_BINARY_DIR}/inkling-fuse-tma-fused.bin)
    add_test(NAME inkling-fuse-chain-tma-exact COMMAND ${CMAKE_COMMAND} -E compare_files
        ${CMAKE_CURRENT_BINARY_DIR}/inkling-fuse-tma-ref.bin ${CMAKE_CURRENT_BINARY_DIR}/inkling-fuse-tma-fused.bin)
    set_tests_properties(inkling-fuse-chain-tma-ref-cuda0 PROPERTIES LABELS "main" TIMEOUT 3600
        ENVIRONMENT "GGML_CUDA_MOE_FUSE=0;GGML_CUDA_MMQ_TMA=1;GGML_CUDA_MMVQ_KQUANT_MMID_MAX=23;GGML_CUDA_F32_FEW=1")
    set_tests_properties(inkling-fuse-chain-tma-fused-cuda0 PROPERTIES LABELS "main" TIMEOUT 3600
        ENVIRONMENT "GGML_CUDA_MOE_FUSE=1;GGML_CUDA_MMQ_TMA=1;GGML_CUDA_MMVQ_KQUANT_MMID_MAX=23;GGML_CUDA_F32_FEW=1")
    set_tests_properties(inkling-fuse-chain-tma-exact PROPERTIES LABELS "main"
        DEPENDS "inkling-fuse-chain-tma-ref-cuda0;inkling-fuse-chain-tma-fused-cuda0")
    endif()
    # [fp8] the e4m3 hi+lo pair algebra and the block-scaled folds of GGML_CUDA_FATTN_FP8_ACCUM, on the device
    # (campaign row inkling-fp8-accumulation; the primitives live in ggml/src/ggml-cuda/fattn-rel-fp8-pair.cuh):
    # the RNE code held against CUDA's own fp8 conversion, the block encode's one ceil scale a warp, a
    # single-lane fold bit-exact, the 32- and 256-partial folds against double sums, and the one-hot per-term
    # fold; the binary skips (77) on a device below compute capability 12.0
    find_package(CUDAToolkit REQUIRED)
    # guarded: test-inkling-fp8-pair.cpp is not in this commit
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test-inkling-fp8-pair.cpp)
    set_source_files_properties(test-inkling-fp8-pair.cpp PROPERTIES LANGUAGE CUDA)
    llama_build(test-inkling-fp8-pair.cpp)
    target_include_directories(test-inkling-fp8-pair PRIVATE ${CUDAToolkit_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/ggml/src)
    # the block-scaled QMMA (kind::mxf8f6f4, ue8m0 scales) is an sm_120a feature: ptxas refuses it on
    # plain sm_120, which a configure that overrides CMAKE_CUDA_ARCHITECTURES would otherwise hand this file
    set_target_properties(test-inkling-fp8-pair PROPERTIES CUDA_ARCHITECTURES 120a)
    # line info so compute-sanitizer names the source line of a fault inside the fold or the merge pattern
    target_compile_options(test-inkling-fp8-pair PRIVATE -lineinfo)
    llama_test(test-inkling-fp8-pair NAME test-inkling-fp8-pair-cuda0)
    set_tests_properties(test-inkling-fp8-pair-cuda0 PROPERTIES LABELS "main" SKIP_RETURN_CODE 77)
    endif()
    # [fp8] the relative decode with the accumulator's pair planes: every partial writer (tile, one-cell and
    # runs kernels) stores two-term slots and the merge folds them on the block-scaled MMA; e4m3 K/V rides the
    # tile kernel so the KV type and the accumulator meet on one path, every relative case against the host
    # (the same cases run switch-off elsewhere in the suite and hold the f32 planes there)
    add_test(NAME test-inkling-kv-fp8acc-tile-cuda0 COMMAND test-backend-ops test -b CUDA0 -o FLASH_ATTN_EXT
        -p "hs=128,.*type_KV=f8_e4m3")
    set_tests_properties(test-inkling-kv-fp8acc-tile-cuda0 PROPERTIES LABELS "main" TIMEOUT 3600
        ENVIRONMENT "GGML_CUDA_FATTN_FP8_ACCUM=1;GGML_CUDA_FATTN_REL_DECODE_ONE_BYTE=1")
    # [experts] the Blackwell sm_120 MMQ rows for the K-quants (mmq-config-blackwell.cuh, campaign row
    # blackwell-kquant-mmq): the six served prefill products (384 experts, top 8, 1,024 rows, gate/up and down in
    # Q3_K/Q4_K/Q5_K) through the CUDA backend against the CPU backend on the same bytes, within the quantized
    # NMSE bound test-backend-ops allows, plus one 16-row pass that keeps the Ampere widths honest; skips with no
    # CUDA device, and it is what judges that row
    llama_build_and_test(test-inkling-mmq-kquant.cpp LINK ggml Threads::Threads)
    set_tests_properties(test-inkling-mmq-kquant PROPERTIES LABELS "main" TIMEOUT 3600)
    llama_test(test-inkling-mmq-kquant NAME test-inkling-mmq-tma-kquant-cuda0 ARGS --tma)
    set_tests_properties(test-inkling-mmq-tma-kquant-cuda0 PROPERTIES LABELS "main" TIMEOUT 3600
        ENVIRONMENT "GGML_CUDA_MMVQ_KQUANT_MMID_MAX=23;GGML_CUDA_MMQ_TMA=1")
    # Blackwell K-quant staged MMQ (kquant-staged-kernel) and 23-row TMA MoE parity across 0/3/8 repeated experts and 1024/1536/4608 shards
    llama_build(test-inkling-kquant-tile.cpp LINK ggml Threads::Threads)
    add_test(NAME test-inkling-kquant-tile-staged-cuda0 COMMAND test-inkling-kquant-tile ${CMAKE_CURRENT_BINARY_DIR}/inkling-kquant-tile-staged.bin)
    add_test(NAME test-inkling-kquant-tile-legacy-cuda0 COMMAND test-inkling-kquant-tile ${CMAKE_CURRENT_BINARY_DIR}/inkling-kquant-tile-legacy.bin)
    add_test(NAME test-inkling-kquant-tile-exact COMMAND ${CMAKE_COMMAND} -E compare_files
        ${CMAKE_CURRENT_BINARY_DIR}/inkling-kquant-tile-staged.bin ${CMAKE_CURRENT_BINARY_DIR}/inkling-kquant-tile-legacy.bin)
    set_tests_properties(test-inkling-kquant-tile-staged-cuda0 PROPERTIES LABELS "main" TIMEOUT 3600
        ENVIRONMENT "GGML_CUDA_MMQ_KQUANT_STAGED_LEGACY=0")
    set_tests_properties(test-inkling-kquant-tile-legacy-cuda0 PROPERTIES LABELS "main" TIMEOUT 3600
        ENVIRONMENT "GGML_CUDA_MMQ_KQUANT_STAGED_LEGACY=1")
    set_tests_properties(test-inkling-kquant-tile-exact PROPERTIES LABELS "main"
        DEPENDS "test-inkling-kquant-tile-staged-cuda0;test-inkling-kquant-tile-legacy-cuda0")
    llama_test(test-inkling-kquant-tile NAME test-inkling-moe-tma-23row-cuda0 ARGS --tma-23)
    set_tests_properties(test-inkling-moe-tma-23row-cuda0 PROPERTIES LABELS "main" TIMEOUT 3600
        ENVIRONMENT "GGML_CUDA_MMVQ_KQUANT_MMID_MAX=23;GGML_CUDA_MMQ_TMA=1")
    # Blackwell K-quant-scoped J rule (kquant-j-rule): pins GLM IQ3_XXS selections unchanged and verifies live sm_120 dispatch
    set_source_files_properties(test-inkling-kquant-jrule.cpp PROPERTIES LANGUAGE CUDA)
    llama_build(test-inkling-kquant-jrule.cpp LINK ggml)
    target_include_directories(test-inkling-kquant-jrule PRIVATE ${CUDAToolkit_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/ggml/src ${PROJECT_SOURCE_DIR}/ggml/src/ggml-cuda)
    add_test(NAME test-inkling-kquant-jrule-cuda0 COMMAND test-inkling-kquant-jrule ${CMAKE_CURRENT_BINARY_DIR}/inkling-kquant-jrule.jsonl)
    add_test(NAME test-inkling-kquant-jrule-legacy-cuda0 COMMAND test-inkling-kquant-jrule ${CMAKE_CURRENT_BINARY_DIR}/inkling-kquant-jrule-legacy.jsonl)
    set_tests_properties(test-inkling-kquant-jrule-cuda0 PROPERTIES LABELS "main"
        ENVIRONMENT "GGML_CUDA_MMQ_KQUANT_SPREAD_LEGACY=0")
    set_tests_properties(test-inkling-kquant-jrule-legacy-cuda0 PROPERTIES LABELS "main"
        ENVIRONMENT "GGML_CUDA_MMQ_KQUANT_SPREAD_LEGACY=1")
endif()

llama_build(test-backend-ops.cpp LINK ggml Threads::Threads)

llama_build_and_test(test-backend-tensor.cpp LINK ggml)
llama_build_and_test(test-kv-restore-runs.cpp LABEL "model" ARGS -m "${MODEL_DEST}")
set_tests_properties(test-kv-restore-runs PROPERTIES FIXTURES_REQUIRED test-download-model)
llama_build_and_test(test-sleep.cpp LINK ggml)
llama_build_and_test(test-backend-sched.cpp LINK ggml)

if (BUILD_SHARED_LIBS)
    llama_build_and_test(test-llama-mmap.cpp ${PROJECT_SOURCE_DIR}/src/llama-impl.cpp ${PROJECT_SOURCE_DIR}/src/llama-mmap.cpp LABEL behavior)
    target_compile_definitions(test-llama-mmap PRIVATE LLAMA_BUILD)
else()
    llama_build_and_test(test-llama-mmap.cpp LABEL behavior)
endif()

llama_build_and_test(test-tiled-mulmat.cpp  LABEL "main behavior backend"  LINK ggml)
llama_build_and_test(test-model-load-cancel.cpp LABEL "model behavior")
llama_build_and_test(test-autorelease.cpp       LABEL "model behavior")
llama_build_and_test(test-backend-sampler.cpp   LABEL "model behavior backend")

if (APPLE)
    llama_build(test-rset-release.cpp)
    llama_build_and_test(test-host-buffer-release.cpp LINK ggml)
endif()

if (NOT GGML_BACKEND_DL AND GGML_CPU)
    # these tests use the backends directly and cannot be built with dynamic loading
    llama_build_and_test(test-barrier.cpp  LABEL "main behavior"  LINK ggml)
    llama_build_and_test(test-quantize-fns.cpp  LABEL "main behavior"  LINK ggml)
    llama_build_and_test(test-quantize-perf.cpp  LABEL "main behavior backend"  LINK ggml)
    llama_build_and_test(test-rope.cpp  LABEL "main behavior backend"  LINK ggml)
    llama_build_and_test(test-col2im-1d.cpp  LABEL "main behavior backend"  LINK ggml)
    llama_build_and_test(test-conv-1d.cpp  LABEL "main behavior backend"  LINK ggml)
    llama_build_and_test(test-gelu-quick.cpp  LABEL "main behavior backend"  LINK ggml)
    llama_build_and_test(test-cpu-repack-buffer.cpp)
    llama_build_and_test(test-cuda-graph-key.cpp)
    llama_build_and_test(test-cuda-graph-src-type.cpp)
    llama_test(test-cuda-graph-src-type NAME test-cuda-graph-moe-cpu ARGS --moe-cpu)
    llama_build_and_test(test-cuda-graph-cap.cpp)
    llama_test(test-cuda-graph-cap NAME test-cuda-graph-cap-nodes ARGS nodes)
    llama_test(test-cuda-graph-cap NAME test-cuda-graph-cap-nodes-legacy ARGS nodes-legacy)
    llama_test(test-cuda-graph-cap NAME test-cuda-graph-cap-idle ARGS idle)
    llama_test(test-cuda-graph-cap NAME test-cuda-graph-cap-no-idle ARGS no-idle)
    llama_test(test-cuda-graph-cap NAME test-cuda-graph-cap-no-mem-stats ARGS no-mem-stats)
    if (GGML_CUDA)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        set_source_files_properties(test-cuda-graph-cap.cpp PROPERTIES LANGUAGE CUDA)
        target_include_directories(test-cuda-graph-cap PRIVATE ${CUDAToolkit_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/ggml/src)
    endif()
    if (GGML_CUDA_GRAPHS)
        target_compile_definitions(test-cuda-graph-cap PRIVATE GGML_CUDA_USE_GRAPHS)
    endif()
    llama_build_and_test(test-cuda-gdn-dp-store.cpp)
    llama_build_and_test(test-backend-meta-views.cpp)
    # with the split decisions of Inkling 2, which load a model with no data over the meta backend
    llama_build_and_test(test-backend-meta-split.cpp test-backend-meta-split-inkling.cpp llama-archs-gguf.cpp)
    target_include_directories(test-backend-meta-split PRIVATE ${PROJECT_SOURCE_DIR}/src)
    if (GGML_CUDA)
        llama_test(test-backend-meta-split NAME test-backend-meta-permuted-read ARGS --permuted-read)
        # a step's 96 reductions through host memory, with the host waiting two back and under GGML_CUDA_AR_SLOT_NO_SYNC, bit for bit
        llama_test(test-backend-meta-split NAME test-backend-meta-reduction-queue ARGS --reduction-queue)
        set_tests_properties(test-backend-meta-reduction-queue PROPERTIES SKIP_RETURN_CODE 77)
        # inkling-execution-ar-capture: the copy-engine route captured whole against segments only, bit for bit,
        # on real ranks (skips where fewer than two distinct GPUs serve the box); its fallback companion proves a
        # refused capture lands back on the live route and completes
        llama_test(test-backend-meta-split NAME test-inkling-ar-capture ARGS --ar-capture)
        set_tests_properties(test-inkling-ar-capture PROPERTIES SKIP_RETURN_CODE 77)
        llama_test(test-backend-meta-split NAME test-inkling-ar-capture-fallback ARGS --ar-fallback)
        set_tests_properties(test-inkling-ar-capture-fallback PROPERTIES SKIP_RETURN_CODE 77)
    endif()
    add_test(NAME test-backend-meta-candidates COMMAND test-backend-meta-split --candidates)
    if (GGML_CPU)
        add_test(NAME test-backend-meta-candidates-cpu COMMAND test-backend-meta-split --candidates-cpu)
        add_test(NAME test-backend-meta-split-cache-cpu COMMAND test-backend-meta-split --split-cache-cpu)
        add_test(NAME test-backend-meta-split-cycle-cpu COMMAND test-backend-meta-split --split-cycle-cpu)
        add_test(NAME test-backend-meta-permuted-read-cpu COMMAND test-backend-meta-split --permuted-read-cpu)
        add_test(NAME test-backend-meta-steps-kept-cpu COMMAND test-backend-meta-split --steps-kept-cpu)
        set_tests_properties(test-backend-meta-split-cache-cpu test-backend-meta-split-cycle-cpu PROPERTIES TIMEOUT 60)
    endif()
    llama_build_and_test(test-backend-meta-capture.cpp)
    llama_test(test-backend-meta-capture NAME test-backend-meta-capture-transition ARGS --split-transition)
    set_tests_properties(test-backend-meta-capture-transition PROPERTIES ENVIRONMENT "GGML_GRAPH_TRACE=1" SKIP_RETURN_CODE 77)
    llama_build_and_test(test-backend-meta-sourceless.cpp)
    llama_build_and_test(test-backend-meta-sampler.cpp)
    llama_build_and_test(test-backend-meta-capacity.cpp)
    # each rank's compute buffer sized from its slice and not the merged tensor: both sizings run,
    # one mode a process
    llama_build_and_test(test-backend-meta-reserve.cpp)
    llama_test(test-backend-meta-reserve NAME test-backend-meta-reserve-merged ARGS --legacy)
    llama_build_and_test(test-backend-meta-allreduce-copy.cpp)
    if (GGML_CUDA)
        # four ranks on the first GPUs, each reduction past the LL to owners and gathered from them
        llama_test(test-backend-meta-allreduce-copy NAME test-backend-meta-allreduce-split-reduce ARGS --logical 4)
        # with no card to place a rank on the test says so and returns 77
        set_tests_properties(test-backend-meta-allreduce-split-reduce PROPERTIES ENVIRONMENT "GGML_CUDA_AR_SPLIT_REDUCE=1" SKIP_RETURN_CODE 77)
    endif()
    llama_build_and_test(test-pq2-mma-device-memory.cpp)
llama_build_and_test(test-ptq1_0-element-map.cpp)
llama_build_and_test(test-ptq1_0-cuda-dot.cpp)
endif()

# libmtmd
llama_build_and_test(test-mtmd-c-api.c  LABEL behavior  LINK mtmd)

llama_build_and_test(test-mtmd-impl.cpp  LABEL behavior  LINK mtmd llama-common)

# GGUF model data fetcher library for tests that need real model metadata
# Only compile when cpp-httplib has SSL support (CPPHTTPLIB_OPENSSL_SUPPORT)
if (TARGET cpp-httplib)
    get_target_property(_cpp_httplib_defs cpp-httplib INTERFACE_COMPILE_DEFINITIONS)
    if (_cpp_httplib_defs MATCHES "CPPHTTPLIB_OPENSSL_SUPPORT")
        add_library(gguf-model-data STATIC gguf-model-data.cpp)
        target_link_libraries(gguf-model-data PRIVATE llama-common cpp-httplib)
        target_include_directories(gguf-model-data PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

        add_executable(test-gguf-model-data test-gguf-model-data.cpp)
        target_link_libraries(test-gguf-model-data PRIVATE gguf-model-data llama-common)
        llama_test(test-gguf-model-data  LABEL "model behavior")

        # test-quant-type-selection requires gguf-model-data for remote model metadata
        llama_build_and_test(test-quant-type-selection.cpp  LABEL "model behavior"  LINK llama)
        target_link_libraries(test-quant-type-selection PRIVATE gguf-model-data)
        target_compile_definitions(test-quant-type-selection PRIVATE
            SNAPSHOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/snapshots")
    endif()
endif()

# dummy executable - not installed
get_filename_component(TEST_TARGET test-c.c NAME_WE)
add_executable(${TEST_TARGET} test-c.c)
target_link_libraries(${TEST_TARGET} PRIVATE llama)

if (NOT LLAMA_USE_SYSTEM_GGML)
    # Needs non-public ggml-impl.h
    llama_build_and_test(test-gguf.cpp  LABEL behavior  LINK ggml)

    # Needs non-public ggml{,-backend}-impl.h
    llama_build_and_test(test-alloc.cpp  LABEL "behavior backend"  LINK ggml ggml-base)
endif()

llama_build(test-export-graph-ops.cpp)
if (TARGET gguf-model-data)
    target_link_libraries(test-export-graph-ops PRIVATE gguf-model-data)
    target_compile_definitions(test-export-graph-ops PRIVATE LLAMA_HF_FETCH)
endif()
