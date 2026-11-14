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

llama_build(test-llama-archs.cpp)
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

llama_build(test-backend-ops.cpp LINK ggml Threads::Threads)

llama_build_and_test(test-backend-tensor.cpp LINK ggml)
llama_build_and_test(test-kv-restore-runs.cpp LABEL "model" ARGS -m "${MODEL_DEST}")
set_tests_properties(test-kv-restore-runs PROPERTIES FIXTURES_REQUIRED test-download-model)

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
    llama_build_and_test(test-backend-meta-split.cpp)
    if (GGML_CUDA)
        llama_test(test-backend-meta-split NAME test-backend-meta-permuted-read ARGS --permuted-read)
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

llama_build_and_test(test-mtmd-impl.cpp  LABEL behavior  LINK mtmd)

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
