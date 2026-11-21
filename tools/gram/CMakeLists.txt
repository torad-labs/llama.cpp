# llama-gram needs the card's own BLAS for its binary32 SYRK, so it is built only with the CUDA backend.
if (NOT GGML_CUDA)
    message(STATUS "llama-gram: skipped, it needs GGML_CUDA")
    return()
endif()

find_package(CUDAToolkit REQUIRED)

set(TARGET llama-gram)
add_executable(${TARGET} gram.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common llama CUDA::cudart CUDA::cublas ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
