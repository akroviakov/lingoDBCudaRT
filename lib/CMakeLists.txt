set(COMMON_COMPILE_OPTIONS_CUDA -O3 -g --ptxas-options=-v -Xcompiler -rdynamic -lineinfo -Wall)

add_library(rt_lib SHARED FlexibleBuffer.cu)

target_compile_features(rt_lib PUBLIC ${COMMON_COMPILE_FEATURES})
set_property(TARGET rt_lib PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET rt_lib PROPERTY CUDA_ARCHITECTURES 75)
target_compile_definitions(rt_lib ${COMMON_DEFINITIONS})
target_compile_options(rt_lib PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${COMMON_COMPILE_OPTIONS_CXX}>)
target_compile_options(rt_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${COMMON_COMPILE_OPTIONS_CUDA}>)
target_include_directories(rt_lib PRIVATE ${COMMON_INCLUDE_DIRECTORIES})
target_link_libraries(rt_lib PUBLIC gallatin cuda)