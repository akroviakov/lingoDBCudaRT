set(COMMON_COMPILE_OPTIONS_CUDA -O3 -g --ptxas-options=-v -Xcompiler -rdynamic -lineinfo -Wall)

add_executable(Bench Bench.cu)
target_compile_features(Bench PUBLIC ${COMMON_COMPILE_FEATURES})
set_property(TARGET Bench PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET Bench PROPERTY CUDA_ARCHITECTURES 75)
target_compile_definitions(Bench ${COMMON_DEFINITIONS})
target_compile_options(Bench PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${COMMON_COMPILE_OPTIONS_CXX}>)
target_compile_options(Bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${COMMON_COMPILE_OPTIONS_CUDA}>)
target_include_directories(Bench PRIVATE ${COMMON_INCLUDE_DIRECTORIES})
target_link_libraries(Bench PRIVATE cuda )
if(USE_GALLATIN)
    target_link_libraries(Bench PRIVATE gallatin )
endif()
