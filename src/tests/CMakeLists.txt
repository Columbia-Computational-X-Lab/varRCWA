include_directories(
    ${MKL_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${TBB_INCLUDE_DIRS}
    ${MAGMA_INCLUDE_DIR}
)

if (BUILD_GPU)
  add_exe(test_gpu
      test_gpu.cpp
      ${RCWA_SRC_FILES}
      ${GDSTK_SRC_FILES}
      ${CORE_SRC_FILES}
      ${GPU_SRC_FILES}
    LINK_LIBRARIES
      ${MKL_LIBRARIES}
      ${TBB_LIBRARIES}
      ${MAGMA_LIBRARIES}
      m
      stdc++
      cusolver
      cublas
    OUT_DIR
      ${ROOT_TESTS_DIR})
  set_property(TARGET test_gpu PROPERTY CXX_STANDARD 17)

  add_exe(test_metasurface_gpu
      test_metasurface_gpu.cpp
      ${RCWA_SRC_FILES}
      ${GDSTK_SRC_FILES}
      ${CORE_SRC_FILES}
      ${GPU_SRC_FILES}
    LINK_LIBRARIES
      ${MKL_LIBRARIES}
      ${TBB_LIBRARIES}
      ${MAGMA_LIBRARIES}
      m
      stdc++
      cusolver
      cublas
    OUT_DIR
      ${ROOT_TESTS_DIR})
  set_property(TARGET test_metasurface_gpu PROPERTY CXX_STANDARD 17)

  add_exe(test_oe_model_gpu
      test_oe_model_gpu.cpp
      ${RCWA_SRC_FILES}
      ${GDSTK_SRC_FILES}
      ${CORE_SRC_FILES}
      ${GPU_SRC_FILES}
    LINK_LIBRARIES
      ${MKL_LIBRARIES}
      ${TBB_LIBRARIES}
      ${MAGMA_LIBRARIES}
      m
      stdc++
      cusolver
      cublas
    OUT_DIR
      ${ROOT_TESTS_DIR})
  set_property(TARGET test_oe_model_gpu PROPERTY CXX_STANDARD 17)

  add_exe(test_gds_gpu
      test_gds_gpu.cpp
      ${RCWA_SRC_FILES}
      ${GDSTK_SRC_FILES}
      ${CORE_SRC_FILES}
      ${GPU_SRC_FILES}
    LINK_LIBRARIES
      ${MKL_LIBRARIES}
      ${TBB_LIBRARIES}
      ${MAGMA_LIBRARIES}
      m
      stdc++
      cusolver
      cublas
    OUT_DIR
      ${ROOT_TESTS_DIR})
  set_property(TARGET test_gds_gpu PROPERTY CXX_STANDARD 17)
endif()

add_exe(test_gds_cpu
    test_gds_cpu.cpp
    ${RCWA_SRC_FILES}
    ${GDSTK_SRC_FILES}
    ${ROOT_SRC_DIR}/core/LayerSampler.cpp
    ${ROOT_SRC_DIR}/core/VarLayerSampler.cpp
    ${ROOT_SRC_DIR}/core/RedhefferIntegrator.cpp
    ${ROOT_SRC_DIR}/core/RCWAIntegrator.cpp
    ${ROOT_SRC_DIR}/core/DifferentialIntegrator.cpp
    ${ROOT_SRC_DIR}/core/GDSIISampler.cpp
    ${ROOT_SRC_DIR}/core/Material.cpp
  LINK_LIBRARIES
    ${MKL_LIBRARIES}
    ${TBB_LIBRARIES}
    m
    stdc++
  OUT_DIR
    ${ROOT_TESTS_DIR})
set_property(TARGET test_gds_cpu PROPERTY CXX_STANDARD 17)

add_exe(test_cpu
    test_cpu.cpp
    ${RCWA_SRC_FILES}
    ${ROOT_SRC_DIR}/core/LayerSampler.cpp
    ${ROOT_SRC_DIR}/core/VarLayerSampler.cpp
    ${ROOT_SRC_DIR}/core/RedhefferIntegrator.cpp
    ${ROOT_SRC_DIR}/core/RCWAIntegrator.cpp
    ${ROOT_SRC_DIR}/core/DifferentialIntegrator.cpp
  LINK_LIBRARIES
    ${MKL_LIBRARIES}
    ${TBB_LIBRARIES}
    m
    stdc++
  OUT_DIR
    ${ROOT_TESTS_DIR})
set_property(TARGET test_cpu PROPERTY CXX_STANDARD 17)

add_exe(test_metasurface_cpu
    test_metasurface_cpu.cpp
    ${RCWA_SRC_FILES}
    ${ROOT_SRC_DIR}/core/LayerSampler.cpp
    ${ROOT_SRC_DIR}/core/VarLayerSampler.cpp
    ${ROOT_SRC_DIR}/core/RedhefferIntegrator.cpp
    ${ROOT_SRC_DIR}/core/RCWAIntegrator.cpp
    ${ROOT_SRC_DIR}/core/DifferentialIntegrator.cpp
  LINK_LIBRARIES
    ${MKL_LIBRARIES}
    ${TBB_LIBRARIES}
    m
    stdc++
  OUT_DIR
    ${ROOT_TESTS_DIR})
set_property(TARGET test_metasurface_cpu PROPERTY CXX_STANDARD 17)

add_exe(test_cpu_diff
    test_cpu_differential_method.cpp
    ${RCWA_SRC_FILES}
    ${ROOT_SRC_DIR}/core/LayerSampler.cpp
    ${ROOT_SRC_DIR}/core/VarLayerSampler.cpp
    ${ROOT_SRC_DIR}/core/RedhefferIntegrator.cpp
    ${ROOT_SRC_DIR}/core/RCWAIntegrator.cpp
    ${ROOT_SRC_DIR}/core/DifferentialIntegrator.cpp
  LINK_LIBRARIES
    ${MKL_LIBRARIES}
    ${TBB_LIBRARIES}
    m
    stdc++
  OUT_DIR
    ${ROOT_TESTS_DIR})
set_property(TARGET test_cpu_diff PROPERTY CXX_STANDARD 17)