cmake_minimum_required(VERSION 3.10)

project(rkface_insight)

set(ARM_CROSS_COMPILE_TOOLCHAIN /opt/toolchain/arm-rockchip830-linux-uclibcgnueabihf)
set(CMAKE_C_COMPILER ${ARM_CROSS_COMPILE_TOOLCHAIN}/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc)
set(CMAKE_CXX_COMPILER ${ARM_CROSS_COMPILE_TOOLCHAIN}/bin/arm-rockchip830-linux-uclibcgnueabihf-g++)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/rkaiq
    ${PROJECT_SOURCE_DIR}/include/rkaiq/common
    ${PROJECT_SOURCE_DIR}/include/rkaiq/uAPI2
    ${PROJECT_SOURCE_DIR}/include/rkaiq/xcore
    ${PROJECT_SOURCE_DIR}/include/rkaiq/algos
    ${PROJECT_SOURCE_DIR}/include/rkaiq/iq_parser
    ${PROJECT_SOURCE_DIR}/include/rkaiq/iq_parser_v2
    ${PROJECT_SOURCE_DIR}/include/rknn
    ${PROJECT_SOURCE_DIR}/include/opencv4
)

link_directories(
    ${PROJECT_SOURCE_DIR}/lib   
)

add_executable(${PROJECT_NAME}
    src/main.cc
    src/luckfox_mpi.cc
    src/font.h
)

target_link_libraries(${PROJECT_NAME} 
    #  rknnmrt
    InspireFace rockit rockchip_mpp rtsp rkaiq rockiva rga sample_comm
    opencv_imgproc opencv_features2d opencv_highgui opencv_photo opencv_video opencv_core 
)