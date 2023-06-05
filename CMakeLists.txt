cmake_minimum_required(VERSION 3.10)
project(yolov5)

set(CMAKE_CXX_STANDARD 14)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
SET(CMAKE_BUILD_TYPE "DEBUG")
# SET(CMAKE_BUILD_TYPE "RELEASE")
message("cur dir: ${PROJECT_SOURCE_DIR}")

# use ffmpeg specialized by sophon
set(FFMPEG_DIR /opt/sophon/sophon-ffmpeg_0.6.0)
set(FFMPEG_INCLUDE_DIRS ${FFMPEG_DIR}/include)
set(FFMPEG_LIB_DIRS ${FFMPEG_DIR}/lib)
set(FFMPEG_LIBS avcodec avdevice avfilter avformat avutil swresample swscale)
link_directories(${FFMPEG_LIB_DIRS})

# use opencv specialized by sophon
set(OPENCV_DIR /opt/sophon/sophon-opencv_0.6.0)
set(OpenCV_INCLUDE_DIRS ${OPENCV_DIR}/include/opencv4)
set(OpenCV_LIB_DIRS ${OPENCV_DIR}/lib)
set(OPENCV_LIBS opencv_imgproc opencv_core opencv_highgui opencv_imgcodecs opencv_videoio)
link_directories(${OpenCV_LIB_DIRS})

# use opencv
# find_package(OpenCV REQUIRED)
# set(OPENCV_LIBS opencv_imgproc opencv_core opencv_highgui opencv_imgcodecs opencv_videoio)

# use sophgo
find_package(libsophon REQUIRED)
set(BM_LIBS bmlib bmrt bmcv)
set(JPU_LIBS bmjpuapi bmjpulite)
link_directories(${LIBSOPHON_LIB_DIRS})

# use openmp
find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# aux_source_directory(. SRC_FILES)
aux_source_directory(./engines ENGINE_SRC_FILES)
add_executable(main main.cpp bmrun_helper.h bmrun_helper.cpp det_structs.h ${ENGINE_SRC_FILES})
target_include_directories(main PUBLIC ${OpenCV_INCLUDE_DIRS} ${LIBSOPHON_INCLUDE_DIRS} ${FFMPEG_INCLUDE_DIRS})
target_link_libraries(main ${BM_LIBS} ${OPENCV_LIBS}  ${JPU_LIBS} ${FFMPEG_LIBS})
message(${PROJECT_SOURCE_DIR})
message(${OpenCV_INCLUDE_DIRS})
message(${LIBSOPHON_INCLUDE_DIRS})
message(${FFMPEG_INCLUDE_DIRS})