CMAKE_MINIMUM_REQUIRED(VERSION 2.8.7)
IF(POLICY CMP0046)
  CMAKE_POLICY(SET CMP0046 NEW)
ENDIF()
IF(POLICY CMP0054)
  CMAKE_POLICY(SET CMP0054 NEW)
ENDIF()

#find package
FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(CUDA REQUIRED)

#define platform4
ADD_DEFINITIONS("-D__X86_64__")

#c14
ADD_DEFINITIONS("-std=c++14")
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++14;
-gencode arch=compute_30,code=sm_30;
-gencode arch=compute_32,code=sm_32;
-gencode arch=compute_53,code=sm_53;
-gencode arch=compute_61,code=sm_61;
-gencode arch=compute_62,code=sm_62;
-gencode arch=compute_70,code=sm_70;)

#add src
FILE(GLOB_RECURSE SRCS 
	"${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu" 
	"${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

#add include
SET(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
LIST(APPEND INCLUDE_DIR_LISTS
	${OpenCV_INCLUDE_DIRS}
	${CUDA_INCLUDE_DIRS}
	${INCLUDE_DIR}/TensorRT-5.0.2.6)

INCLUDE_DIRECTORIES(${INCLUDE_DIR_LISTS})

#add library
SET(LIBS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib")
LINK_DIRECTORIES(${LIBS_DIR})
LINK_DIRECTORIES(${LIBS_DIR}/TensorRT-5.0.2.6)

LIST(APPEND TENSORRT_LIBS
	nvcaffe_parser
	nvonnxparser
	nvinfer
	nvinfer_plugin
	nvparsers)

LIST(APPEND LIB_LIST
	${CUDA_LIBRARIES}
	${TENSORRT_LIBS}
	pthread)

#generate binary
SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)
CUDA_ADD_EXECUTABLE(tensorrt-custom ${SRCS})
TARGET_LINK_LIBRARIES(tensorrt-custom ${LIB_LIST})


