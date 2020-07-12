CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(grid_controller)


# Set the output directory for the build executables and libraries
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${grid_controller_SOURCE_DIR}/lib)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${grid_controller_SOURCE_DIR}/lib)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${grid_controller_SOURCE_DIR}/bin)

# Set search directory for looking for our custom CMake scripts to
# look for Eigen3
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${grid_controller_SOURCE_DIR}/../cmake_modules)

ADD_DEFINITIONS(-DUNIX)
MESSAGE(STATUS "Compiling on Unix")

SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG") 
SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -DNDEBUG") 
SET(CMAKE_BUILD_TYPE Release)

# Find Eigen3
# FIND_PACKAGE(Eigen3 REQUIRED)
# INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})
# INCLUDE_DIRECTORIES(/usr/include/eigen3)

# Find Opencv
FIND_PACKAGE(OpenCV  REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

# Set up the top-level include directories
INCLUDE_DIRECTORIES(${grid_controller_SOURCE_DIR/src})

ADD_SUBDIRECTORY(src)
