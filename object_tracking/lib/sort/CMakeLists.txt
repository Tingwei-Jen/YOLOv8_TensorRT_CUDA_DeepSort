cmake_minimum_required(VERSION 3.10)

# Project name and version
project(sort_lib)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Find Eigen3 package
find_package(Eigen3 REQUIRED)

# Add the executable
add_library (${PROJECT_NAME} STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/track.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kalman_filter.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/iou_matching.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nn_matching.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hungarian.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/linear_assignment.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tracker.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/detector.cpp)

# Include directories for OpenCV and Eigen
target_include_directories(${PROJECT_NAME} 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include 
    PRIVATE
        ${OpenCV_INCLUDE_DIRS} 
        ${EIGEN3_INCLUDE_DIR})

# Link the OpenCV libraries
target_link_libraries(${PROJECT_NAME} 
    PRIVATE 
        ${OpenCV_LIBS})