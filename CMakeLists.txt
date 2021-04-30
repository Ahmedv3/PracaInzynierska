cmake_minimum_required(VERSION 2.8)
project(inz)

find_package( OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( inz inz.cpp )
target_link_libraries( inz ${OpenCV_LIBS})
