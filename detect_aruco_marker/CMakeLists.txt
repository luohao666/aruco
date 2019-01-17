cmake_minimum_required( VERSION 2.8 )
project( detect_aruco_marker )

# 添加c++ 11标准支持
set( CMAKE_CXX_FLAGS "-std=c++11" )

# 寻找OpenCV库
find_package( OpenCV REQUIRED )
# 添加头文件
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( detect_markers detect_markers.cpp )
# 链接OpenCV库
target_link_libraries( detect_markers ${OpenCV_LIBS} )

add_executable( detect_markers_new detect_markers_new.cpp )
# 链接OpenCV库
target_link_libraries( detect_markers_new ${OpenCV_LIBS} )

add_executable( detect_markers_image detect_markers_image.cpp )
# 链接OpenCV库
target_link_libraries( detect_markers_image ${OpenCV_LIBS} )
