# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/luohao/aruco/detect_aruco_marker

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/luohao/aruco/detect_aruco_marker/build

# Include any dependencies generated for this target.
include CMakeFiles/detect_markers.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/detect_markers.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/detect_markers.dir/flags.make

CMakeFiles/detect_markers.dir/detect_markers.cpp.o: CMakeFiles/detect_markers.dir/flags.make
CMakeFiles/detect_markers.dir/detect_markers.cpp.o: ../detect_markers.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/luohao/aruco/detect_aruco_marker/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/detect_markers.dir/detect_markers.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/detect_markers.dir/detect_markers.cpp.o -c /home/luohao/aruco/detect_aruco_marker/detect_markers.cpp

CMakeFiles/detect_markers.dir/detect_markers.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/detect_markers.dir/detect_markers.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/luohao/aruco/detect_aruco_marker/detect_markers.cpp > CMakeFiles/detect_markers.dir/detect_markers.cpp.i

CMakeFiles/detect_markers.dir/detect_markers.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/detect_markers.dir/detect_markers.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/luohao/aruco/detect_aruco_marker/detect_markers.cpp -o CMakeFiles/detect_markers.dir/detect_markers.cpp.s

CMakeFiles/detect_markers.dir/detect_markers.cpp.o.requires:

.PHONY : CMakeFiles/detect_markers.dir/detect_markers.cpp.o.requires

CMakeFiles/detect_markers.dir/detect_markers.cpp.o.provides: CMakeFiles/detect_markers.dir/detect_markers.cpp.o.requires
	$(MAKE) -f CMakeFiles/detect_markers.dir/build.make CMakeFiles/detect_markers.dir/detect_markers.cpp.o.provides.build
.PHONY : CMakeFiles/detect_markers.dir/detect_markers.cpp.o.provides

CMakeFiles/detect_markers.dir/detect_markers.cpp.o.provides.build: CMakeFiles/detect_markers.dir/detect_markers.cpp.o


# Object files for target detect_markers
detect_markers_OBJECTS = \
"CMakeFiles/detect_markers.dir/detect_markers.cpp.o"

# External object files for target detect_markers
detect_markers_EXTERNAL_OBJECTS =

detect_markers: CMakeFiles/detect_markers.dir/detect_markers.cpp.o
detect_markers: CMakeFiles/detect_markers.dir/build.make
detect_markers: /usr/local/lib/libopencv_superres.so.4.0.0
detect_markers: /usr/local/lib/libopencv_shape.so.4.0.0
detect_markers: /usr/local/lib/libopencv_stitching.so.4.0.0
detect_markers: /usr/local/lib/libopencv_ml.so.4.0.0
detect_markers: /usr/local/lib/libopencv_videostab.so.4.0.0
detect_markers: /usr/local/lib/libopencv_viz.so.4.0.0
detect_markers: /usr/local/lib/libopencv_photo.so.4.0.0
detect_markers: /usr/local/lib/libopencv_objdetect.so.4.0.0
detect_markers: /usr/local/lib/libopencv_dnn.so.4.0.0
detect_markers: /usr/local/lib/libopencv_video.so.4.0.0
detect_markers: /usr/local/lib/libopencv_aruco.so.4.0.0
detect_markers: /usr/local/lib/libopencv_calib3d.so.4.0.0
detect_markers: /usr/local/lib/libopencv_features2d.so.4.0.0
detect_markers: /usr/local/lib/libopencv_flann.so.4.0.0
detect_markers: /usr/local/lib/libopencv_highgui.so.4.0.0
detect_markers: /usr/local/lib/libopencv_videoio.so.4.0.0
detect_markers: /usr/local/lib/libopencv_imgcodecs.so.4.0.0
detect_markers: /usr/local/lib/libopencv_imgproc.so.4.0.0
detect_markers: /usr/local/lib/libopencv_core.so.4.0.0
detect_markers: CMakeFiles/detect_markers.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/luohao/aruco/detect_aruco_marker/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable detect_markers"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/detect_markers.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/detect_markers.dir/build: detect_markers

.PHONY : CMakeFiles/detect_markers.dir/build

CMakeFiles/detect_markers.dir/requires: CMakeFiles/detect_markers.dir/detect_markers.cpp.o.requires

.PHONY : CMakeFiles/detect_markers.dir/requires

CMakeFiles/detect_markers.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/detect_markers.dir/cmake_clean.cmake
.PHONY : CMakeFiles/detect_markers.dir/clean

CMakeFiles/detect_markers.dir/depend:
	cd /home/luohao/aruco/detect_aruco_marker/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/luohao/aruco/detect_aruco_marker /home/luohao/aruco/detect_aruco_marker /home/luohao/aruco/detect_aruco_marker/build /home/luohao/aruco/detect_aruco_marker/build /home/luohao/aruco/detect_aruco_marker/build/CMakeFiles/detect_markers.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/detect_markers.dir/depend

