# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.3

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

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = C:\Programmi2\cmake-3.3.1-win32-x86\bin\cmake.exe

# The command to remove a file.
RM = C:\Programmi2\cmake-3.3.1-win32-x86\bin\cmake.exe -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = E:\Universita\Didattica\LAS\git\las_stuff\source\grid_controller

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = E:\Universita\Didattica\LAS\git\las_stuff\source\grid_controller\build

# Include any dependencies generated for this target.
include src/CMakeFiles/grid_controller.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/grid_controller.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/grid_controller.dir/flags.make

src/CMakeFiles/grid_controller.dir/grid_controller.cpp.obj: src/CMakeFiles/grid_controller.dir/flags.make
src/CMakeFiles/grid_controller.dir/grid_controller.cpp.obj: src/CMakeFiles/grid_controller.dir/includes_CXX.rsp
src/CMakeFiles/grid_controller.dir/grid_controller.cpp.obj: ../src/grid_controller.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=E:\Universita\Didattica\LAS\git\las_stuff\source\grid_controller\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/grid_controller.dir/grid_controller.cpp.obj"
	cd /d E:\Universita\Didattica\LAS\git\las_stuff\source\grid_controller\build\src && C:\MinGW\bin\g++.exe   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles\grid_controller.dir\grid_controller.cpp.obj -c E:\Universita\Didattica\LAS\git\las_stuff\source\grid_controller\src\grid_controller.cpp

src/CMakeFiles/grid_controller.dir/grid_controller.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/grid_controller.dir/grid_controller.cpp.i"
	cd /d E:\Universita\Didattica\LAS\git\las_stuff\source\grid_controller\build\src && C:\MinGW\bin\g++.exe  $(CXX_DEFINES) $(CXX_FLAGS) -E E:\Universita\Didattica\LAS\git\las_stuff\source\grid_controller\src\grid_controller.cpp > CMakeFiles\grid_controller.dir\grid_controller.cpp.i

src/CMakeFiles/grid_controller.dir/grid_controller.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/grid_controller.dir/grid_controller.cpp.s"
	cd /d E:\Universita\Didattica\LAS\git\las_stuff\source\grid_controller\build\src && C:\MinGW\bin\g++.exe  $(CXX_DEFINES) $(CXX_FLAGS) -S E:\Universita\Didattica\LAS\git\las_stuff\source\grid_controller\src\grid_controller.cpp -o CMakeFiles\grid_controller.dir\grid_controller.cpp.s

src/CMakeFiles/grid_controller.dir/grid_controller.cpp.obj.requires:

.PHONY : src/CMakeFiles/grid_controller.dir/grid_controller.cpp.obj.requires

src/CMakeFiles/grid_controller.dir/grid_controller.cpp.obj.provides: src/CMakeFiles/grid_controller.dir/grid_controller.cpp.obj.requires
	$(MAKE) -f src\CMakeFiles\grid_controller.dir\build.make src/CMakeFiles/grid_controller.dir/grid_controller.cpp.obj.provides.build
.PHONY : src/CMakeFiles/grid_controller.dir/grid_controller.cpp.obj.provides

src/CMakeFiles/grid_controller.dir/grid_controller.cpp.obj.provides.build: src/CMakeFiles/grid_controller.dir/grid_controller.cpp.obj


src/CMakeFiles/grid_controller.dir/grid_controller_main.cpp.obj: src/CMakeFiles/grid_controller.dir/flags.make
src/CMakeFiles/grid_controller.dir/grid_controller_main.cpp.obj: src/CMakeFiles/grid_controller.dir/includes_CXX.rsp
src/CMakeFiles/grid_controller.dir/grid_controller_main.cpp.obj: ../src/grid_controller_main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=E:\Universita\Didattica\LAS\git\las_stuff\source\grid_controller\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/CMakeFiles/grid_controller.dir/grid_controller_main.cpp.obj"
	cd /d E:\Universita\Didattica\LAS\git\las_stuff\source\grid_controller\build\src && C:\MinGW\bin\g++.exe   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles\grid_controller.dir\grid_controller_main.cpp.obj -c E:\Universita\Didattica\LAS\git\las_stuff\source\grid_controller\src\grid_controller_main.cpp

src/CMakeFiles/grid_controller.dir/grid_controller_main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/grid_controller.dir/grid_controller_main.cpp.i"
	cd /d E:\Universita\Didattica\LAS\git\las_stuff\source\grid_controller\build\src && C:\MinGW\bin\g++.exe  $(CXX_DEFINES) $(CXX_FLAGS) -E E:\Universita\Didattica\LAS\git\las_stuff\source\grid_controller\src\grid_controller_main.cpp > CMakeFiles\grid_controller.dir\grid_controller_main.cpp.i

src/CMakeFiles/grid_controller.dir/grid_controller_main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/grid_controller.dir/grid_controller_main.cpp.s"
	cd /d E:\Universita\Didattica\LAS\git\las_stuff\source\grid_controller\build\src && C:\MinGW\bin\g++.exe  $(CXX_DEFINES) $(CXX_FLAGS) -S E:\Universita\Didattica\LAS\git\las_stuff\source\grid_controller\src\grid_controller_main.cpp -o CMakeFiles\grid_controller.dir\grid_controller_main.cpp.s

src/CMakeFiles/grid_controller.dir/grid_controller_main.cpp.obj.requires:

.PHONY : src/CMakeFiles/grid_controller.dir/grid_controller_main.cpp.obj.requires

src/CMakeFiles/grid_controller.dir/grid_controller_main.cpp.obj.provides: src/CMakeFiles/grid_controller.dir/grid_controller_main.cpp.obj.requires
	$(MAKE) -f src\CMakeFiles\grid_controller.dir\build.make src/CMakeFiles/grid_controller.dir/grid_controller_main.cpp.obj.provides.build
.PHONY : src/CMakeFiles/grid_controller.dir/grid_controller_main.cpp.obj.provides

src/CMakeFiles/grid_controller.dir/grid_controller_main.cpp.obj.provides.build: src/CMakeFiles/grid_controller.dir/grid_controller_main.cpp.obj


# Object files for target grid_controller
grid_controller_OBJECTS = \
"CMakeFiles/grid_controller.dir/grid_controller.cpp.obj" \
"CMakeFiles/grid_controller.dir/grid_controller_main.cpp.obj"

# External object files for target grid_controller
grid_controller_EXTERNAL_OBJECTS =

../bin/grid_controller.exe: src/CMakeFiles/grid_controller.dir/grid_controller.cpp.obj
../bin/grid_controller.exe: src/CMakeFiles/grid_controller.dir/grid_controller_main.cpp.obj
../bin/grid_controller.exe: src/CMakeFiles/grid_controller.dir/build.make
../bin/grid_controller.exe: C:/Programmi2/opencv2.4.11/build/x86/mingw/lib/libopencv_videostab2411.dll.a
../bin/grid_controller.exe: C:/Programmi2/opencv2.4.11/build/x86/mingw/lib/libopencv_ts2411.a
../bin/grid_controller.exe: C:/Programmi2/opencv2.4.11/build/x86/mingw/lib/libopencv_superres2411.dll.a
../bin/grid_controller.exe: C:/Programmi2/opencv2.4.11/build/x86/mingw/lib/libopencv_stitching2411.dll.a
../bin/grid_controller.exe: C:/Programmi2/opencv2.4.11/build/x86/mingw/lib/libopencv_contrib2411.dll.a
../bin/grid_controller.exe: C:/Programmi2/opencv2.4.11/build/x86/mingw/lib/libopencv_nonfree2411.dll.a
../bin/grid_controller.exe: C:/Programmi2/opencv2.4.11/build/x86/mingw/lib/libopencv_ocl2411.dll.a
../bin/grid_controller.exe: C:/Programmi2/opencv2.4.11/build/x86/mingw/lib/libopencv_gpu2411.dll.a
../bin/grid_controller.exe: C:/Programmi2/opencv2.4.11/build/x86/mingw/lib/libopencv_photo2411.dll.a
../bin/grid_controller.exe: C:/Programmi2/opencv2.4.11/build/x86/mingw/lib/libopencv_objdetect2411.dll.a
../bin/grid_controller.exe: C:/Programmi2/opencv2.4.11/build/x86/mingw/lib/libopencv_legacy2411.dll.a
../bin/grid_controller.exe: C:/Programmi2/opencv2.4.11/build/x86/mingw/lib/libopencv_video2411.dll.a
../bin/grid_controller.exe: C:/Programmi2/opencv2.4.11/build/x86/mingw/lib/libopencv_ml2411.dll.a
../bin/grid_controller.exe: C:/Programmi2/opencv2.4.11/build/x86/mingw/lib/libopencv_calib3d2411.dll.a
../bin/grid_controller.exe: C:/Programmi2/opencv2.4.11/build/x86/mingw/lib/libopencv_features2d2411.dll.a
../bin/grid_controller.exe: C:/Programmi2/opencv2.4.11/build/x86/mingw/lib/libopencv_highgui2411.dll.a
../bin/grid_controller.exe: C:/Programmi2/opencv2.4.11/build/x86/mingw/lib/libopencv_imgproc2411.dll.a
../bin/grid_controller.exe: C:/Programmi2/opencv2.4.11/build/x86/mingw/lib/libopencv_flann2411.dll.a
../bin/grid_controller.exe: C:/Programmi2/opencv2.4.11/build/x86/mingw/lib/libopencv_core2411.dll.a
../bin/grid_controller.exe: src/CMakeFiles/grid_controller.dir/linklibs.rsp
../bin/grid_controller.exe: src/CMakeFiles/grid_controller.dir/objects1.rsp
../bin/grid_controller.exe: src/CMakeFiles/grid_controller.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=E:\Universita\Didattica\LAS\git\las_stuff\source\grid_controller\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable ..\..\bin\grid_controller.exe"
	cd /d E:\Universita\Didattica\LAS\git\las_stuff\source\grid_controller\build\src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\grid_controller.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/grid_controller.dir/build: ../bin/grid_controller.exe

.PHONY : src/CMakeFiles/grid_controller.dir/build

src/CMakeFiles/grid_controller.dir/requires: src/CMakeFiles/grid_controller.dir/grid_controller.cpp.obj.requires
src/CMakeFiles/grid_controller.dir/requires: src/CMakeFiles/grid_controller.dir/grid_controller_main.cpp.obj.requires

.PHONY : src/CMakeFiles/grid_controller.dir/requires

src/CMakeFiles/grid_controller.dir/clean:
	cd /d E:\Universita\Didattica\LAS\git\las_stuff\source\grid_controller\build\src && $(CMAKE_COMMAND) -P CMakeFiles\grid_controller.dir\cmake_clean.cmake
.PHONY : src/CMakeFiles/grid_controller.dir/clean

src/CMakeFiles/grid_controller.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" E:\Universita\Didattica\LAS\git\las_stuff\source\grid_controller E:\Universita\Didattica\LAS\git\las_stuff\source\grid_controller\src E:\Universita\Didattica\LAS\git\las_stuff\source\grid_controller\build E:\Universita\Didattica\LAS\git\las_stuff\source\grid_controller\build\src E:\Universita\Didattica\LAS\git\las_stuff\source\grid_controller\build\src\CMakeFiles\grid_controller.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/grid_controller.dir/depend

