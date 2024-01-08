# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/lib/python3.10/dist-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /usr/local/lib/python3.10/dist-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /content/mini-dnn-cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /content/mini-dnn-cpp/build

# Include any dependencies generated for this target.
include CMakeFiles/demo.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/demo.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/demo.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/demo.dir/flags.make

CMakeFiles/demo.dir/demo.cc.o: CMakeFiles/demo.dir/flags.make
CMakeFiles/demo.dir/demo.cc.o: /content/mini-dnn-cpp/demo.cc
CMakeFiles/demo.dir/demo.cc.o: CMakeFiles/demo.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/content/mini-dnn-cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/demo.dir/demo.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/demo.dir/demo.cc.o -MF CMakeFiles/demo.dir/demo.cc.o.d -o CMakeFiles/demo.dir/demo.cc.o -c /content/mini-dnn-cpp/demo.cc

CMakeFiles/demo.dir/demo.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/demo.dir/demo.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /content/mini-dnn-cpp/demo.cc > CMakeFiles/demo.dir/demo.cc.i

CMakeFiles/demo.dir/demo.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/demo.dir/demo.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /content/mini-dnn-cpp/demo.cc -o CMakeFiles/demo.dir/demo.cc.s

# Object files for target demo
demo_OBJECTS = \
"CMakeFiles/demo.dir/demo.cc.o"

# External object files for target demo
demo_EXTERNAL_OBJECTS =

demo: CMakeFiles/demo.dir/demo.cc.o
demo: CMakeFiles/demo.dir/build.make
demo: src/libMiniDNNLib.a
demo: CMakeFiles/demo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/content/mini-dnn-cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable demo"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/demo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/demo.dir/build: demo
.PHONY : CMakeFiles/demo.dir/build

CMakeFiles/demo.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/demo.dir/cmake_clean.cmake
.PHONY : CMakeFiles/demo.dir/clean

CMakeFiles/demo.dir/depend:
	cd /content/mini-dnn-cpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /content/mini-dnn-cpp /content/mini-dnn-cpp /content/mini-dnn-cpp/build /content/mini-dnn-cpp/build /content/mini-dnn-cpp/build/CMakeFiles/demo.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/demo.dir/depend

