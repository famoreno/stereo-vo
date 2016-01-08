# ------------------------------------------------------------------------+
#                    libstereo-odometry package                           |
#                                                                         |
#  A set of Robust Stereo Odometry (RSO) tools.                           |
#                                                                         |
#  This is the main CMake script for the entire project.                  |
#  See README.txt                                                         |
#                                                                         |
# Copyright (C) 2012-2016 Francisco-Angel Moreno, Jose-Luis Blanco-Claraco|
#                     and Javier Gonzalez-Jimenez                         |
#                     (University of Malaga)                              |
#                                                Released under GNU GPL 3 |
# ------------------------------------------------------------------------+
PROJECT(stereo-vo)

cmake_minimum_required(VERSION 2.4.5)
SET(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS 1)

# Save libs and executables in the same place
SET( LIBRARY_OUTPUT_PATH ${${PROJECT_NAME}_BINARY_DIR}/lib CACHE PATH "Output directory for libraries" )
SET( EXECUTABLE_OUTPUT_PATH ${${PROJECT_NAME}_BINARY_DIR}/bin CACHE PATH "Output directory for applications" )

# Create a .h file with the SOURCE_DIR so compiled programs can load predefined datsets and such. 
# (BTW: It seems impossible to do this thru a -D flag!)
FILE(WRITE ${${PROJECT_NAME}_BINARY_DIR}/rba-dir-config.h "const char *RBA_PATH_BASE=\"${CMAKE_SOURCE_DIR}\";\n")
INCLUDE_DIRECTORIES(${${PROJECT_NAME}_BINARY_DIR})

# Most of the options,  FIND_*(), etc. are done within the scripts of 
#  the individual projects, so all we have to do here is:

IF (UNIX)
	SET(DEFAULT_SHARED_LIBS ON)
ELSE()
	SET(DEFAULT_SHARED_LIBS OFF)
ENDIF()

SET(BUILD_SHARED_LIBS ${DEFAULT_SHARED_LIBS} CACHE BOOL "Build shared libraries (.dll/.so) instead of static ones (.lib/.a)")


# Minimum version required here: 1.0.0 (for mrpt libs, see www.mrpt.org)
#SET(MRPT_DONT_USE_DBG_LIBS 1) # For CMake "Debug" builds in Linux
FIND_PACKAGE(MRPT REQUIRED hwdrivers gui vision obs base opengl)
FIND_PACKAGE(OpenCV 2.4.5 REQUIRED core highgui features2d video)
#FIND_PACKAGE(OpenCV 3.1.0 REQUIRED core highgui features2d video xfeatures2d)

IF (OpenCV_FOUND)
	MESSAGE(STATUS "OpenCV version found : ${OpenCV_VERSION}")
	IF ($ENV{VERBOSE})
		MESSAGE(STATUS "  OpenCV  include dirs : ${OpenCV_INCLUDE_DIRS}")
	ENDIF()	
ENDIF()

IF (MRPT_FOUND)
	MESSAGE(STATUS "MRPT version found   : ${MRPT_VERSION}")
	IF ($ENV{VERBOSE})
		MESSAGE(STATUS "MRPT include dirs  : ${MRPT_INCLUDE_DIRS}")
	ENDIF()	
ENDIF()


# Shared options between GCC and CLANG:
# ======================================
IF (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang" OR CMAKE_COMPILER_IS_GNUCXX)
	# Set debug or optimized building:
	IF (CMAKE_BUILD_TYPE MATCHES "Debug")
	        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
	ELSE ()
	        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mtune=native") # Native should automatically select any of these: -msse2 -msse3 -mssse3
	ENDIF ()

	# We need pthreads
	IF (UNIX AND NOT APPLE)
		SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
	ENDIF(UNIX AND NOT APPLE)

	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -Wno-unused-local-typedefs ")
ENDIF()

# Avoid silly warnings.
IF(MSVC)
        ADD_DEFINITIONS(-D_CRT_SECURE_NO_WARNINGS)
ENDIF(MSVC)


# The main library itself (including unit tests)
ADD_SUBDIRECTORY(libstereo-odometry)

# Demo apps:
ADD_SUBDIRECTORY(demo-stereo-odometry)
