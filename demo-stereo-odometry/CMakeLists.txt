# ------------------------------------------------------------------------+
#                    libstereo-odometry package                           |
#                                                                         |
#  A set of Robust Stereo Odometry (RSO) tools.                           |
#                                                                         |
#  Refer to the main CMake script at the root directory.                  |
#                                                                         |
# Copyright (C) 2012 Jose-Luis Blanco-Claraco, Francisco-Angel Moreno     |
#                     and Javier Gonzalez-Jimenez                         |
#                     (University of Malaga)                              |
#                                                Released under GNU GPL 3 |
# ------------------------------------------------------------------------+
PROJECT(demo-stereo-odometry)

CMAKE_MINIMUM_REQUIRED(VERSION 2.4)
# --------------------------------------------------------------
# Indicate CMake 2.7 and above that we don't want to mix relative
#  and absolute paths in linker lib lists.
# Run "cmake --help-policy CMP0003" for more information.
# --------------------------------------------------------------
if(COMMAND cmake_policy)
      cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

# Use stereo-odometry headers:
INCLUDE_DIRECTORIES("${libstereo_odometry_SOURCE_DIR}/include")

# Set debug or optimized building:
IF(CMAKE_COMPILER_IS_GNUCXX)
	IF (CMAKE_BUILD_TYPE MATCHES "Debug")
	        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
	ELSE (CMAKE_BUILD_TYPE MATCHES "Debug")
	        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mtune=native -msse2 -msse3 -mssse3")
	ENDIF (CMAKE_BUILD_TYPE MATCHES "Debug")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

FILE(GLOB PROGSRCS "./*.cpp")
FILE(GLOB PROGHDRS "./*.h")

ADD_EXECUTABLE(${PROJECT_NAME} ${PROGSRCS} ${PROGHDRS})
ADD_DEPENDENCIES(demo-stereo-odometry stereo-odometry)
# NOTE: It seems crucial the order of the linking libs!! First, our lib, then MRPT ones
TARGET_LINK_LIBRARIES(demo-stereo-odometry stereo-odometry )
TARGET_LINK_LIBRARIES(demo-stereo-odometry ${MRPT_LIBS} ${OpenCV_LIBS})

