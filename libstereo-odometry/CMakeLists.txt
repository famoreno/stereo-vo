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
PROJECT(libstereo_odometry)

CMAKE_MINIMUM_REQUIRED(VERSION 2.4)
# --------------------------------------------------------------
# Indicate CMake 2.7 and above that we don't want to mix relative
#  and absolute paths in linker lib lists.
# Run "cmake --help-policy CMP0003" for more information.
# --------------------------------------------------------------
if(COMMAND cmake_policy)
      cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES("${${PROJECT_NAME}_SOURCE_DIR}/include")
INCLUDE_DIRECTORIES("${OpenCV_INCLUDE_DIRS}")

# Declare the library "libsrba"
FILE(GLOB LIBHDRS "src/*.h")
FILE(GLOB LIBSRCS "src/*.cpp")
FILE(GLOB LIBHDRSPUB "include/*.h")
ADD_LIBRARY(stereo-odometry ${LIBHDRS} ${LIBSRCS} ${LIBHDRSPUB})
TARGET_LINK_LIBRARIES(stereo-odometry ${MRPT_LIBS} ${OpenCV_LIBS})


# Unit tests:
SET( BUILD_TESTING ON CACHE BOOL "Build and run unit tests")

IF( BUILD_TESTING)
	# Try using libgtest (Google testing library) from the system, if available
	IF(UNIX)
		FIND_FILE(GTEST_CONFIG_FILE gtest-config)
		IF(GTEST_CONFIG_FILE)
			MARK_AS_ADVANCED(GTEST_CONFIG_FILE)

			SET(CMAKE_LIBSTEREO_HAS_GTEST_SYSTEM 1)

			# Get the version:
			EXECUTE_PROCESS(COMMAND ${GTEST_CONFIG_FILE} --version
				RESULT_VARIABLE CMAKE_GTEST_CONFIG_RES
				OUTPUT_VARIABLE CMAKE_LIBSTEREO_HAS_GTEST_VERSION
				OUTPUT_STRIP_TRAILING_WHITESPACE
				)
			IF(${CMAKE_GTEST_CONFIG_RES})
				MESSAGE("Error invoking GTEST config file:\n ${GTEST_CONFIG_FILE} --version")
			ENDIF(${CMAKE_GTEST_CONFIG_RES})

			# Get the config params:
			EXECUTE_PROCESS(COMMAND ${GTEST_CONFIG_FILE} --libs
				RESULT_VARIABLE CMAKE_GTEST_CONFIG_RES
				OUTPUT_VARIABLE CMAKE_GTEST_LIBS
				OUTPUT_STRIP_TRAILING_WHITESPACE
				)
			IF(${CMAKE_GTEST_CONFIG_RES})
				MESSAGE("Error invoking GTEST config file:\n ${GTEST_CONFIG_FILE} --libs")
			ENDIF(${CMAKE_GTEST_CONFIG_RES})

			EXECUTE_PROCESS(COMMAND ${GTEST_CONFIG_FILE} --cxxflags
				RESULT_VARIABLE CMAKE_GTEST_CONFIG_RES
				OUTPUT_VARIABLE CMAKE_GTEST_CFLAGS
				OUTPUT_STRIP_TRAILING_WHITESPACE
				)
			IF(${CMAKE_GTEST_CONFIG_RES})
				MESSAGE("Error invoking GTEST config file:\n ${GTEST_CONFIG_FILE} --cxxflags")
			ENDIF(${CMAKE_GTEST_CONFIG_RES})

		ELSE(GTEST_CONFIG_FILE)
			SET(CMAKE_LIBSTEREO_HAS_GTEST_SYSTEM 0)
		ENDIF(GTEST_CONFIG_FILE)
	ENDIF(UNIX)

	add_subdirectory(tests)
ENDIF( BUILD_TESTING)


