# Tests based on Google gtest:
# -----------------------------
PROJECT(tests)

#SET(EXECUTABLE_OUTPUT_PATH "${${PROJECT_NAME}_BINARY_DIR}")

# Minimum version required here: 0.9.5
#  (for mrpt libs, see www.mrpt.org)
#SET(MRPT_DONT_USE_DBG_LIBS 1) # For CMake "Debug" builds in Linux
FIND_PACKAGE(MRPT REQUIRED base vision gui)

if(POLICY CMP0037)
	cmake_policy(SET CMP0037 OLD)  # Allow creating target named "test"
endif()

# -----------------------------
# Add the "make test" target:
# -----------------------------
ADD_CUSTOM_TARGET(test)

# Include gtest from embeded lib:
IF (NOT CMAKE_LIBSTEREO_HAS_GTEST_SYSTEM)
	INCLUDE_DIRECTORIES("${tests_SOURCE_DIR}/")  # So we can do: <gtest/gtest.h>
	SET(EXTRA_GTEST_FILES "${tests_SOURCE_DIR}/gtest/gtest-all.cc")
ENDIF(NOT CMAKE_LIBSTEREO_HAS_GTEST_SYSTEM)

# If using system library, add C++ flags:
IF (NOT "${CMAKE_GTEST_CFLAGS}" STREQUAL "")
	ADD_DEFINITIONS(${CMAKE_GTEST_CFLAGS})
ENDIF (NOT "${CMAKE_GTEST_CFLAGS}" STREQUAL "")

# Create list of unit test files:
FILE(GLOB TESTFILES "*.cpp")

# Test project:
ADD_EXECUTABLE( test_libstereo-odometry ${TESTFILES} ${EXTRA_GTEST_FILES})

# Add the required libraries for linking:
IF (CMAKE_LIBSTEREO_HAS_GTEST_SYSTEM)
	TARGET_LINK_LIBRARIES(test_libstereo-odometry "gtest")
ENDIF(CMAKE_LIBSTEREO_HAS_GTEST_SYSTEM)

# Link against mrpt libs	
TARGET_LINK_LIBRARIES(test_libstereo-odometry ${MRPT_LIBS})

# ------------ Deps: ---------------
# Include dirs:
include_directories("${libstereo-odometry_SOURCE_DIR}/include")

# Linking deps:
ADD_DEPENDENCIES(test_libstereo-odometry        stereo-odometry)
TARGET_LINK_LIBRARIES(test_libstereo-odometry   stereo-odometry)

# Run it:
ADD_CUSTOM_TARGET(run_tests_libstereo-odometry COMMAND "$<TARGET_FILE:test_libstereo-odometry>" "${libstereo_odometry_SOURCE_DIR}")
ADD_DEPENDENCIES(run_tests_libstereo-odometry test_libstereo-odometry)
ADD_DEPENDENCIES(test run_tests_libstereo-odometry)
