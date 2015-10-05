/* +-------------------------------------------------------------------------+
   |               Sparse Relative Bundle Adjustment Library                 |
   |                          (libsrba)                                      |
   |                                                                         |
   |           Copyright (C) 2010-2011  Jose-Luis Blanco-Claraco             |
   +-------------------------------------------------------------------------+ */

#include <libstereo-odometry.h>
#include "../src/internal_libstereo-odometry.h" // Include this internal header for running tests on it.

#include <gtest/gtest.h>

using namespace rso;
using namespace mrpt;
using namespace mrpt::utils;
using namespace std;

extern std::string RSO_GLOBAL_UNITTEST_SRC_DIR;

MRPT_TODO("Re-enable this test if tracking_SAD_SSE4() is implemented")
#if RSO_HAS_SSE4 && 0

namespace rso {
	extern void tracking_SAD_default(
		const uint8_t *img_data1,
		const uint8_t *img_data2,
		const size_t img_stride,
		const mrpt::utils::TPixelCoord &pt1_center,
		const mrpt::utils::TPixelCoord &pt2_center,
		const int search_window_x, 
		const int search_window_y,
		mrpt::utils::TPixelCoord & out_best_match, 
		uint32_t &out_minimum_SAD		
		);

	extern void tracking_SAD_SSE4(
		const uint8_t *img_data1,
		const uint8_t *img_data2,
		const size_t img_stride,
		const mrpt::utils::TPixelCoord &pt1_center,
		const mrpt::utils::TPixelCoord &pt2_center,
		const int search_window_x, 
		const int search_window_y,
		mrpt::utils::TPixelCoord & out_best_match, 
		uint32_t &out_minimum_SAD		
		);
}

TEST(SAD_Tests,trackingSAD_Normal_vs_SSE4)
{
	CImage imgL,imgR; 
	if ( !imgL.loadFromFile(RSO_GLOBAL_UNITTEST_SRC_DIR+string("/tests/0L.png")) ) FAIL() << "Couldn't load test image!";
	if ( !imgR.loadFromFile(RSO_GLOBAL_UNITTEST_SRC_DIR+string("/tests/0R.png")) ) FAIL() << "Couldn't load test image!";

	const TPixelCoord pt1(646, 263);
	const TPixelCoord pt2(624, 263);

	const int WIN_X = 16;
	const int WIN_Y = 16;

	TPixelCoord match_pt_normal, match_pt_sse;
	uint32_t    match_value_normal,match_value_sse;
	rso::tracking_SAD_default ( imgL.get_unsafe(0,0),imgR.get_unsafe(0,0),imgL.getRowStride(),pt1,pt2,WIN_X,WIN_Y,match_pt_normal, match_value_normal );
	rso::tracking_SAD_SSE4    ( imgL.get_unsafe(0,0),imgR.get_unsafe(0,0),imgL.getRowStride(),pt1,pt2,WIN_X,WIN_Y,match_pt_normal, match_value_normal );

	EXPECT_EQ(match_pt_normal.x, match_pt_sse.x);
	EXPECT_EQ(match_pt_normal.y, match_pt_sse.y);
	EXPECT_EQ(match_value_normal,match_value_sse);
}

#endif // RSO_HAS_SSE4
