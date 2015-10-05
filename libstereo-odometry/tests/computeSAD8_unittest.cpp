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

TEST(SAD_Tests,computeSAD8)
{
	CImage imgL,imgR; 
	if ( !imgL.loadFromFile(RSO_GLOBAL_UNITTEST_SRC_DIR+string("/tests/0L.png")) ) FAIL() << "Couldn't load test image!";
	if ( !imgR.loadFromFile(RSO_GLOBAL_UNITTEST_SRC_DIR+string("/tests/0R.png")) ) FAIL() << "Couldn't load test image!";


	// TestImg0: L:(646, 263) <-> R:(624,263)
	// Minimum of a good match seems to be ~300-500
	uint32_t SADs[3][3];
	for (int iy=-1;iy<=1;iy++)
		for (int ix=-1;ix<=1;ix++)
			SADs[iy+1][ix+1]= compute_SAD8(
				imgL.get_unsafe(0,0),imgR.get_unsafe(0,0),imgL.getRowStride(),
				TPixelCoord(646, 263),TPixelCoord(624+ix,263+iy));

	// SADs[1][1] should be a local minimum:
	for (int iy=-1;iy<=1;iy++)
		for (int ix=-1;ix<=1;ix++)
			if (ix!=0 || iy!=0)
				EXPECT_GT(SADs[1+ix][1+iy],SADs[1][1]);
}

#if RSO_HAS_SSE4

namespace rso {
	extern uint32_t compute_SAD8_SSE4(
		const uint8_t *img_data_L,
		const uint8_t *img_data_R,
		const size_t img_stride,
		const mrpt::utils::TPixelCoord &pt_L,
		const mrpt::utils::TPixelCoord &pt_R);

	extern uint32_t compute_SAD8_default(
		const uint8_t *img_data_L,
		const uint8_t *img_data_R,
		const size_t img_stride,
		const mrpt::utils::TPixelCoord &pt_L,
		const mrpt::utils::TPixelCoord &pt_R);
}

TEST(SAD_Tests,computeSAD8_Normal_vs_SSE4)
{
	CImage imgL,imgR; 
	if ( !imgL.loadFromFile(RSO_GLOBAL_UNITTEST_SRC_DIR+string("/tests/0L.png")) ) FAIL() << "Couldn't load test image!";
	if ( !imgR.loadFromFile(RSO_GLOBAL_UNITTEST_SRC_DIR+string("/tests/0R.png")) ) FAIL() << "Couldn't load test image!";

	for (int iy=0;iy<10;iy++) 
	{
		for (int ix=0;ix<10;ix++)
		{
			const uint32_t sad_normal = compute_SAD8_default( imgL.get_unsafe(0,0),imgR.get_unsafe(0,0),imgL.getRowStride(),TPixelCoord(646+ix, 263+iy),TPixelCoord(624,263) );
			const uint32_t sad_sse4   = compute_SAD8_SSE4( imgL.get_unsafe(0,0),imgR.get_unsafe(0,0),imgL.getRowStride(),TPixelCoord(646+ix, 263+iy),TPixelCoord(624,263) );
			EXPECT_EQ(sad_normal, sad_sse4);
		}
	}
}
#endif // RSO_HAS_SSE4
