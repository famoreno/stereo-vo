/* +-------------------------------------------------------------------------+
   |                   Robust Stereo Odometry library                        |
   |                        (libstereo-odometry)                             |
   |                                                                         |
   | Copyright (C) 2012 Jose-Luis Blanco-Claraco, Francisco-Angel Moreno     |
   |                     and Javier Gonzalez-Jimenez                         |
   |                                                                         |
   | This program is free software: you can redistribute it and/or modify    |
   | it under the terms of the GNU General Public License as published by    |
   | the Free Software Foundation, either version 3 of the License, or       |
   | (at your option) any later version.                                     |
   |                                                                         |
   | This program is distributed in the hope that it will be useful,         |
   | but WITHOUT ANY WARRANTY; without even the implied warranty of          |
   | MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           |
   | GNU General Public License for more details.                            |
   |                                                                         |
   | You should have received a copy of the GNU General Public License       |
   | along with this program.  If not, see <http://www.gnu.org/licenses/>.   |
   +-------------------------------------------------------------------------+ */

#include <libstereo-odometry.h>
#include "internal_libstereo-odometry.h"
#include <mrpt/utils/SSE_types.h>

namespace rso
{

// (SSE4 version)
void tracking_SAD_SSE4(
	const uint8_t *img_data1,
	const uint8_t *img_data2,
	const size_t img_stride,
	const mrpt::utils::TPixelCoord &pt1_center,
	const mrpt::utils::TPixelCoord &pt2_center,
	const int search_window_x,
	const int search_window_y,
	mrpt::utils::TPixelCoord & out_best_match,
	uint32_t &out_minimum_SAD
	)
{
#if RSO_HAS_SSE4
	const uint8_t *ptrL = img_data1+img_stride*(pt1_center.y-3) + (pt1_center.x-3);
	//const uint8_t *ptrR = img_data1+img_stride*(pt2_center.y-3) + (pt2_center.x-3);

	// Refer to the documentation of _mm_mpsadbw_epu8() for details
	// See also: http://software.intel.com/en-us/articles/motion-estimation-with-intel-streaming-simd-extensions-4-intel-sse4/
	const int mask_00 = 0x00;   // SAD of bytes 3:0 of both L&R images
	const int mask_44 = 0x05;   // SAD of bytes 7:4 of both L&R images

	int16_t total_SAD=0;
#if 0
	for (int y=0;y<8;y++)
	{
		// Load 8 pixels from each image:
		const __m128i imgL = _mm_loadu_si128((const __m128i*)ptrL); // "u" allows 16-unaligned ptrs
		const __m128i imgR = _mm_loadu_si128((const __m128i*)ptrR); // "u" allows 16-unaligned ptrs

		// We'll only use the lowest 16bit sum (we are wasting a lot of potential of this instruction!!)
		const __m128i sad00 = _mm_mpsadbw_epu8(imgL,imgR, mask_00);
		const __m128i sad44 = _mm_mpsadbw_epu8(imgL,imgR, mask_44);

		total_SAD+= sad00.m128i_i16[0]+sad44.m128i_i16[0];

		ptrL+=img_stride;
		ptrR+=img_stride;
	}
#endif
#endif
}

// (Generic version)
void tracking_SAD_default(
	const uint8_t *img_data1,
	const uint8_t *img_data2,
	const size_t img_stride,
	const mrpt::utils::TPixelCoord &pt1_center,
	const mrpt::utils::TPixelCoord &pt2_center,
	const int search_window_x,
	const int search_window_y,
	mrpt::utils::TPixelCoord & out_best_match,
	uint32_t &out_minimum_SAD
	)
{
	ASSERTMSG_(search_window_x>0 && (search_window_x & 0x07)==0, "search_window_x must be a multiple of 8 and >0")

	// Window is: [x-3,x+4]*[y-3,y+4]
	//       v
	// ...01234567....
	const uint8_t *ptr1_init = img_data1+img_stride*(pt1_center.y-3) + (pt1_center.x-3);

	uint32_t min_SAD = std::numeric_limits<uint32_t>::max();
	mrpt::utils::TPixelCoord min_SAD_pt_incr;

	for (int iy=-search_window_y;iy<=search_window_y;iy++)
	{
		for (int ix=-search_window_x;ix<=search_window_x;ix++)
		{
			// Window at pt1 in img1, and at pt2+increment in img2:
			const uint8_t *ptr1 = ptr1_init;
			const uint8_t *ptr2 = img_data2+img_stride*(pt2_center.y+iy-3) + (pt2_center.x+ix-3);

			uint32_t SAD_sum = 0;
			for (int y=0;y<8;y++) {
				for (int x=0;x<8;x++) {
					const int32_t dif = ptr1[x]-ptr2[x];
					SAD_sum+= dif>0 ? dif:-dif;
				} // end for x
				ptr1+=img_stride;
				ptr2+=img_stride;
			} // end for y

			if (SAD_sum<min_SAD)
			{
				min_SAD=SAD_sum;
				min_SAD_pt_incr.x = ix;
				min_SAD_pt_incr.y = iy;
			}
		} // end for ix
	} // end for iy

	out_best_match.x  = pt2_center.x+min_SAD_pt_incr.x;
	out_best_match.y  = pt2_center.y+min_SAD_pt_incr.y;
	out_minimum_SAD = min_SAD;
}

} // end NS

/** Sum of Absolute Differences (SAD) in a window of size 8.
  *  Use optimized SSE4 version if available; falls back to standard C++ version otherwise.
  * (Implemented in compute_SAD8.cpp)
  * \note This function doesn't check if the [-3,+4] window falls out of the image!
  */
void rso::tracking_SAD(
	const uint8_t *img_data1,
	const uint8_t *img_data2,
	const size_t img_stride,
	const mrpt::utils::TPixelCoord &pt1_center,
	const mrpt::utils::TPixelCoord &pt2_center,
	const int search_window_x,
	const int search_window_y,
	mrpt::utils::TPixelCoord & out_best_match,
	uint32_t &out_minimum_SAD
	)
{
#if 0 && RSO_HAS_SSE4
	return tracking_SAD_SSE4(img_data1,img_data2,img_stride,pt1_center,pt2_center,search_window_x, search_window_y,out_best_match, out_minimum_SAD);
#else
	return tracking_SAD_default(img_data1,img_data2,img_stride,pt1_center,pt2_center,search_window_x, search_window_y,out_best_match, out_minimum_SAD);
#endif
}
