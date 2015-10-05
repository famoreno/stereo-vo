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

uint32_t compute_SAD8_SSE4(
	const uint8_t *img_data_L,
	const uint8_t *img_data_R,
	const size_t img_stride,
	const mrpt::utils::TPixelCoord &pt_L,
	const mrpt::utils::TPixelCoord &pt_R)
{
#if RSO_HAS_SSE4
	const uint8_t *ptrL = img_data_L+img_stride*(pt_L.y-3) + (pt_L.x-3);
	const uint8_t *ptrR = img_data_R+img_stride*(pt_R.y-3) + (pt_R.x-3);

	// Refer to the documentation of _mm_mpsadbw_epu8() for details
	// See also: http://software.intel.com/en-us/articles/motion-estimation-with-intel-streaming-simd-extensions-4-intel-sse4/
	const int mask_00 = 0x00;   // SAD of bytes 3:0 of both L&R images
	const int mask_44 = 0x05;   // SAD of bytes 7:4 of both L&R images

	int16_t total_SAD=0;
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
	return total_SAD;
#else
	return 0; // shouldn't ever reach this anyway
#endif
}

// Sum of Absolute Differences (SAD) in a window of size 8:
// (Generic version)
// Note: This function doesn't check if the [-3,+4] window falls out of the image!
uint32_t compute_SAD8_default(
	const uint8_t *img_data_L,
	const uint8_t *img_data_R,
	const size_t img_stride,
	const mrpt::utils::TPixelCoord &pt_L,
	const mrpt::utils::TPixelCoord &pt_R)
{
	// Window is: [x-3,x+4]*[y-3,y+4]
	//       v
	// ...01234567....

	const uint8_t *ptrL = img_data_L+img_stride*(pt_L.y-3) + (pt_L.x-3);
	const uint8_t *ptrR = img_data_R+img_stride*(pt_R.y-3) + (pt_R.x-3);

	uint32_t SAD_sum = 0;
	for (int y=0;y<8;y++)
	{
		for (int x=0;x<8;x++)
		{
			const int32_t dif = ptrL[x]-ptrR[x];
			SAD_sum+= dif>0 ? dif:-dif;
		}
		ptrL+=img_stride;
		ptrR+=img_stride;
	}
	return SAD_sum;
}

} // end namespace



/** Sum of Absolute Differences (SAD) in a window of size 8. 
  *  Use optimized SSE4 version if available; falls back to standard C++ version otherwise.
  * (Implemented in compute_SAD8.cpp)
  * \note This function doesn't check if the [-3,+4] window falls out of the image!
  */
uint32_t rso::compute_SAD8(
	const uint8_t *img_data_L,
	const uint8_t *img_data_R,
	const size_t img_stride,
	const mrpt::utils::TPixelCoord &pt_L,
	const mrpt::utils::TPixelCoord &pt_R)
{
#if RSO_HAS_SSE4
	// ~80ns for Intel i5 2.27GHz
	return compute_SAD8_SSE4(img_data_L,img_data_R,img_stride,pt_L,pt_R);
#else
	// ~ 310ns for Intel i5 2.27GHz
	return compute_SAD8_default(img_data_L,img_data_R,img_stride,pt_L,pt_R);
#endif
}
