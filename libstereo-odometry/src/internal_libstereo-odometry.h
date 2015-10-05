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

#pragma once

// Include public part of the API first
#include <libstereo-odometry.h>

#define VERBOSE_LEVEL(_LEV) if (m_verbose_level>=_LEV) std::cout

// Assert level:
#ifdef _DEBUG
	#define RSO_ASSERT(_COND_) { const bool cnd=_COND_; if (!cnd) rso::debug_pre_assert(#_COND_); ASSERT_(_COND_) }
#else
	#define RSO_ASSERT(_COND_) ASSERT_(_COND_)
#endif

/** Use optimized functions with the SSE3 machine instructions set */
#if defined WIN32 && (!defined WIN64 || defined EM64T) && \
 (_MSC_VER >= 1500) || (defined __SSE4__ && defined __GNUC__ && __GNUC__ >= 4)
	#define RSO_HAS_SSE4  0 
#else
	#define RSO_HAS_SSE4  0
#endif

#if RSO_HAS_SSE4  
    extern "C" {
	#include <smmintrin.h>  // Same header for MSVC & GCC
    }
#endif


namespace rso
{
	using namespace std;
	using namespace mrpt::vision;

	// Auxiliary function for debugging asserts:
	void debug_pre_assert(const char* failed_test);

	/** Generates a string for a container in the format [A,B,C,...], and the fmt string for <b>each</b> vector element. */
	template <typename T>
	std::string sprintf_container(const char *fmt, const T &V )
	{
		std::string ret = "[";
		typename T::const_iterator it=V.begin();
		for (;it!=V.end();)
		{
			ret+= format(fmt,*it);
			++it;
			if (it!=V.end())
				ret+= ",";
		}
		ret+="]";
		return ret;
	}

	/** Sum of Absolute Differences (SAD) in a window of size 8. 
	  *  Use optimized SSE4 version if available; falls back to standard C++ version otherwise.
	  * (Implemented in compute_SAD8.cpp)
	  * \note This function doesn't check if the [-3,+4] window falls out of the image!
	  * \note A feature patch comprises the 8x8 pixels in the area: (x-3,y-3)-(x+4,y+4)
	  */
	uint32_t compute_SAD8(
		const uint8_t *img_data_L,
		const uint8_t *img_data_R,
		const size_t img_stride,
		const mrpt::utils::TPixelCoord &pt_L,
		const mrpt::utils::TPixelCoord &pt_R);


	/** Feature tracking from an \a img1 by search of the minimum SAD in a window within \a img2. 
	  *  \param  search_window_x For efficiency, this must be multiple of 8.
	  *  \param  search_window_y Can has any positive value
	  *
	  *  Use optimized SSE4 version if available; falls back to standard C++ version otherwise.
	  * \note This function doesn't check if the window falls out of the image!
	  * \note A feature patch comprises the 8x8 pixels in the area: (x-3,y-3)-(x+4,y+4)
	  */
	void tracking_SAD(
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

} // end of namespace

