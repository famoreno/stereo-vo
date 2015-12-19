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

using namespace rso;

CStereoOdometryEstimator::TRectifyParams::TRectifyParams() :
	nOctaves(3)
{
}

/**  Operations:
  *   - Convert to grayscale (if input is RGB)
  *   - Rectify stereo images (if input not already rectified)
  *   - Build pyramid
  */
void CStereoOdometryEstimator::stage1_prepare_rectify(
	CStereoOdometryEstimator::TStereoOdometryRequest & in_imgs,
	CStereoOdometryEstimator::TImagePairData & out_imgpair )
{
	m_profiler.enter("_stg1");

	CObservationStereoImages *obs = in_imgs.stereo_imgs.pointer(); // Use a plain pointer to avoid dereferencing thru the smart pointer

	out_imgpair.timestamp = obs->timestamp;

	// - Convert to grayscale (if input is RGB)
	m_profiler.enter("stg1.grayscale");

	mrpt::utils::CImage img_left_gray  ( obs->imageLeft,  FAST_REF_OR_CONVERT_TO_GRAY );
	mrpt::utils::CImage img_right_gray ( obs->imageRight, FAST_REF_OR_CONVERT_TO_GRAY );

	m_profiler.leave("stg1.grayscale");

	// - Rectify stereo images (if input not already rectified)
	m_profiler.enter("stg1.rectify");

	mrpt::utils::CImage img_left_rect, img_right_rect;
	if (obs->areImagesRectified())
	{
		img_left_rect.copyFastFrom(  img_left_gray  );
		img_right_rect.copyFastFrom( img_right_gray );
	}
	else
	{
		// Precompute stereo rectification map upon first usage:
		if (!m_stereo_rectifier.isSet())
			m_stereo_rectifier.setFromCamParams( *obs );

		m_stereo_rectifier.rectify(
			obs->imageLeft, obs->imageRight,
			img_left_rect, img_right_rect );
	}
	m_profiler.leave("stg1.rectify");

	// - Build pyramids
	m_profiler.enter("stg1.pyramids");

	// ORB computes the features in multi-scale itself so it does not need to build the pyramid (ORB: 1 scale, rest: 
	const size_t n_octaves = params_detect.detect_method == TDetectParams::dmORB ? 1 : params_rectify.nOctaves;

	out_imgpair.left.pyr.buildPyramidFast ( img_left_rect, n_octaves  );
	out_imgpair.right.pyr.buildPyramidFast( img_right_rect, n_octaves );

	m_profiler.leave("stg1.pyramids");
	m_profiler.leave("_stg1");

	if( this->params_general.vo_save_files )
	{
		obs->imageLeft.saveToFile( mrpt::format("%s/left_image_%04d.png",params_general.vo_out_dir.c_str(),m_it_counter).c_str() );
		obs->imageRight.saveToFile( mrpt::format("%s/right_image_%04d.png",params_general.vo_out_dir.c_str(),m_it_counter).c_str() );
	}
} // end-stage1_rectify
