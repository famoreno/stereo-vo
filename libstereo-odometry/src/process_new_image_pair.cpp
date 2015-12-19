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

#include <mrpt/version.h>
#if MRPT_VERSION>=0x130
using mrpt::obs::CObservationStereoImages;
#else
using mrpt::slam::CObservationStereoImages;
#endif

CStereoOdometryEstimator::TGeneralParams::TGeneralParams() :
	vo_use_matches_ids(false), vo_save_files(false), vo_debug(false), vo_pause_it(false), vo_out_dir("out") {}


/*--------------------------------------------------------------------
						processNewImagePair
----------------------------------------------------------------------*/
void CStereoOdometryEstimator::processNewImagePair(
	TStereoOdometryRequest & request_data,
	TStereoOdometryResult & result )
{
	// take this to the constructor or the initialization (not to be done every time!)
	if( (params_general.vo_debug || params_general.vo_save_files) && !mrpt::system::directoryExists( params_general.vo_out_dir ) )
		mrpt::system::createDirectory( params_general.vo_out_dir );
	// ----------------------------------------------------------------

	result.error_code = voecNone;

	// take this to the constructor or the initialization (not to be done every time!)
	string detect_method_str, st_match_method_str, if_match_method_str;
	switch( params_detect.detect_method )
	{
		case TDetectParams::dmORB		: detect_method_str = "[ORB]"; break;
		case TDetectParams::dmFASTER	: detect_method_str = "[FASTER]"; break;
		case TDetectParams::dmFAST_ORB	: detect_method_str = "[FAST + ORB]"; break;
		case TDetectParams::dmKLT		: detect_method_str = "[KLT]"; break;
		default: THROW_EXCEPTION( "[Visual Odometry Error] Detect method not correct." ); break;
	}
	switch( params_lr_match.match_method )
	{
		case TLeftRightMatchParams::smDescBF	: st_match_method_str = "[ORB Descriptor Brute-Force]"; break;
		case TLeftRightMatchParams::smDescRbR	: st_match_method_str = "[ORB Descriptor Row-by-row]"; break;
		case TLeftRightMatchParams::smSAD		: st_match_method_str = "[SAD]"; break;
		default: THROW_EXCEPTION( "[Visual Odometry Error] Stereo matching method not correct." ); break;
	}
	switch( params_if_match.ifm_method )
	{
		case TInterFrameMatchingParams::ifmDescBF		: if_match_method_str = "[ORB Descriptor Brute-Force]"; break;
		case TInterFrameMatchingParams::ifmDescWin		: if_match_method_str = "[ORB Descriptor in a window]"; break;
		case TInterFrameMatchingParams::ifmOpticalFlow	: if_match_method_str = "[Optical flow]"; break;
		case TInterFrameMatchingParams::ifmSAD			: if_match_method_str = "[SAD in a window]"; break;
		default: THROW_EXCEPTION( "[Visual Odometry Error] Interframe matching method not correct." ); break;
	}
	// ----------------------------------------------------------------

	m_profiler.enter("processNewImagePair");

	ASSERTMSG_( request_data.stereo_imgs.present(),"Pointer 'request_data.stereo_imgs' must be set to stereo observation data!" )

	// -------------------------------------------
	// 0) Shift the list of two previous images:
	// -------------------------------------------
	if( !request_data.repeat && 
		m_error != voecBadTracking && 
		m_error != voecBadCondNumber )
		m_prev_imgpair = m_current_imgpair;   // these are smart pointers, so it only implies copying a pointer

	if( request_data.repeat )
	{	VERBOSE_LEVEL(1) << "[sVO] Repeating (no swapping) ... " << endl;	}

	m_error_in_tracking = false;
	m_error = voecNone;

	// -------------------------------------------
	// 1) Prepare new image pair:
	// -------------------------------------------
	m_current_imgpair = TImagePairDataPtr( new TImagePairData() );
	TImagePairData & cur_imgpair = *m_current_imgpair;

    VERBOSE_LEVEL(1) << "[sVO] RECTIFYING... ";
	stage1_prepare_rectify( request_data, cur_imgpair );
	VERBOSE_LEVEL(1) << endl << "	done" << endl;
	
	// number of octaves
	const size_t nOctaves = cur_imgpair.left.pyr.images.size();
	
	// "copyFastFrom" has "move semantics", so the original input images are no longer available to subsequent stages:
	if( !request_data.repeat && 
		!m_error_in_tracking )
	{
		CObservationStereoImages *stObs = const_cast<CObservationStereoImages*>(request_data.stereo_imgs.pointer());
		
		// gui info
		m_next_gui_info->timestamp = stObs->timestamp;
		m_next_gui_info->img_left.swap( stObs->imageLeft );
		m_next_gui_info->img_right.swap( stObs->imageRight );
	}

	if( m_prev_imgpair )
	{	VERBOSE_LEVEL(2) << "[sVO] Image Timestamps -- PRE:" << m_prev_imgpair->timestamp << " and CUR: " << m_next_gui_info->timestamp << endl;	}
	else
	{	VERBOSE_LEVEL(2) << "[sVO] Image Timestamps -- PRE: None and CUR: " << m_next_gui_info->timestamp << endl;	}
    
	// -------------------------------------------
	// 2) Detect features:
	// -------------------------------------------
    VERBOSE_LEVEL(1) << "[sVO] DETECTING FEATURES... with " << detect_method_str;
    if( request_data.use_precomputed_data )
    {
        VERBOSE_LEVEL(2) << endl << "	[sVO] Use precomputed data" << endl;

        // somebody already computed the features --> just copy them into this engine
		//	-- left and right features
		ASSERT_( request_data.precomputed_left_feats && request_data.precomputed_right_feats )
		ASSERT_( request_data.precomputed_left_feats->size() == request_data.precomputed_right_feats->size() )
		ASSERTMSG_( request_data.precomputed_left_feats->size() == nOctaves, "Number of octaves in precomputed data does not match number of octaves in the system (params_rectify.nOctaves)" )

		cur_imgpair.left.pyr_feats_kps.resize(nOctaves);
		cur_imgpair.left.pyr_feats_desc.resize(nOctaves);
		cur_imgpair.right.pyr_feats_kps.resize(nOctaves);
		cur_imgpair.right.pyr_feats_desc.resize(nOctaves);

		for( size_t octave = 0; octave < nOctaves; ++octave )
		{
			// left
			cur_imgpair.left.pyr_feats_kps[octave].resize( request_data.precomputed_left_feats->size() );
			std::copy( request_data.precomputed_left_feats->operator[](octave).begin(), request_data.precomputed_left_feats->operator[](octave).end(), cur_imgpair.left.pyr_feats_kps[octave].begin() );

			// right 
			cur_imgpair.right.pyr_feats_kps[octave].resize( request_data.precomputed_right_feats->size() );
			std::copy( request_data.precomputed_right_feats->operator[](octave).begin(), request_data.precomputed_right_feats->operator[](octave).end(), cur_imgpair.right.pyr_feats_kps[octave].begin() );

			//	-- left and right descriptors
			ASSERT_( request_data.precomputed_left_desc && request_data.precomputed_right_desc )
			
			request_data.precomputed_left_desc->operator[](octave).copyTo( cur_imgpair.left.pyr_feats_desc[octave] );
			request_data.precomputed_right_desc->operator[](octave).copyTo( cur_imgpair.right.pyr_feats_desc[octave] );
		} // end-for-octaves
    } // end-if
    else
    {
        VERBOSE_LEVEL(2) << endl << "	[sVO] Full process" << endl;
        stage2_detect_features( cur_imgpair.left, m_next_gui_info->img_left );
        stage2_detect_features( cur_imgpair.right, m_next_gui_info->img_right );
    } // end-else

	// fill the output structure
	result.detected_feats.resize(nOctaves);
	for(size_t i = 0; i < nOctaves; ++i )
	{
		result.detected_feats[i].first	= cur_imgpair.left.pyr_feats_kps[i].size();
		result.detected_feats[i].second = cur_imgpair.right.pyr_feats_kps[i].size();
	}

	// save data in text files
	if( params_general.vo_save_files )
	{
		FILE *f = mrpt::system::os::fopen( mrpt::format("%s/left_feats_%04d.txt",params_general.vo_out_dir.c_str(),m_it_counter).c_str(), "wt");
		for( size_t octave = 0; octave < nOctaves; ++octave )
		{
			for( vector<cv::KeyPoint>::iterator it = cur_imgpair.left.pyr_feats_kps[octave].begin(); it != cur_imgpair.left.pyr_feats_kps[octave].end(); ++it )
				mrpt::system::os::fprintf(f, "%d %.2f %.2f %.8f\n", static_cast<int>(octave), it->pt.x, it->pt.y, it->response );

			cv::FileStorage file( mrpt::format("%s/left_desc_o%02d_%04d.yml",params_general.vo_out_dir.c_str(), static_cast<int>(octave),m_it_counter).c_str(), cv::FileStorage::WRITE);
			file << "leftDescMat" << cur_imgpair.left.pyr_feats_desc[octave];
			file.release();
		}
		mrpt::system::os::fclose(f);

		f = mrpt::system::os::fopen( mrpt::format("%s/right_feats_%04d.txt",params_general.vo_out_dir.c_str(),m_it_counter).c_str(), "wt");
		for( size_t octave = 0; octave < nOctaves; ++octave )
		{
			for( vector<cv::KeyPoint>::iterator it = cur_imgpair.right.pyr_feats_kps[octave].begin(); it != cur_imgpair.right.pyr_feats_kps[octave].end(); ++it )
				mrpt::system::os::fprintf(f, "%d %.2f %.2f %.8f\n",  static_cast<int>(octave), it->pt.x, it->pt.y, it->response );

			cv::FileStorage file( mrpt::format("%s/right_desc_o%02d_%04d.yml",params_general.vo_out_dir.c_str(), static_cast<int>(octave),m_it_counter).c_str(), cv::FileStorage::WRITE);
			file << "rightDescMat" << cur_imgpair.right.pyr_feats_desc[octave];
			file.release();
		}
		mrpt::system::os::fclose(f);
	}
	VERBOSE_LEVEL(1) << "	done: " << endl;
	if( m_verbose_level >= 1 )
		for( size_t octave = 0; octave < nOctaves; ++octave )
			cout << "		Octave " << octave << " -> [" << cur_imgpair.left.pyr_feats_kps[octave].size() << "," << cur_imgpair.right.pyr_feats_kps[octave].size() << "] detected keypoints." << endl; 

	if( params_detect.detect_method == TDetectParams::dmORB )
	{	VERBOSE_LEVEL(2) << "	Current FAST threshold: " << m_current_fast_th << endl;	}

	VERBOSE_LEVEL(1) << endl;

	// -------------------------------------------
	// 3) Match L/R:
	// -------------------------------------------
	VERBOSE_LEVEL(1) << "[sVO] STEREO MATCHING... with " << st_match_method_str << endl;
	if( request_data.use_precomputed_data )
    {
        ASSERT_( request_data.precomputed_matches )

        // the features have been already matched --> just copy them into this engine
		//	-- matches
		cur_imgpair.lr_pairing_data.resize(nOctaves);
		for(size_t octave = 0; octave < nOctaves; ++octave)
		{
			cur_imgpair.lr_pairing_data[octave].matches_lr_dm.resize( request_data.precomputed_matches->operator[](octave).size() );
			std::copy( request_data.precomputed_matches->operator[](octave).begin(), request_data.precomputed_matches->operator[](octave).end(), cur_imgpair.lr_pairing_data[octave].matches_lr_dm.begin() );
		}

		// if wanted, copy also the IDs of the matches
		if( params_general.vo_use_matches_ids )
		{
			ASSERT_( request_data.precomputed_matches_ID )
			m_last_match_ID = 0;
			for( size_t octave = 0; octave < nOctaves; ++octave )
			{
				cur_imgpair.lr_pairing_data[octave].matches_IDs.resize( request_data.precomputed_matches_ID->operator[](octave).size() );
				std::copy( request_data.precomputed_matches_ID->operator[](octave).begin(), request_data.precomputed_matches_ID->operator[](octave).end(), cur_imgpair.lr_pairing_data[octave].matches_IDs.begin() );
				
				// get the maximum match ID
				m_last_match_ID = std::max(m_last_match_ID, *std::max_element(cur_imgpair.lr_pairing_data[octave].matches_IDs.begin(), cur_imgpair.lr_pairing_data[octave].matches_IDs.end()) );
			}
		} // end-if
		else
		{
			if( request_data.precomputed_matches_ID )
				SHOW_WARNING("Using precomputed data: Inserted IDs will be ignored as the VOdometer is set to not take control of them. To activate ID control, set 'vo_use_matches_ids' in the GENERAL section of the .ini file")
		} // end-else
    } // end-if
	else
	{
		if( m_reset )
		{
			// When RESET flag is set
			//		- clear the IDs from the previous frame and reset them to the range 0...N-1
			//		- set the maximum match IDs
			m_last_match_ID = 0;
			for( size_t octave = 0; octave < nOctaves; ++octave )
				for( size_t m = 0; m < m_prev_imgpair->lr_pairing_data[octave].matches_IDs.size(); ++m )
					m_prev_imgpair->lr_pairing_data[octave].matches_IDs[m] = m_last_match_ID++;
			m_reset = false;
			
			//		- set this frame as KF
			m_last_kf_max_id = m_last_match_ID-1;
		}

		stage3_match_left_right( cur_imgpair, request_data.stereo_cam );

	} // end-else

	// fill the result struct
	result.stereo_matches.resize(nOctaves);
	for( size_t octave = 0; octave < nOctaves; octave++)
		result.stereo_matches[octave] = cur_imgpair.lr_pairing_data[octave].matches_lr_dm.size();

	if( params_general.vo_save_files )
	{
		FILE *f = mrpt::system::os::fopen( mrpt::format("%s/matches_%04d.txt",params_general.vo_out_dir.c_str(),m_it_counter).c_str(), "wt");
		for( size_t octave = 0; octave < nOctaves; ++octave )
		{
			for( vector<cv::DMatch>::iterator it = m_current_imgpair->lr_pairing_data[octave].matches_lr_dm.begin(); it != m_current_imgpair->lr_pairing_data[octave].matches_lr_dm.end(); ++it )
				mrpt::system::os::fprintf(f, "%d %d %d %.2f\n",  static_cast<int>(octave), it->queryIdx, it->trainIdx, it->distance);
		}
		mrpt::system::os::fclose(f);
	}

	VERBOSE_LEVEL(1) << "	done: " << endl;
	if( m_verbose_level >= 1 )
	{
		for( size_t octave = 0; octave < nOctaves; ++octave )
			cout << "		Octave " << octave << " -> [" << cur_imgpair.lr_pairing_data[octave].matches_lr_dm.size() << "] stereo matches found." << endl; 
	
		if( params_lr_match.match_method == TLeftRightMatchParams::smDescBF || 
			params_lr_match.match_method == TLeftRightMatchParams::smDescRbR )
			cout << "	ORB threshold: " << m_current_orb_th << endl;				// this is dynamic
		else if( params_lr_match.match_method == TLeftRightMatchParams::smSAD )
			cout << "	SAD threshold: " << params_lr_match.sad_max_distance << endl;	// this is not dynamic (by now)
		
		cout << endl;
	}

	//	:: Only if this is not the first step:
	if( m_prev_imgpair.present() && m_current_imgpair.present() )
	{
		// -------------------------------------------
		// 4) Match consecutive stereo images:
		// -------------------------------------------
		TTrackingData tracking_data;
		TImagePairData & prev_imgpair = *m_prev_imgpair;

		VERBOSE_LEVEL(1) << "[sVO] TRACKING... with " << if_match_method_str;
        stage4_track( tracking_data, prev_imgpair, cur_imgpair );
		VERBOSE_LEVEL(1) << endl << "	done: " << endl;
		if( m_verbose_level >= 1 )
		{
			for( size_t octave = 0; octave < nOctaves; ++octave )
				cout << "		Octave " << octave << " -> [" << tracking_data.tracked_pairs[octave].size() << "] tracked feats." << endl; 
			cout << "		Total: " << m_num_tracked_pairs_from_last_frame << endl;
		}

		// -------------------------------------------
		// 4.1) Robustness stage --> Check tracking
		// -------------------------------------------
		if( m_num_tracked_pairs_from_last_frame < params_least_squares.bad_tracking_th )
		{
			m_error_in_tracking = true;
			m_error = result.error_code = voecBadTracking;
		}

		if( m_error != voecBadTracking ) 
		{
			// -------------------------------------------
			// 5) Optimize incremental pose:
			// -------------------------------------------
			VERBOSE_LEVEL(1) << "[sVO] LEAST SQUARES... ";
			stage5_optimize( tracking_data, request_data.stereo_cam, result );
			VERBOSE_LEVEL(1) << endl << "	done: [Resulting pose: " << result.outPose << "]" << endl;

		} // end-if

		if( result.error_code != voecNone )
		{
			DUMP_VO_ERROR_CODE(result.error_code)
		}
	}
	else
	{
		result.error_code = voecFirstIteration;
		result.valid = false;
	}

	VERBOSE_LEVEL(1) << endl << "[sVO] End visual odometry" << endl;

	// GUI stuff:
	if (params_gui.show_gui)
	{
		m_profiler.enter("processNewImagePair.send2gui");

		// Send information to the GUI & alternate between the two instances of the structure:
		m_gui_info_cs.enter();
		m_gui_info = m_next_gui_info;
		m_gui_info_cs.leave();

		m_gui_info_cache_next_index = m_gui_info_cache_next_index ? 0:1; // Alternate
		m_next_gui_info = &m_gui_info_cache[m_gui_info_cache_next_index];

		// Upon first iteration, launch the GUI thread:
		if (m_thread_gui.isClear())
		{
			m_thread_gui = mrpt::system::createThreadFromObjectMethod( this, &CStereoOdometryEstimator::thread_gui );
		}

		m_profiler.leave("processNewImagePair.send2gui");
	}
	m_profiler.leave("processNewImagePair");

	// update iteration counter
	if( !request_data.repeat )
		++m_it_counter;

	if( params_general.vo_pause_it )
		mrpt::system::pause();
}
