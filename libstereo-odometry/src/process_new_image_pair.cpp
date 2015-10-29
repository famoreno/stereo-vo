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
	string detect_method_str;
	switch( params_detect.detect_method )
	{
		case TDetectParams::dmORB : detect_method_str = "[ORB]"; break;
		case TDetectParams::dmFASTER : detect_method_str = "[FASTER]"; break;
		case TDetectParams::dmFAST_ORB : detect_method_str = "[FAST + ORB]"; break;
	}
	// ----------------------------------------------------------------

	m_profiler.enter("processNewImagePair");

	ASSERTMSG_( request_data.stereo_imgs.present(),"Pointer 'request_data.stereo_imgs' must be set to stereo observation data!" )

	// -------------------------------------------
	// 0) Shift the list of two previous images:
	// -------------------------------------------
	if( !request_data.repeat && this->m_error != voecBadTracking && this->m_error != voecBadCondNumber ) // !this->m_error_in_tracking )
		m_prev_imgpair = m_current_imgpair;   // these are smart pointers, so it only implies copying a pointer

	if( request_data.repeat )
		cout << "[sVO] Repeating... " << endl;

	this->m_error_in_tracking = false;
	this->m_error = voecNone;

	// -------------------------------------------
	// 1) Prepare new image pair:
	// -------------------------------------------
	m_current_imgpair = TImagePairDataPtr( new TImagePairData() );
	TImagePairData & cur_imgpair = *m_current_imgpair;

    VERBOSE_LEVEL(1) << "[sVO] RECTIFYING... ";
	this->stage1_prepare_rectify( request_data, cur_imgpair );
	VERBOSE_LEVEL(1) << endl << "	done" << endl;
	
	// "copyFastFrom" has "move semantics", so the original input images are no longer available to subsequent stages:
	if( !request_data.repeat && !this->m_error_in_tracking )
	{
		mrpt::obs::CObservationStereoImages *stObs = const_cast<mrpt::obs::CObservationStereoImages*>(request_data.stereo_imgs.pointer());
		m_next_gui_info->timestamp = stObs->timestamp;
		m_next_gui_info->img_left.swap( stObs->imageLeft );
		m_next_gui_info->img_right.swap( stObs->imageRight );
	}
	else
		cout << "no swapping" << endl;

	if( m_prev_imgpair )
	{
		VERBOSE_LEVEL(2) << "[sVO] Image Timestamps -- PRE:" << m_prev_imgpair->timestamp << " and CUR: " << m_next_gui_info->timestamp << endl;
	}
	else
	{
		VERBOSE_LEVEL(2) << "[sVO] Image Timestamps -- PRE: None and CUR: " << m_next_gui_info->timestamp << endl;
	}
    // -------------------------------------------
	// 2) Detect features:
	// -------------------------------------------
    VERBOSE_LEVEL(1) << "[sVO] DETECTING FEATURES... " << detect_method_str;
    if( request_data.use_precomputed_data )
    {
        VERBOSE_LEVEL(2) << endl << "   [sVO] Use precomputed data" << endl;

        // somebody already computed the ORB features --> just copy them into this engine
		//	-- left and right features
		ASSERT_( request_data.precomputed_left_feats && request_data.precomputed_right_feats )
        cur_imgpair.left.orb_feats.resize( request_data.precomputed_left_feats->size() );
		std::copy( request_data.precomputed_left_feats->begin(), request_data.precomputed_left_feats->end(), cur_imgpair.left.orb_feats.begin() );

		cur_imgpair.right.orb_feats.resize( request_data.precomputed_right_feats->size() );
        std::copy( request_data.precomputed_right_feats->begin(), request_data.precomputed_right_feats->end(), cur_imgpair.right.orb_feats.begin() );

        //	-- left and right descriptors
		ASSERT_( request_data.precomputed_left_desc && request_data.precomputed_right_desc )
        request_data.precomputed_left_desc->copyTo( cur_imgpair.left.orb_desc );
        request_data.precomputed_right_desc->copyTo( cur_imgpair.right.orb_desc );

        const size_t nPyrs = cur_imgpair.left.pyr.images.size();
        cur_imgpair.left.pyr_feats.resize(nPyrs);
        cur_imgpair.right.pyr_feats.resize(nPyrs);

    } // end-if
    else
    {
        VERBOSE_LEVEL(2) << endl << "   [sVO] Full process" << endl;
        this->stage2_detect_features( cur_imgpair.left, m_next_gui_info->img_left );
        this->stage2_detect_features( cur_imgpair.right, m_next_gui_info->img_right );
    } // end-else

	result.detected_feats.first = cur_imgpair.left.orb_feats.size();
	result.detected_feats.second = cur_imgpair.right.orb_feats.size();

	if( this->params_general.vo_save_files )
	{
		FILE *f1 = mrpt::system::os::fopen( mrpt::format("%s/left_feats_%04d.txt",params_general.vo_out_dir.c_str(),m_it_counter).c_str(), "wt");
		for( vector<cv::KeyPoint>::iterator it = cur_imgpair.left.orb_feats.begin(); it != cur_imgpair.left.orb_feats.end(); ++it )
			mrpt::system::os::fprintf(f1, "%.2f %.2f\n", it->pt.x, it->pt.y );
		mrpt::system::os::fclose(f1);

		FILE *f2 = mrpt::system::os::fopen( mrpt::format("%s/right_feats_%04d.txt",params_general.vo_out_dir.c_str(),m_it_counter).c_str(), "wt");
		for( vector<cv::KeyPoint>::iterator it = cur_imgpair.right.orb_feats.begin(); it != cur_imgpair.right.orb_feats.end(); ++it )
			mrpt::system::os::fprintf(f2, "%.2f %.2f\n", it->pt.x, it->pt.y );
		mrpt::system::os::fclose(f2);
	}
	if( params_detect.detect_method == TDetectParams::dmORB || params_detect.detect_method == TDetectParams::dmFAST_ORB )
	{	VERBOSE_LEVEL(1) << endl << "	done: (FAST Th: " << m_current_fast_th << "): Detected keypoints: [" << cur_imgpair.left.orb_feats.size() << "," << cur_imgpair.right.orb_feats.size() << "]" << endl; }
	else
	{	VERBOSE_LEVEL(1) << endl << "	done: Detected keypoints: [" << cur_imgpair.left.pyr_feats[0].size() << "," << cur_imgpair.right.pyr_feats[0].size() << "]" << endl; }

	// -------------------------------------------
	// 3) Match L/R:
	// -------------------------------------------
	VERBOSE_LEVEL(1) << "[sVO] STEREO MATCHING... " << detect_method_str;
	if( request_data.use_precomputed_data )
    {
        ASSERT_( request_data.precomputed_matches )

        // the ORB features have been already matched --> just copy them into this engine
		//	-- matches
        cur_imgpair.orb_matches.resize( request_data.precomputed_matches->size() );
        std::copy( request_data.precomputed_matches->begin(), request_data.precomputed_matches->end(), cur_imgpair.orb_matches.begin() );

		// if wanted, copy also the IDs of the matches
		if( params_general.vo_use_matches_ids )
		{
			ASSERT_( request_data.precomputed_matches_ID )
			cur_imgpair.lr_pairing_data[0].matches_IDs.resize( request_data.precomputed_matches_ID->size() );
			std::copy( request_data.precomputed_matches_ID->begin(), request_data.precomputed_matches_ID->end(), cur_imgpair.lr_pairing_data[0].matches_IDs.begin() );
			//cur_imgpair.orb_matches_ID.resize( request_data.precomputed_matches_ID->size() );
			//std::copy( request_data.precomputed_matches_ID->begin(), request_data.precomputed_matches_ID->end(), cur_imgpair.orb_matches_ID.begin() );

			this->m_kf_ids.resize( request_data.precomputed_matches_ID->size() );
			std::copy( request_data.precomputed_matches_ID->begin(), request_data.precomputed_matches_ID->end(), this->m_kf_ids.begin() );

			// set the maximum match ID and the maximum match ID from the last KF
			//this->m_last_match_ID = this->m_kf_max_match_ID = *cur_imgpair.orb_matches_ID.rbegin();					// must be the last
			this->m_last_match_ID = this->m_kf_max_match_ID = *cur_imgpair.lr_pairing_data[0].matches_IDs.rbegin();		// must be the last
		} // end-if
		else
		{
			if( request_data.precomputed_matches_ID )
				SHOW_WARNING("Using precomputed data: Inserted IDs will be ignored as the VOdometer is set to not take control of them. To activate ID control, set 'vo_use_matches_ids' in the GENERAL section of the .ini file")
		} // end-else
    } // end-if
	else
	{
		if( this->m_reset )
		{
			// in RESET flag is set
			//		clear the IDs from the previous frame and reset them to the range 0...N-1
			//		set the maximum match IDs
			const size_t num_p_matches = this->m_prev_imgpair->orb_matches.size();
			this->m_last_match_ID = this->m_kf_max_match_ID = num_p_matches-1;
			this->m_kf_ids.resize( num_p_matches );
			for( size_t m = 0; m < num_p_matches; ++m )
				this->m_kf_ids[m] = this->m_prev_imgpair->lr_pairing_data[0].matches_IDs[m] = m; //this->m_kf_ids[m] = this->m_prev_imgpair->orb_matches_ID[m] = m;
			this->m_reset = false;													// unset RESET flag
		}

		this->stage3_match_left_right( cur_imgpair, request_data.stereo_cam );

	} // end-else

	result.stereo_matches = cur_imgpair.orb_matches.size();

	if( this->params_general.vo_save_files )
	{
		FILE *f = mrpt::system::os::fopen( mrpt::format("%s/matches_%04d.txt",params_general.vo_out_dir.c_str(),m_it_counter).c_str(), "wt");
		for( vector<cv::DMatch>::iterator it = this->m_current_imgpair->orb_matches.begin(); it != this->m_current_imgpair->orb_matches.end(); ++it )
			mrpt::system::os::fprintf(f, "%d %d %.2f\n",it->queryIdx, it->trainIdx, it->distance);
		mrpt::system::os::fclose(f);
	}

	if( params_detect.detect_method == TDetectParams::dmORB || params_detect.detect_method == TDetectParams::dmFAST_ORB )
	{	VERBOSE_LEVEL(1) << endl << "	done: (ORB Th:" << m_current_orb_th << ") [" << cur_imgpair.orb_matches.size() << " matches]" << endl; }
	else
	{	VERBOSE_LEVEL(1) << endl << "	done: [" << cur_imgpair.lr_pairing_data[0].matches_lr.size() << " matches]" << endl; }

	//	:: Only if this is not the first step:
	if( m_prev_imgpair.present() && m_current_imgpair.present() )
	{
		// -------------------------------------------
		// 4) Match consecutive stereo images:
		// -------------------------------------------
		TTrackingData tracking_data;
		TImagePairData &prev_imgpair = *m_prev_imgpair;

		VERBOSE_LEVEL(1) << "[sVO] TRACKING... " << detect_method_str;
        this->stage4_track( tracking_data, prev_imgpair, cur_imgpair );
		VERBOSE_LEVEL(1) << endl << "	done: [" << this->m_num_tracked_pairs_from_last_frame << "/"
			<< this->m_num_tracked_pairs_from_last_kf << " tracked from last Frame/KF]" << endl;

		// -------------------------------------------
		// 4.1) Robustness stage --> Check tracking
		// -------------------------------------------
		if( this->m_num_tracked_pairs_from_last_frame < this->params_least_squares.bad_tracking_th )
		{
			this->m_error_in_tracking = true;
			this->m_error = result.error_code = voecBadTracking;
		}

		if( this->m_error != voecBadTracking ) // !this->m_error_in_tracking )
		{
			// -------------------------------------------
			// 5) Optimize incremental pose:
			// -------------------------------------------
			VERBOSE_LEVEL(1) << "[sVO] LEAST SQUARES... " << detect_method_str;
			this->stage5_optimize( tracking_data, request_data.stereo_cam, result );
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

	if( this->params_general.vo_pause_it )
		mrpt::system::pause();
}
