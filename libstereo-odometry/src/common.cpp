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

// #include <libstereo-odometry.h>
#include "internal_libstereo-odometry.h"

using namespace rso;

/** Default constructor */
CStereoOdometryEstimator::CStereoOdometryEstimator() :
	m_profiler(true),
	m_lastID(0),
	m_num_tracked_pairs_from_last_kf(0),
	m_reset(false),
	m_last_match_ID(0),
	m_kf_max_match_ID(0),
	m_current_fast_th(20),
	m_current_orb_th(60),
    m_error_in_tracking(false),
	m_error(voecNone),
	m_verbose_level(1),
	m_it_counter(0),
	m_threads_must_close(false),
	m_win_keyhit(0),
	m_next_gui_info(NULL),
	m_gui_info_cache_next_index(0),
	m_gui_info(NULL)
{
	// Set alternating structure cache pointing to the first slot:
	m_next_gui_info = &m_gui_info_cache[m_gui_info_cache_next_index];
	m_last_computed_pose.resize(6,0);
}

CStereoOdometryEstimator::~CStereoOdometryEstimator()
{
	m_threads_must_close=true;
	if (!m_thread_gui.isClear())
	{
		VERBOSE_LEVEL(1) << "[CStereoOdometryEstimator:dtor] Waiting for all threads to end...\n";
		mrpt::system::joinThread(m_thread_gui);
		VERBOSE_LEVEL(1) << "[CStereoOdometryEstimator:dtor] All threads finished.\n";
	}
}

// Auxiliary function for debugging asserts:
void rso::debug_pre_assert(const char* failed_test)
{
	std::cerr << "[srba::debug_pre_assert] " << failed_test << std::endl;
}

CStereoOdometryEstimator::TLeastSquaresParams::TLeastSquaresParams() :
	use_robust_kernel				( true ),
	kernel_param					( 3. ),
	max_iters						( 100 ),
	initial_max_iters				( 10 ),
	min_mod_out_vector				( 1e-3 ),
	std_noise_pixels				( 1. ),
	max_incr_cost					( 3 ),
	residual_threshold				( 10. ),
	bad_tracking_th					( 5 ),
	use_previous_pose_as_initial	( true ),
	use_custom_initial_pose			( false )
{
}

CStereoOdometryEstimator::TInterFrameMatchingParams::TInterFrameMatchingParams() {}

/*--------------------------------------------------------------------
						m_dump_keypoints_to_stream
----------------------------------------------------------------------*/
bool CStereoOdometryEstimator::m_dump_keypoints_to_stream(
				std::ofstream				& stream,
				const vector<cv::KeyPoint>	& keypoints,
				const cv::Mat				& descriptors )
{
	/* FORMAT
	- # of features in image
		- feat x coord
		- feat y coord
		- feat response
		- feat size
		- feat angle
		- feat octave
		- feat class_id
	- # of dimensions of descriptors (D): rows, cols and type
		feat descriptor d_0 ... d_{D-1}
	*/
	if( !stream.is_open() )
		return false;

	size_t num_kp = keypoints.size();
	stream.write( reinterpret_cast<char*>(&num_kp), sizeof(size_t));

	for( size_t f = 0; f < keypoints.size(); ++f )
	{
		stream.write( (char*)(&(keypoints[f].pt.x)), sizeof(float) );
		stream.write( (char*)(&(keypoints[f].pt.y)), sizeof(float) );
		stream.write( (char*)(&(keypoints[f].response)), sizeof(float) );
		stream.write( (char*)(&(keypoints[f].size)), sizeof(float) );
		stream.write( (char*)(&(keypoints[f].angle)), sizeof(float) );
		stream.write( (char*)(&(keypoints[f].octave)), sizeof(int) );
		stream.write( (char*)(&(keypoints[f].class_id)), sizeof(int) );
	} // end-for-keypoints

	int drows = descriptors.rows, dcols = descriptors.cols, dtype = descriptors.type();
	stream.write( (char*)&drows, sizeof(int) );
	stream.write( (char*)&dcols, sizeof(int) );
	stream.write( (char*)&dtype, sizeof(int) );

	for( cv::MatConstIterator_<uchar> it = descriptors.begin<uchar>(); it != descriptors.end<uchar>(); ++it )
	{
		uchar value = *it;
		stream.write( (char*)&value, sizeof(uchar) );
	}

	return true;
} // end-m_dump_keypoints_to_stream

/*--------------------------------------------------------------------
						m_dump_matches_to_stream
----------------------------------------------------------------------*/
bool CStereoOdometryEstimator::m_dump_matches_to_stream(
				std::ofstream				& stream,
				const vector<cv::DMatch>	& matches,
				const vector<size_t>		& matches_ids )
{
	/* FORMAT
	- # of matches
		- match id
		- queryIdx
		- trainIdx
		- distance
	*/
	if( !stream.is_open() )
		return false;

	size_t num_m = matches.size(), num_m_id = matches_ids.size();
	stream.write( (char*)&num_m, sizeof(size_t) );
	stream.write( (char*)&num_m_id, sizeof(size_t) );
	const bool add_ids = num_m == num_m_id;
	for( size_t m = 0; m < matches.size(); ++m )
	{
		if( add_ids ) stream.write( (char*)&(matches_ids[m]), sizeof(size_t) );
		stream.write( (char*)&(matches[m].queryIdx), sizeof(matches[m].queryIdx) );
		stream.write( (char*)&(matches[m].trainIdx), sizeof(matches[m].trainIdx) );
		stream.write( (char*)&(matches[m].distance), sizeof(matches[m].distance) );
		stream.write( (char*)&(matches[m].imgIdx), sizeof(matches[m].imgIdx) );
	} // end-for-matches

	return true;
} // end-m_dump_matches_to_stream

/*--------------------------------------------------------------------
						m_load_keypoints_from_stream
----------------------------------------------------------------------*/
bool CStereoOdometryEstimator::m_load_keypoints_from_stream(
		std::ifstream			& stream,
		vector<cv::KeyPoint>	& keypoints,
		cv::Mat					& descriptors )
{
	/* FORMAT
	- # of features in image
	- # of dimensions of descriptors (D)
		- feat x coord
		- feat y coord
		- feat response
		- feat scale
		- feat orientation
		- feat descriptor d_0 ... d_{D-1}
	*/
	if( !stream.is_open() )
		return false;

	size_t num_kp;
	stream.read( (char*)&num_kp, sizeof(size_t) );
	keypoints.resize( num_kp );

	for( size_t f = 0; f < keypoints.size(); ++f )
	{
		stream.read( (char*)(&(keypoints[f].pt.x)), sizeof(float) );
		stream.read( (char*)(&(keypoints[f].pt.y)), sizeof(float) );
		stream.read( (char*)(&(keypoints[f].response)), sizeof(float) );
		stream.read( (char*)(&(keypoints[f].size)), sizeof(float) );
		stream.read( (char*)(&(keypoints[f].angle)), sizeof(float) );
		stream.read( (char*)(&(keypoints[f].octave)), sizeof(int) );
		stream.read( (char*)(&(keypoints[f].class_id)), sizeof(int) );

	} // end-for-keypoints
	int drows,dcols,dtype;
	stream.read( (char*)&drows, sizeof(int) );
	stream.read( (char*)&dcols, sizeof(int) );
	stream.read( (char*)&dtype, sizeof(int) );
	descriptors.create(drows,dcols,dtype);

	for( cv::MatIterator_<uchar> it = descriptors.begin<uchar>(); it != descriptors.end<uchar>(); ++it ) // stream << *it;
	{
		uchar value;
		stream.read( (char*)&value, sizeof(uchar) );
		*it = value;
	}
	return true;
} // end-loadKeyPointsFromStream

/*--------------------------------------------------------------------
						m_load_matches_from_stream
----------------------------------------------------------------------*/
bool CStereoOdometryEstimator::m_load_matches_from_stream(
		std::ifstream		& stream,
		vector<cv::DMatch>	& matches,
		vector<size_t>		& matches_ids )
{
	/* FORMAT
	- # of matches
		- match id
		- queryIdx
		- trainIdx
		- distance
	*/
	if( !stream.is_open() )
		return false;

	size_t num_matches, num_matches_id;
	stream.read( (char*)&num_matches, sizeof(size_t) );
	stream.read( (char*)&num_matches_id, sizeof(size_t) );
	matches.resize( num_matches );
	matches_ids.resize( num_matches_id );
	const bool add_ids = num_matches == num_matches_id;
	for( size_t m = 0; m < matches.size(); ++m )
	{
		if( add_ids ) stream.read( (char*)&(matches_ids[m]), sizeof(size_t) );
		stream.read( (char*)&(matches[m].queryIdx), sizeof(matches[m].queryIdx) );
		stream.read( (char*)&(matches[m].trainIdx), sizeof(matches[m].trainIdx) );
		stream.read( (char*)&(matches[m].distance), sizeof(matches[m].distance) );
		stream.read( (char*)&(matches[m].imgIdx), sizeof(matches[m].imgIdx) );
	} // end-for-matches

	return true;
} // end-loadMatchesFromStream

/*--------------------------------------------------------------------
						loadStateFromFile
----------------------------------------------------------------------*/
bool CStereoOdometryEstimator::loadStateFromFile( const string & filename )
{
	std::ifstream vo_state_file_stream( filename.c_str(), ios::in | ios::binary );	// read
	if( !vo_state_file_stream.is_open()/*.fileOpenCorrectly()*/ )
		return false;

	size_t npyr;
	vo_state_file_stream.read( (char*)&npyr, sizeof(size_t) );

	// create previous and current data storage
	this->m_prev_imgpair	= TImagePairDataPtr( new TImagePairData() );
	this->m_current_imgpair = TImagePairDataPtr( new TImagePairData() );

	this->m_prev_imgpair->left.pyr.images.resize(npyr);
	this->m_prev_imgpair->left.pyr_feats.resize(npyr);
	this->m_prev_imgpair->left.pyr_feats_desc.resize(npyr);
	this->m_prev_imgpair->left.pyr_feats_index.resize(npyr);

	this->m_prev_imgpair->right.pyr.images.resize(npyr);
	this->m_prev_imgpair->right.pyr_feats.resize(npyr);
	this->m_prev_imgpair->right.pyr_feats_desc.resize(npyr);
	this->m_prev_imgpair->right.pyr_feats_index.resize(npyr);

	this->m_current_imgpair->left.pyr.images.resize(npyr);
	this->m_current_imgpair->left.pyr_feats.resize(npyr);
	this->m_current_imgpair->left.pyr_feats_desc.resize(npyr);
	this->m_current_imgpair->left.pyr_feats_index.resize(npyr);

	this->m_current_imgpair->right.pyr.images.resize(npyr);
	this->m_current_imgpair->right.pyr_feats.resize(npyr);
	this->m_current_imgpair->right.pyr_feats_desc.resize(npyr);
	this->m_current_imgpair->right.pyr_feats_index.resize(npyr);

	/**/
	// prev data:
	// left
	if( !m_load_keypoints_from_stream( vo_state_file_stream, this->m_prev_imgpair->left.orb_feats, this->m_prev_imgpair->left.orb_desc ) )
	{
		cout << "ERROR while saving the state -- PRE left keypoints could not be saved. Closed stream?" << endl;
		return false;
	}
	// right
	if( !m_load_keypoints_from_stream( vo_state_file_stream, this->m_prev_imgpair->right.orb_feats, this->m_prev_imgpair->right.orb_desc ) )
	{
		cout << "ERROR while saving the state -- PRE right keypoints could not be saved. Closed stream?" << endl;
		return false;
	}
	// matches
	if( !m_load_matches_from_stream( vo_state_file_stream, this->m_prev_imgpair->orb_matches, this->m_prev_imgpair->orb_matches_ID ) )
	{
		cout << "ERROR while saving the state -- PRE matches could not be saved. Closed stream?" << endl;
		return false;
	}
	/**/

	// current data:
	// left
	if( !m_load_keypoints_from_stream( vo_state_file_stream, this->m_current_imgpair->left.orb_feats, this->m_current_imgpair->left.orb_desc ) )
	{
		cout << "ERROR while saving the state -- CUR left keypoints could not be saved. Closed stream?" << endl;
		return false;
	}
	// right
	if( !m_load_keypoints_from_stream( vo_state_file_stream, this->m_current_imgpair->right.orb_feats, this->m_current_imgpair->right.orb_desc ) )
	{
		cout << "ERROR while saving the state -- CUR right keypoints could not be saved. Closed stream?" << endl;
		return false;
	}
	// matches
	if( !m_load_matches_from_stream( vo_state_file_stream, this->m_current_imgpair->orb_matches, this->m_current_imgpair->orb_matches_ID ) )
	{
		cout << "ERROR while saving the state -- CUR matches could not be saved. Closed stream?" << endl;
		return false;
	}

	// parameters
	vo_state_file_stream.read( (char*)&this->m_reset, sizeof(bool) );
	vo_state_file_stream.read( (char*)&this->m_lastID, sizeof(size_t) );
	vo_state_file_stream.read( (char*)&this->m_num_tracked_pairs_from_last_kf, sizeof(size_t) );
	vo_state_file_stream.read( (char*)&this->m_num_tracked_pairs_from_last_frame, sizeof(size_t) );

	size_t v_s;
	vo_state_file_stream.read( (char*)&v_s, sizeof(size_t) );

	vo_state_file_stream.read( (char*)&this->m_last_match_ID, sizeof(size_t) );
	vo_state_file_stream.read( (char*)&this->m_kf_max_match_ID, sizeof(size_t) );

	vo_state_file_stream.close();
	return true;
}

/*--------------------------------------------------------------------
						getChangeInPose
----------------------------------------------------------------------*/
bool CStereoOdometryEstimator::getChangeInPose(
	const vector_index_pairs_t				& tracked_pairs,
	const vector<cv::DMatch>				& pre_matches,
	const vector<cv::DMatch>				& cur_matches,
	const vector<cv::KeyPoint>				& pre_left_feats,
	const vector<cv::KeyPoint>				& pre_right_feats,
	const vector<cv::KeyPoint>				& cur_left_feats,
	const vector<cv::KeyPoint>				& cur_right_feats,
	const mrpt::utils::TStereoCamera		& stereo_camera,
	TStereoOdometryResult					& result,				// output
	const vector<double>					& ini_estimation )
{
	const size_t octave = 0; // only one octave

	// prepare input
	CStereoOdometryEstimator::TTrackingData out_tracked_feats;
	out_tracked_feats.tracked_pairs.resize(1);

	out_tracked_feats.tracked_pairs[octave].resize( tracked_pairs.size() );
	std::copy( tracked_pairs.begin(), tracked_pairs.end(), out_tracked_feats.tracked_pairs[octave].begin() );

	TImagePairData prev_imgpair, cur_imgpair;

	prev_imgpair.lr_pairing_data.resize(1);
	prev_imgpair.lr_pairing_data[octave].matches_lr_dm.resize( pre_matches.size() );			
	std::copy( pre_matches.begin(), pre_matches.end(), prev_imgpair.lr_pairing_data[octave].matches_lr_dm.begin() );
	
	cur_imgpair.lr_pairing_data.resize(1);
	cur_imgpair.lr_pairing_data[octave].matches_lr_dm.resize( cur_matches.size() );			
	std::copy( cur_matches.begin(), cur_matches.end(), cur_imgpair.lr_pairing_data[octave].matches_lr_dm.begin() );

	prev_imgpair.left.pyr_feats_kps.resize(1);
	prev_imgpair.left.pyr_feats_kps[octave].resize( pre_left_feats.size() );	
	std::copy( pre_left_feats.begin(), pre_left_feats.end(), prev_imgpair.left.pyr_feats_kps[octave].begin() );
	
	prev_imgpair.right.pyr_feats_kps.resize(1);
	prev_imgpair.right.pyr_feats_kps[octave].resize( pre_right_feats.size() );	
	std::copy( pre_right_feats.begin(), pre_right_feats.end(), prev_imgpair.right.pyr_feats_kps[octave].begin() );

	cur_imgpair.left.pyr_feats_kps.resize(1);
	cur_imgpair.left.pyr_feats_kps[octave].resize( cur_left_feats.size() );		
	std::copy( cur_left_feats.begin(), cur_left_feats.end(), cur_imgpair.left.pyr_feats_kps[octave].begin() );
	
	cur_imgpair.right.pyr_feats_kps.resize(1);
	cur_imgpair.right.pyr_feats_kps[octave].resize( cur_right_feats.size() );	
	std::copy( cur_right_feats.begin(), cur_right_feats.end(), cur_imgpair.right.pyr_feats_kps[octave].begin() );

	prev_imgpair.img_h = stereo_camera.leftCamera.nrows;
	prev_imgpair.img_w = stereo_camera.leftCamera.ncols;

	out_tracked_feats.prev_imgpair = & prev_imgpair;
	out_tracked_feats.cur_imgpair  = & cur_imgpair;

	// get change in pose
	stage5_optimize( out_tracked_feats, stereo_camera, result, ini_estimation );

	return result.valid;

} // end-getChangeInPose

void CStereoOdometryEstimator::getProjectedCoords(
			const vector<cv::DMatch>					& pre_matches,						// all the matches in the previous keyframe
			const vector<cv::KeyPoint>					& pre_left_feats,
			const vector<cv::KeyPoint>					& pre_right_feats,					// all the features in the previuous keyframe
			const vector< pair<int,float> >				& other_matches_tracked,			// tracking info for OTHER matches [size=A]
			const mrpt::utils::TStereoCamera			& stereo_camera,					// stereo camera intrinsic and extrinsic parameters
			const CPose3D								& change_pose,						// the estimated change in pose between keyframes
			vector< pair<TPixelCoordf,TPixelCoordf> >	& pro_pre_feats )					// [out] coords of the features in the left image [size=B (only for those whose 'other_matches_tracked' entry is false]
{
	// 3D landmark prediction
    vector<TPoint3D> lmks;
	lmks.reserve( other_matches_tracked.size() );
	for( size_t m = 0; m < other_matches_tracked.size(); ++m)
    {
		if( other_matches_tracked[m].first != -1 )
			continue;

        // indexes of the matches in the previous step
        const size_t mpreIdx = m;

        TSimpleFeature featL, featR;

        // left and right feature indexes
        const size_t lpreIdx = pre_matches[mpreIdx].queryIdx;
        const size_t rpreIdx = pre_matches[mpreIdx].trainIdx;

        const double ul  = pre_left_feats[lpreIdx].pt.x;
        const double vl  = pre_left_feats[lpreIdx].pt.y;
        const double ur  = pre_right_feats[rpreIdx].pt.x;

        const double cul = stereo_camera.leftCamera.cx();
        const double cvl = stereo_camera.leftCamera.cy();
        const double fl  = stereo_camera.leftCamera.fx();

        const double cur = stereo_camera.rightCamera.cx();
        const double fr  = stereo_camera.rightCamera.fx();

        const double disparity = fl*(cur-ur)+fr*(ul-cul);
        const double baseline = stereo_camera.rightCameraPose[0];

        const double b_d = baseline/disparity;

		lmks.push_back( TPoint3D(b_d*fr*(ul-cul),b_d*fr*(vl-cvl),b_d*fl*fr) ); // (X,Y,Z)
    } // end for m

    // landmark projections on the image
	vector<double> delta_pose(6);	// [w1 w2 w3 t1 t2 t3]
	CPose3D auxPose(change_pose);
	auxPose.inverse();
	CPose3DRotVec auxPoseRVT(auxPose);
	delta_pose[0] = auxPoseRVT.m_rotvec[0];	delta_pose[1] = auxPoseRVT.m_rotvec[1];	delta_pose[2] = auxPoseRVT.m_rotvec[2];
	delta_pose[3] = auxPoseRVT.m_coords[0];	delta_pose[4] = auxPoseRVT.m_coords[1];	delta_pose[5] = auxPoseRVT.m_coords[2];
	vector<Eigen::MatrixXd> out_jacobian;
    m_pinhole_stereo_projection( lmks, stereo_camera, delta_pose, pro_pre_feats, out_jacobian );

} // end-getProjectedCoords

/*--------------------------------------------------------------------
						saveStateToFile
----------------------------------------------------------------------*/
bool CStereoOdometryEstimator::saveStateToFile( const string & filename )
{
	std::ofstream vo_state_file_stream( filename.c_str(), ios::out | ios::binary );			// write
	if( !vo_state_file_stream.is_open() )
		return false;

	/**/
	if( !this->m_prev_imgpair )
		this->m_prev_imgpair = TImagePairDataPtr( new TImagePairData() );

	if( !this->m_current_imgpair )
		this->m_current_imgpair = TImagePairDataPtr( new TImagePairData() );

	const size_t npyr = this->m_prev_imgpair->left.pyr.images.size();
	vo_state_file_stream.write( (char*)&npyr, sizeof(size_t) );

	// prev data:
	// left
	if( !m_dump_keypoints_to_stream( vo_state_file_stream, this->m_prev_imgpair->left.orb_feats, this->m_prev_imgpair->left.orb_desc ) )
	{
		cout << "ERROR while saving the state -- PRE left keypoints could not be saved. Closed stream?" << endl;
		return false;
	}
	// right
	if( !m_dump_keypoints_to_stream( vo_state_file_stream, this->m_prev_imgpair->right.orb_feats, this->m_prev_imgpair->right.orb_desc ) )
	{
		cout << "ERROR while saving the state -- PRE right keypoints could not be saved. Closed stream?" << endl;
		return false;
	}
	// matches
	if( !m_dump_matches_to_stream( vo_state_file_stream, this->m_prev_imgpair->orb_matches, this->m_prev_imgpair->orb_matches_ID ) )
	{
		cout << "ERROR while saving the state -- PRE matches could not be saved. Closed stream?" << endl;
		return false;
	}
	/**/

	// current data:
	// left
	if( !m_dump_keypoints_to_stream( vo_state_file_stream, this->m_current_imgpair->left.orb_feats, this->m_current_imgpair->left.orb_desc ) )
	{
		cout << "ERROR while saving the state -- CUR left keypoints could not be saved. Closed stream?" << endl;
		return false;
	}
	// right
	if( !m_dump_keypoints_to_stream( vo_state_file_stream, this->m_current_imgpair->right.orb_feats, this->m_current_imgpair->right.orb_desc ) )
	{
		cout << "ERROR while saving the state -- CUR right keypoints could not be saved. Closed stream?" << endl;
		return false;
	}
	// matches
	if( !m_dump_matches_to_stream( vo_state_file_stream, this->m_current_imgpair->orb_matches, this->m_current_imgpair->orb_matches_ID ) )
	{
		cout << "ERROR while saving the state -- CUR matches could not be saved. Closed stream?" << endl;
		return false;
	}

	// parameters
	vo_state_file_stream.write( (char*)&this->m_reset, sizeof(bool) );
	vo_state_file_stream.write( (char*)&this->m_lastID, sizeof(size_t) );
	vo_state_file_stream.write( (char*)&this->m_num_tracked_pairs_from_last_kf, sizeof(size_t) );
	vo_state_file_stream.write( (char*)&this->m_num_tracked_pairs_from_last_frame, sizeof(size_t) );

	vo_state_file_stream.write( (char*)&this->m_last_match_ID, sizeof(size_t) );
	vo_state_file_stream.write( (char*)&this->m_kf_max_match_ID, sizeof(size_t) );

	vo_state_file_stream.close();
	return true;
}
