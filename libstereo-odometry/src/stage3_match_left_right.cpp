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

#define USE_MATCHER 0			// 0 : Bruteforce -- 1 : Standard with ORB descriptors -- 2 : SAD

#include <libstereo-odometry.h>
#include "internal_libstereo-odometry.h"
#include <mrpt/gui/CDisplayWindow.h>

using namespace rso;
using namespace std;
using namespace mrpt;
using namespace mrpt::vision;
using namespace mrpt::utils;

using namespace cv;

typedef struct TFeat2MatchInfo
{
    bool assigned;
    size_t idxL,idxMatch;
    double distance;	

    TFeat2MatchInfo() : assigned(false), idxL(0), idxMatch(0), distance(0.0) {}

} TFeat2MatchInfo;

CStereoOdometryEstimator::TLeftRightMatchParams::TLeftRightMatchParams() :
	match_method(smSAD),
	sad_max_distance(200),
	sad_max_ratio(0.5),
	orb_max_distance(40),
	orb_min_th(30), orb_max_th(100),
	enable_robust_1to1_match(false),
	rectified_images(false),
	max_y_diff(0),
	min_z(0.3), max_z(5)
{
}

/**  Stage3 operations:
  *   - Match left and right keypoints at each scale (this should work well for stereo matching)
  */
void CStereoOdometryEstimator::stage3_match_left_right( CStereoOdometryEstimator::TImagePairData & imgpair, const TStereoCamera & stereoCamera )
{
	m_profiler.enter("_stg3");

	const size_t nOctaves = imgpair.left.pyr_feats.size();	// '1' for ORB features, 'n' for the rest
	const bool use_ids = params_general.vo_use_matches_ids && !this->m_prev_imgpair.present(); /*first iteration*/

	// Alloc lists of pairings for each octave
    imgpair.lr_pairing_data.resize(nOctaves);

	size_t nPotentialMatches    = 0;
    size_t nMatches             = 0;
    vector<size_t> nMatchesPerOctave(nOctaves);

    // Search for pairings:
    m_profiler.enter("stg3.find_pairings");

	// **********************************************************************
	// Descriptor based brute force
	//		- ORB : Hamming distance between descriptors
	// **********************************************************************
	if( params_lr_match.match_method == TLeftRightMatchParams::smDescBF )
	{
		const size_t nOctaves = imgpair.left.pyr.images.size();

		// perform match
	    cv::BFMatcher matcher(cv::NORM_HAMMING,false);
		for( size_t octave = 0; octave < nOctaves; ++octave )
		{
			matcher.match( 
				imgpair.left.pyr_feats_desc[octave],				// query
				imgpair.right.pyr_feats_desc[octave],				// train 
				imgpair.lr_pairing_data[octave].matches_lr_dm );	// size of query

			// shorcuts
			const TKeyPointList  & leftKps	= imgpair.left.pyr_feats_kps[octave];
			const TKeyPointList  & rightKps = imgpair.right.pyr_feats_kps[octave];
			vector<DMatch> & matches		= imgpair.lr_pairing_data[octave].matches_lr_dm;
			//Mat & leftMatches				= imgpair.left.pyr_feats_desc[octave];
			Mat & rightMatches				= imgpair.right.pyr_feats_desc[octave];

			// DEBUG:
			if( params_general.vo_debug )
			{
				// save matches
				FILE *fm = mrpt::system::os::fopen( mrpt::format("%s/matches_before_filter%04d.txt",params_general.vo_out_dir.c_str(),m_it_counter).c_str(), "wt");
				for( vector<DMatch>::iterator it = matches.begin(); it != matches.end(); ++it )
				{
					mrpt::system::os::fprintf(fm, "%d %.2f %.2f %d %.2f %.2f %.2f\n", 
						it->queryIdx,
						leftKps[it->queryIdx].pt.x,
						leftKps[it->queryIdx].pt.y,
						it->trainIdx,
						rightKps[it->trainIdx].pt.x,
						rightKps[it->trainIdx].pt.y,
						it->distance );
				} // end-for
				mrpt::system::os::fclose(fm);
			}

			/**/
			// 1-to-1 matchings
			if( params_lr_match.enable_robust_1to1_match )
			{
				// for each right feature: 'distance' and 'left idx'
				vector< pair< double, size_t > >  right_cand( rightMatches.rows, make_pair(-1.0,0) );

				// loop over the matches
				for( size_t k = 0; k < matches.size(); ++k )
				{
					const size_t idR = matches[k].trainIdx;
					if( right_cand[idR].first < 0 || right_cand[idR].first > matches[k].distance )
					{
						right_cand[idR].first  = matches[k].distance;
						right_cand[idR].second = matches[k].queryIdx;
					}
				} // end-for

				vector<cv::DMatch>::iterator itMatch;
				for( itMatch = matches.begin(); itMatch != matches.end();  )
				{
					if( itMatch->queryIdx != int(right_cand[ itMatch->trainIdx ].second) )
						itMatch = matches.erase( itMatch );
					else
						++itMatch;
				} // end-for
			} // end-1-to-1 matchings
			/**/

			// reserve space for IDs in case we use them (and this is the first iteration, otherwise this will be done in next step)
			if( use_ids )
				imgpair.lr_pairing_data[octave].matches_IDs.reserve( matches.size() ); // imgpair.orb_matches_ID.reserve( matches.size() );

			const double min_disp = 1;											// stereoCamera.rightCameraPose[0]*stereoCamera.leftCamera.fx()/params_lr_match.max_z;
			const double max_disp = imgpair.left.pyr.images[octave].getWidth(); // stereoCamera.rightCameraPose[0]*stereoCamera.leftCamera.fx()/params_lr_match.min_z;

			// keep only those that fulfill the epipolar and distance constraints
		    vector<cv::DMatch>::iterator itM = matches.begin();
			while( itM != matches.end() )
			{
				const int diff = leftKps[itM->queryIdx].pt.y-rightKps[itM->trainIdx].pt.y;
				const int disp = leftKps[itM->queryIdx].pt.x-rightKps[itM->trainIdx].pt.x;
				if( std::abs(diff) > params_lr_match.max_y_diff || itM->distance > m_current_orb_th ||
					disp < min_disp || disp > max_disp )
				{
					itM = matches.erase(itM);
				}
				else
				{
					++itM;
					if( use_ids )																	
						imgpair.lr_pairing_data[octave].matches_IDs.push_back( m_last_match_ID++ ); // imgpair.orb_matches_ID.push_back( this->m_last_match_ID++ );				
				}
			} // end-while
			
		} // end-octaves
	} // end brute-force (only for ORB)
	
	// **********************************************************************
	// Sum of absolute differences or ORB distance Row-by-Row (with tolerance)
	//		- SAD : 8x8 patches (will use SSE instructions if available)
	//		- ORB : Hamming distance between descriptors
	// **********************************************************************
	else if( params_lr_match.match_method == TLeftRightMatchParams::smSAD || 
		params_lr_match.match_method == TLeftRightMatchParams::smDescRbR )
	{
		// Set minimal response for a keypoint to be considered
		double minimum_response = 0;
		if( params_detect.detect_method == TDetectParams::dmKLT ) 
			minimum_response = params_detect.minimum_KLT_response;
		else if( params_detect.detect_method == TDetectParams::dmORB )  
			minimum_response = params_detect.minimum_ORB_response;

		// Define parameters
		double max_ratio = 1;
		size_t max_distance = 0;
		if( params_lr_match.match_method == TLeftRightMatchParams::smSAD )
		{
			max_ratio		= params_lr_match.sad_max_ratio;
			max_distance	= size_t(params_lr_match.sad_max_distance);
		}
		else
		{
			max_distance	= size_t(params_lr_match.orb_max_distance);
		}

		// Process every octave ('1' for ORB, 'n' for the rest)
		for( size_t octave = 0; octave < nOctaves; ++octave)
		{
			// The list of keypoints
			const TKeyPointList & feats_left	= imgpair.left.pyr_feats_kps[octave];
			const TKeyPointList & feats_right	= imgpair.right.pyr_feats_kps[octave];

			// References to the feature indices by row:
			const vector_size_t & idxL = imgpair.left.pyr_feats_index[octave];
			const vector_size_t & idxR = imgpair.right.pyr_feats_index[octave];

			// Get references to the descriptors lists (for OBR only)
			Mat desc_left			= imgpair.left.pyr_feats_desc[octave];
			Mat desc_right			= imgpair.right.pyr_feats_desc[octave];

			ASSERTDEB_(idxL.size()==idxR.size())
			const size_t nRowsMax = idxL.size();
		
			// 121 robust stereo matching ------------------------
			const uint32_t MAX_D = std::numeric_limits<uint32_t>::max();
			vector<int> left_matches_idxs( feats_left.size(), INVALID_IDX );									// for storing left-right associations
			vector< pair<int,uint32_t> > right_feat_assign( feats_right.size(), make_pair(INVALID_IDX,MAX_D) );
			// ---------------------------------------------------

			// Get information from the images
			const CImage imgL = imgpair.left.pyr.images[octave];
			const CImage imgR = imgpair.right.pyr.images[octave];
			const TImageSize max_pt( imgL.getWidth()-4-1, imgL.getHeight()-4-1 );

			const unsigned char *img_data_L = imgL.get_unsafe(0,0);
			const unsigned char *img_data_R = imgR.get_unsafe(0,0);
			const size_t img_stride			= imgpair.left.pyr.images[octave].getRowStride();
			ASSERTDEB_(img_stride == imgpair.right.pyr.images[octave].getRowStride())
		
			// DEBUG: 
			FILE *f = mrpt::system::os::fopen("dist.txt","wt");

			// Prepare output
			imgpair.lr_pairing_data[octave].matches_lr_dm.reserve( feats_left.size() ); // maximum number of matches: number of left features
			const int max_disparity = static_cast<int>(imgL.getWidth()*0.7);

			// Match features row by row:
			for( size_t y = 0; y < nRowsMax-1; y++ )
			{
				// select current rows (with user-defined tolerance)
				const size_t idx_feats_L0 = idxL[y]; const size_t idx_feats_L1 = idxL[y+1];
				const size_t min_row_right = max(int(0),int(y)-int(round(params_lr_match.max_y_diff)));
				const size_t max_row_right = min(size_t(imgL.getHeight()-1),size_t(y)+size_t(round(params_lr_match.max_y_diff)));
				const size_t idx_feats_R0 = idxR[min_row_right]; const size_t idx_feats_R1 = idxR[max_row_right];

				// The number of feats in the row "y" in each image:
				const size_t nFeatsL = idx_feats_L1 - idx_feats_L0;
				const size_t nFeatsR = idx_feats_R1 - idx_feats_R0;

				if( (nFeatsL==0) || (nFeatsR==0) )
					continue; // No way we can match a damn thing here!

				for( size_t idx_feats_L = idx_feats_L0; idx_feats_L < idx_feats_L1; idx_feats_L++ )
				{
					const KeyPoint & featL	= imgpair.left.pyr_feats_kps[octave][idx_feats_L];		// left keypoint

					// two lowest distances and lowest distance index
					uint32_t min_1, min_2;
					min_1 = min_2 = std::numeric_limits<uint32_t>::max();
					int min_idx = INVALID_IDX;

					for( size_t idx_feats_R = idx_feats_R0; idx_feats_R < idx_feats_R1; idx_feats_R++ )
					{
						const KeyPoint & featR	= imgpair.right.pyr_feats_kps[octave][idx_feats_R];	// right keypoint

						// Reponse filter
						if( featL.response < minimum_response || featR.response < minimum_response )
							continue;

						// Disparity filter
						const int disparity = featL.pt.x-featR.pt.x;
						if( disparity < 1 || disparity > max_disparity )
							continue;

						// Too-close-to-border filter (only for SAD)
						MRPT_TODO("Optimize too-close-border");
						if( (params_lr_match.match_method == TLeftRightMatchParams::smSAD) &&
							(featL.pt.x < 3 || featR.pt.x < 3 ||
							featL.pt.y < 3 || featR.pt.y < 3 ||
							featL.pt.x > max_pt.x || featR.pt.x > max_pt.x ||
							featL.pt.y > max_pt.y || featR.pt.y > max_pt.y )
							)
							continue;
 
						// We've got a potential match --> compute distance
						nPotentialMatches++;

						size_t dist;
						if( params_lr_match.match_method == TLeftRightMatchParams::smSAD )
						{
							// SAD
							// WARNING: Uncomment this profiler entries only for debugging purposes, don't
							//  leave for production code since it's called so many times it will become a
							//  performance issue:
							//m_profiler.enter("stg3.compute_SAD8");

							const uint32_t d = rso::compute_SAD8(
								img_data_L,img_data_R,img_stride,
								TPixelCoord(featL.pt.x,featL.pt.y),TPixelCoord(featR.pt.x,featR.pt.y));
							
							dist = size_t(d);

							//m_profiler.leave("stg3.compute_SAD8");

						} // end-if
						else
						{
							// ORB descriptor Hamming distance
							uint8_t d = 0;
							for( uint8_t k = 0; k < desc_left.cols; ++k )
							{
								uint8_t x_or = desc_left.at<uint8_t>(idx_feats_L,k) ^ desc_right.at<uint8_t>(idx_feats_R,k);
								uint8_t count;								// from : Wegner, Peter (1960), "A technique for counting ones in a binary computer", Communications of the ACM 3 (5): 322, doi:10.1145/367236.367286
								for( count = 0; x_or; count++ )				// ...
									x_or &= x_or-1;							// ...
								d += count;
							}

							dist = size_t(d);
						} // end-else

						if( dist > max_distance )
							continue; // bad match
							
						// keep the closest
						if( dist < min_1 )
						{
							min_2    = min_1;
							min_1    = dist;
							min_idx  = idx_feats_R;
						}
						else if( dist < min_2 )
							min_2 = dist;

						const double this_ratio = 1.0*min_1/min_2;
						if( this_ratio > max_ratio )
							continue;

						// DEBUG:
						mrpt::system::os::fprintf(f,"%d,%.2f,%.2f,%.2f,%.2f,%d\n",static_cast<int>(octave),featL.pt.x,featL.pt.y,featR.pt.x,featR.pt.y,static_cast<int>(dist));

					} // end for feats_R

					// We've got a potential match
					if( min_idx != INVALID_IDX )
					{
						if( params_lr_match.enable_robust_1to1_match )
						{
							// check if the right feature has been already assigned
							if( right_feat_assign[min_idx].first == INVALID_IDX )
							{
								// set the new match
								left_matches_idxs[idx_feats_L]		= min_idx;
								right_feat_assign[min_idx].first	= idx_feats_L;		// will keep the **BEST** match
								right_feat_assign[min_idx].second	= min_1;
							}
							else if( min_1 < right_feat_assign[min_idx].second )
							{
								// undo the previous one and set the new one
								left_matches_idxs[right_feat_assign[min_idx].first] = INVALID_IDX;
								left_matches_idxs[idx_feats_L]			= min_idx;
								right_feat_assign[min_idx].first		= idx_feats_L;		// will keep the **BEST** match
								right_feat_assign[min_idx].second		= min_1;
							}
						} // end-if
						else
						{
							// check if the right feature has been already assigned
							if( right_feat_assign[min_idx].first == INVALID_IDX )
							{
								left_matches_idxs[idx_feats_L]			= min_idx;
								right_feat_assign[min_idx].first		= idx_feats_L;		// will keep the **FIRST** match
								right_feat_assign[min_idx].second		= min_1;
							}
						} // end--else
					} // end-if- INVALID_IDX
				} // end--left-for
			} // end--rows-for

			// DEBUG:
			mrpt::system::os::fclose(f);

			// Create output matches
			const size_t out_size = left_matches_idxs.size();
			imgpair.lr_pairing_data[octave].matches_lr_dm.reserve( out_size );
			for(size_t i = 0; i < out_size; ++i )
			{
				if( left_matches_idxs[i] != INVALID_IDX )
				{
					const size_t fr = left_matches_idxs[i];
					const float d = float(right_feat_assign[fr].second);
					imgpair.lr_pairing_data[octave].matches_lr_dm.push_back( DMatch(i,fr,d) );

					if( use_ids )
						imgpair.lr_pairing_data[octave].matches_IDs.push_back( m_last_match_ID++ );
				}
			} // end--for
		} // end--octave-for

		// Final stats
		size_t nMatches = 0;
		for(uint8_t octave = 0; octave < nOctaves; ++octave)
		{
			nMatchesPerOctave[octave] = imgpair.lr_pairing_data[octave].matches_lr_dm.size();
            nMatches += nMatchesPerOctave[octave];
		}
	} // end--SAD-matching

	// Filter for robust pairings ----------------------------------------
	// MRPT_TODO("Filter out spurious with a Homography model or such")

	// Build the row-sorted index of pairings ------------------------------
	m_profiler.enter("stg3.pairings-row-index");
	for( size_t octave = 0; octave < nOctaves; octave++ )
	{
		const TDMatchList	& octave_pairings		= imgpair.lr_pairing_data[octave].matches_lr_dm;
		vector<size_t>		& octave_pairings_idx	= imgpair.lr_pairing_data[octave].matches_lr_row_index;
		const TKeyPointList	& octave_feats_left		= imgpair.left.pyr_feats_kps[octave];

		size_t idx = 0;
		const size_t imgH   = imgpair.left.pyr.images[octave].getHeight();
		const size_t nFeats = octave_pairings.size();

		octave_pairings_idx.resize(imgH+1);  // the last entry, [nRows] = total # of feats
		for( size_t y = 0; y < imgH; y++ )
		{
			octave_pairings_idx[y] = idx;
			// Move on "idx" until we reach (at least) the next row:
			while( idx < nFeats && octave_feats_left[octave_pairings[idx].queryIdx].pt.y <= int(y) ) { idx++; }
		}
		octave_pairings_idx[imgH] = octave_feats_left.size();
	}
	m_profiler.leave("stg3.pairings-row-index");

	m_profiler.leave("stg3.find_pairings");

	// Draw pairings -----------------------------------------------------
	if (params_gui.draw_lr_pairings)
	{
	    m_profiler.enter("stg3.send2gui");
		m_next_gui_info->draw_pairings_all.clear(); // "soft" clear (without dealloc)
        for( size_t octave = 0; octave < nOctaves; octave++ )
        {
			const TKeyPointList	& fL				= imgpair.left.pyr_feats_kps[octave];
            const TKeyPointList	& fR				= imgpair.right.pyr_feats_kps[octave];
            const TDMatchList	& octave_pairings	= imgpair.lr_pairing_data[octave].matches_lr_dm;
            const size_t nFeats						= octave_pairings.size();

            const size_t i0 = m_next_gui_info->draw_pairings_all.size();
            m_next_gui_info->draw_pairings_all.resize(i0 + nFeats);
            for( size_t i = 0; i < nFeats; ++i )
            {
                // (X,Y) coordinates in the left/right images for this pairing:
                m_next_gui_info->draw_pairings_all[i0+i].first	= TPixelCoord(fL[octave_pairings[i].queryIdx].pt.x,fL[octave_pairings[i].queryIdx].pt.y);
                m_next_gui_info->draw_pairings_all[i0+i].second = TPixelCoord(fR[octave_pairings[i].trainIdx].pt.x,fR[octave_pairings[i].trainIdx].pt.y);
            }
		} // end-for-octaves
        m_profiler.leave("stg3.send2gui");
	}

	m_profiler.leave("_stg3");

    m_next_gui_info->text_msg_from_lr_match = mrpt::format(
	"L/R match: %u potential pairings | %u accepted (octaves: ",
	static_cast<unsigned int>(nPotentialMatches),
	static_cast<unsigned int>(nMatches)
	);

    for( size_t octave = 0; octave < nOctaves; octave++)
        m_next_gui_info->text_msg_from_lr_match+=mrpt::format("%u,",static_cast<unsigned int>(nMatchesPerOctave[octave]));
    m_next_gui_info->text_msg_from_lr_match+=string(")");
}


