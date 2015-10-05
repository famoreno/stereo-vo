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

#define USE_MATCHER 0			// 0 : Bruteforce -- 1 : Standard

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
	maximum_SAD(200),
	enable_robust_1to1_match(false),
	max_SAD_ratio(0.5),
	rectified_images(false),
	max_y_diff(0),
	orb_max_distance(40),
	min_z(0.3), max_z(5),
	orb_min_th(30), orb_max_th(100)
{
}

/**  Stage2 operations:
  *   - Detect features on each image and on each scale.
  */
void CStereoOdometryEstimator::stage3_match_left_right( CStereoOdometryEstimator::TImagePairData & imgpair, const TStereoCamera & stereoCamera )
{
	m_profiler.enter("_stg3");

	const size_t nOctaves = imgpair.left.pyr_feats.size();

	// Alloc lists of pairings for each octave
    imgpair.lr_pairing_data.resize(nOctaves);

	size_t nPotentialMatches    = 0;
    size_t nMatches             = 0;
    vector<size_t> nMatchesPerOctave(nOctaves);

    // Search for pairings:
    m_profiler.enter("stg3.find_pairings");

	if( params_detect.detect_method == TDetectParams::dmORB || params_detect.detect_method == TDetectParams::dmFAST_ORB )
	{
#if USE_MATCHER == 0			// USE OPENCV'S BRUTE-FORCE MATCHER
        // CTimeLogger tLog;
		// tLog.enter("match");

		// perform match
	    cv::BFMatcher matcher(cv::NORM_HAMMING,false);
        matcher.match( imgpair.left.orb_desc /*query*/, imgpair.right.orb_desc /*train*/, imgpair.orb_matches /* size of query*/);

		if( params_general.vo_debug )
		{
			// save matches
			FILE *fm = mrpt::system::os::fopen( mrpt::format("%s/matches_before_filter%04d.txt",params_general.vo_out_dir.c_str(),m_it_counter).c_str(), "wt");
			for( vector<DMatch>::iterator it = imgpair.orb_matches.begin(); it != imgpair.orb_matches.end(); ++it )
			{
				mrpt::system::os::fprintf(fm, "%d %.2f %.2f %d %.2f %.2f %.2f\n", 
					it->queryIdx,
					imgpair.left.orb_feats[it->queryIdx].pt.x,
					imgpair.left.orb_feats[it->queryIdx].pt.y,
					it->trainIdx,
					imgpair.right.orb_feats[it->trainIdx].pt.x,
					imgpair.right.orb_feats[it->trainIdx].pt.y,
					it->distance );
			} // end-for
			mrpt::system::os::fclose(fm);
		}

		/**/
		// 1-to-1 matchings
		if( params_lr_match.enable_robust_1to1_match )
		{
			// for each right feature: 'distance' and 'left idx'
			const size_t right_size = imgpair.right.orb_desc.rows;
			vector< pair< double, size_t > >  right_cand( right_size, make_pair(-1.0,0) );

			// loop over the matches
			for( size_t k = 0; k < imgpair.orb_matches.size(); ++k )
			{
				const size_t idR = imgpair.orb_matches[k].trainIdx;
				if( right_cand[idR].first < 0 || right_cand[idR].first > imgpair.orb_matches[k].distance )
				{
					right_cand[idR].first  = imgpair.orb_matches[k].distance;
					right_cand[idR].second = imgpair.orb_matches[k].queryIdx;
				}
			} // end-for

			vector<cv::DMatch>::iterator itMatch;
			for( itMatch = imgpair.orb_matches.begin(); itMatch != imgpair.orb_matches.end();  )
			{
				if( itMatch->queryIdx != int(right_cand[ itMatch->trainIdx ].second) )
					itMatch = imgpair.orb_matches.erase( itMatch );
				else
					++itMatch;
			} // end-for
		} // end-1-to-1 matchings
		/**/

		const bool use_ids = params_general.vo_use_matches_ids && !this->m_prev_imgpair.present();

        // the ids of the matches
		/*if( params_general.vo_use_matches_ids )
			imgpair.orb_matches_ID.reserve( imgpair.orb_matches.size() );*/

        // reserve space for IDs in case we use them (and this is the first iteration, otherwise this will be done in next step)
		if( use_ids )
			imgpair.orb_matches_ID.reserve( imgpair.orb_matches.size() );

		const double min_disp = stereoCamera.rightCameraPose[0]*stereoCamera.leftCamera.fx()/params_lr_match.max_z;
		const double max_disp = stereoCamera.rightCameraPose[0]*stereoCamera.leftCamera.fx()/params_lr_match.min_z;

		// keep only those that fulfill the epipolar and distance constraints
        vector<cv::DMatch>::iterator itM = imgpair.orb_matches.begin();
        while( itM != imgpair.orb_matches.end() )
        {
            const int diff = imgpair.left.orb_feats[itM->queryIdx].pt.y-imgpair.right.orb_feats[itM->trainIdx].pt.y;
            const int disp = imgpair.left.orb_feats[itM->queryIdx].pt.x-imgpair.right.orb_feats[itM->trainIdx].pt.x;
			if( std::abs(diff) > params_lr_match.max_y_diff || itM->distance > m_current_orb_th ||
				disp < min_disp || disp > max_disp )
			{
                itM = imgpair.orb_matches.erase(itM);
			}
			else
            {
                ++itM;
				if( use_ids )																	
					imgpair.orb_matches_ID.push_back( this->m_last_match_ID++ );				
            }
        } // end-while

		if( use_ids )
		{
			// save this ids to get tracking info for them
			this->m_kf_ids.resize( imgpair.orb_matches_ID.size() );
			std::copy( imgpair.orb_matches_ID.begin(), imgpair.orb_matches_ID.end(), this->m_kf_ids.begin() );
		}

		// tLog.leave("match");
		// cout << "match: " << tLog.getMeanTime("match") << endl;

#elif USE_MATCHER == 1														// STANDARD MATCHING PROCESS: 1 by 1 with restrictions (possibly faster than BFMatcher)
		const Mat & desc_left					= imgpair.left.orb_desc;
		const Mat & desc_right					= imgpair.right.orb_desc;
		const vector<KeyPoint> & feats_left		= imgpair.left.orb_feats;
		const vector<KeyPoint> & feats_right	= imgpair.right.orb_feats;

		imgpair.orb_matches.reserve( feats_left.size() );	// maximum number of matches: number of left features

		vector< pair<int,size_t> > right_feat_assign( feats_right.size(), make_pair(-1,255) );

		// CTimeLogger tLog;
		// tLog.enter("match");
		for( size_t fl = 0; fl < feats_left.size(); ++fl )
		{
			for( size_t fr = 0; fr < feats_right.size(); ++fr )
			{
				// filter 1: epipolar
				if( mrpt::utils::abs_diff(feats_left[fl].pt.y,feats_right[fr].pt.y) > params_lr_match.max_y_diff )
					continue;

				// filter 2: disparity
				const double disp = feats_left[fl].pt.x - feats_right[fr].pt.y;
				if( disp < 1 || disp > params_lr_match.max_disparity )
					continue;

				// filter 3: orb hamming distance
				// Descriptors XOR + Hamming weight
				uint8_t distance = 0;
				for( uint8_t k = 0; k < desc_left.cols; ++k )
				{
					uint8_t x_or = desc_left.at<uint8_t>(fl,k) ^ desc_right.at<uint8_t>(fr,k);
					uint8_t count;								// from : Wegner, Peter (1960), "A technique for counting ones in a binary computer", Communications of the ACM 3 (5): 322, doi:10.1145/367236.367286
					for( count = 0; x_or; count++ )				// ...
						x_or &= x_or-1;							// ...
					distance += count;
				}

				if( distance > m_current_orb_th )
					continue;

				// filter 4: (opt) 1 to 1 robust match -- possible not too useful for stereo --> consider removal
				if( params_lr_match.enable_robust_1to1_match )
				{
					// check if the right feature has been already assigned
					if( distance < right_feat_assign[fr].second )
					{
						right_feat_assign[fr].first		= fl;
						right_feat_assign[fr].second	= distance;
					}
				} // end-if
				else
				{
					imgpair.orb_matches.push_back( DMatch(fl,fr,float(distance)) );
					if( params_general.vo_use_matches_ids )
						imgpair.orb_matches_ID.push_back( ++this->m_last_match_ID );
				}
			} // end-for
		} // end-for

		// if 'robust 1to1 matches' is set, then create output matches (it is ordered by right --train-- idx --> consider order it the other way around)
		if( params_lr_match.enable_robust_1to1_match )
		{
			for( size_t k = 0; k < right_feat_assign.size(); ++k )
			{
				if( right_feat_assign[k].first != -1 )
				{
					imgpair.orb_matches.push_back( DMatch(right_feat_assign[k].first,k,float(right_feat_assign[k].second)) );
					if( params_general.vo_use_matches_ids )
						imgpair.orb_matches_ID.push_back( ++this->m_last_match_ID );
				}
			}
		} // end-if

		// tLog.leave("match");
		// cout << "match: " << tLog.getMeanTime("match") << endl;
#endif
	} // end method orb feats
	else if( params_detect.detect_method == TDetectParams::dmFASTER )
    {
        const double minimum_KLT	    = params_detect.minimum_KLT;
        const double maximum_SAD	    = params_lr_match.maximum_SAD;
        const double max_SAD_ratio	    = params_lr_match.max_SAD_ratio;

        // imgpair.left.pyr.images[0].saveToFile("left.jpg");
        // imgpair.right.pyr.images[0].saveToFile("right.jpg");

        // FILE *fmatch = mrpt::system::os::fopen("fmatch.txt","wt");
        // cout << endl;
        for (size_t octave=0;octave<nOctaves;octave++)
        {
            // get pointers to the image data of the grayscale pyramids for quick SAD computation:
            const CImage imgL = imgpair.left.pyr.images[octave];
            const CImage imgR = imgpair.right.pyr.images[octave];

            const int max_disparity = static_cast<int>(imgL.getWidth()*0.7);

            // Maximum feature position for not being out of the image when computing its SAD.
            const TImageSize max_pt( imgL.getWidth()-4-1, imgL.getHeight()-4-1 );

            const unsigned char *img_data_L = imgL.get_unsafe(0,0);
            const unsigned char *img_data_R = imgR.get_unsafe(0,0);
            const size_t img_stride = imgpair.left.pyr.images[octave].getRowStride();
            ASSERTDEB_(img_stride == imgpair.right.pyr.images[octave].getRowStride())

            // References to the feature indices by row:
            const vector_size_t & idxL = imgpair.left.pyr_feats_index[octave];
            const vector_size_t & idxR = imgpair.right.pyr_feats_index[octave];

            ASSERTDEB_(idxL.size()==idxR.size())
            const size_t nRowsMax = idxL.size()-1;

            vector_index_pairs_t & octave_pairings = imgpair.lr_pairing_data[octave].matches_lr;

            // to allow robust 1to1 matches
            vector<TFeat2MatchInfo> feat2MatchInfo( imgpair.left.pyr_feats[octave].size() );
            vector<bool> matchToDelete( imgpair.left.pyr_feats[octave].size(), false );
            size_t idx_match = 0;

            // Match features row by row:
            for (size_t y=0;y<nRowsMax-1;y++)
            {
                const size_t idx_feats_L0 = idxL[y]; const size_t idx_feats_L1 = idxL[y+1];
                const size_t idx_feats_R0 = idxR[y]; const size_t idx_feats_R1 = idxR[y+1];

                // The number of feats in the row "y" in each image:
                const size_t nFeatsL=idx_feats_L1-idx_feats_L0;
                const size_t nFeatsR=idx_feats_R1-idx_feats_R0;
    //			cout << idx_feats_L0 << " to " << idx_feats_L1 << endl;
    //			cout << idx_feats_R0 << " to " << idx_feats_R1 << endl;

                if (!nFeatsL || !nFeatsR)
                    continue; // No way we can match a damn thing here!

                for (size_t idx_feats_L=idx_feats_L0;idx_feats_L<idx_feats_L1;idx_feats_L++)
                {
                    const TSimpleFeature &featL = imgpair.left.pyr_feats[octave][idx_feats_L];
                    uint32_t minSAD_1, minSAD_2;
                    minSAD_1 = minSAD_2 = std::numeric_limits<uint32_t>::max();
                    size_t minSAD_idx = 0;

                    for (size_t idx_feats_R=idx_feats_R0;idx_feats_R<idx_feats_R1;idx_feats_R++)
                    {
                        const TSimpleFeature &featR = imgpair.right.pyr_feats[octave][idx_feats_R];
                        const int disparity = featL.pt.x-featR.pt.x;
                        if (disparity<1 || disparity>max_disparity)
                            continue; // Not a valid potential pairing.

                        // Check if the two features match: SAD matching
                        if (featL.pt.x>=3 && featR.pt.x>=3 &&
                            featL.pt.y>=3 && featR.pt.y>=3 &&
                            featL.pt.x<=max_pt.x && featR.pt.x<=max_pt.x &&
                            featL.pt.y<=max_pt.y && featR.pt.y<=max_pt.y &&
                            featL.response>=minimum_KLT && featR.response>=minimum_KLT
                            )
                        {
                            nPotentialMatches++;
    //						cout << "potm" << endl;

                            // WARNING: Uncomment this profiler entries only for debugging purposes, don't
                            //  leave for production code since it's called so many times it will become a
                            //  performance issue:
                            //m_profiler.enter("stg3.compute_SAD8");

                            const uint32_t SAD = rso::compute_SAD8(
                                img_data_L,img_data_R,img_stride,
                                featL.pt,featR.pt);

    //						cout << idx_feats_L << "[" << featL.pt.x << "," << featL.pt.y << "," << featL.response << "]- "
    //							 << idx_feats_R << "[" << featR.pt.x << "," << featR.pt.y << "," << featR.response << "] O("
    //							 << int(octave) << ") -> " << SAD << endl;

                            //m_profiler.leave("stg3.compute_SAD8");

                            if (SAD < maximum_SAD)
                            {
                                if( SAD < minSAD_1 )
                                {
                                    minSAD_2    = minSAD_1;
                                    minSAD_1    = SAD;
                                    minSAD_idx  = idx_feats_R;
                                }
                                else if( SAD < minSAD_2 )
                                    minSAD_2 = SAD;
                            } // end-if SAD < maximum_SAD
                        } // end-if check feats properties
    //					else
    //                        cout << featL.response << "," << featR.response << endl;
                    } // end for feats_R

                    const double SAD_ratio = 1.0*minSAD_1/minSAD_2;
                    if( SAD_ratio < max_SAD_ratio )	// Accept this only if the ratio between the SAD is below a threshold
                    {
                        if( params_lr_match.enable_robust_1to1_match )
                        {
                            // check that feat in 2 has not been already assigned to another one
                            if( feat2MatchInfo[minSAD_idx].assigned )
                            {
                                // show info:
                                /** /
                                cout << "Right " << minSAD_idx << " with "
                                     << feat2MatchInfo[minSAD_idx].idxL << " (" << feat2MatchInfo[minSAD_idx].distance << ") and "
                                     << idx_feats_L << " (" << minSAD_1 << ")";
                                /**/
                                if( minSAD_1 < feat2MatchInfo[minSAD_idx].distance )
                                {
                                    // cout << " --> delete: " << feat2MatchInfo[minSAD_idx].idxL << " match: " << feat2MatchInfo[minSAD_idx].idxMatch;
                                    matchToDelete[ feat2MatchInfo[minSAD_idx].idxMatch ] = true; // mark old match to be deleted

                                    // update the information for this pairing
                                    feat2MatchInfo[minSAD_idx].idxL         = idx_feats_L;
                                    feat2MatchInfo[minSAD_idx].idxMatch     = idx_match;
                                    feat2MatchInfo[minSAD_idx].distance     = minSAD_1;
                                }
                                else
                                {
                                    // cout << " --> delete: " << idx_feats_L << " match: " << idx_match;
                                    matchToDelete[ idx_match ] = true;  // mark the new match to be deleted

                                    // keep the old information for his pairing
                                }
                                // cout << endl;
                            }
                            else
                            {
                                // set info for the new pairing
                                feat2MatchInfo[minSAD_idx].assigned  = true;
                                feat2MatchInfo[minSAD_idx].idxL      = idx_feats_L;
                                feat2MatchInfo[minSAD_idx].idxMatch  = idx_match;
                                feat2MatchInfo[minSAD_idx].distance  = minSAD_1;
                            }

                        } // end-enable_robust_1to1_match

                        // we've got a l-r match!
                        octave_pairings.push_back( std::make_pair(idx_feats_L, minSAD_idx) );
                        idx_match++;

                        // update right feats ids --> we'll keep the one from the left image
                        imgpair.right.pyr_feats[octave][minSAD_idx].ID =
                            imgpair.left.pyr_feats[octave][idx_feats_L].ID;

    //                    cout << minSAD_1 << endl;

                        // OK, accept the pairing:
                        //cout << "MATCHED: " << idx_feats_L << " & " << idx_feats_R << endl;
                        /** /
                        const TSimpleFeature &featR = imgpair.right.pyr_feats[octave][minSAD_idx];
                        mrpt::system::os::fprintf(fmatch,"%d %d %d %.3f %d %d %d %.3f %d %d\n",
                        int(idx_feats_L), featL.pt.x, featL.pt.y, featL.response,
                        int(minSAD_idx), featR.pt.x, featR.pt.y, featR.response,
                        int(octave), minSAD_1);
                        /**/
                    } // end-if
                } // end for feats_L
            } // end for "y"

            size_t nMatchesPre = 0;
            for(uint8_t o = 0; o < imgpair.lr_pairing_data.size(); ++o)
                nMatchesPre += imgpair.lr_pairing_data[octave].matches_lr.size();

            cout << "pre: " << nMatchesPre << endl;

            // delete not 1to1 robust pairings:
            if( params_lr_match.enable_robust_1to1_match )
            {
                size_t counter = 0;
                vector_index_pairs_t::iterator it = octave_pairings.begin();
                while( it != octave_pairings.end() )
                {
                    if( matchToDelete[counter] )
                        it = octave_pairings.erase(it);
                    else
                        ++it;
                    ++counter;
                } // end-while
            } // end-if

            nMatchesPerOctave[octave] = octave_pairings.size();
            nMatches += octave_pairings.size();

            cout << "post: " << nMatches << endl;
        } // end for each octave
        // mrpt::system::os::fclose(fmatch);

        // save matches for octave 0
        /** /
        {
            size_t counter = 0;
            FILE *f121 = mrpt::system::os::fopen("1to1matches.txt","wt");
            for( vector_index_pairs_t::iterator it = imgpair.lr_pairing_data[0].matches_lr.begin();
                 it != imgpair.lr_pairing_data[0].matches_lr.end(); ++it )
            {
                const TSimpleFeature &featL = imgpair.left.pyr_feats[0][it->first];
                const TSimpleFeature &featR = imgpair.right.pyr_feats[0][it->second];
                mrpt::system::os::fprintf( f121, "%d %d %d %d %d %d\n",
                       it->first, featL.pt.x, featL.pt.y,
                       it->second, featR.pt.x, featR.pt.y );
            }
            mrpt::system::os::fclose(f121);
        }
        /**/

		// Filter for robust pairings ----------------------------------------
		// MRPT_TODO("Filter out spurious with a Homography model or such")

		// Build the row-sorted index of pairings ------------------------------
		m_profiler.enter("stg3.pairings-row-index");
		for (size_t octave=0;octave<nOctaves;octave++)
		{
			const vector_index_pairs_t & octave_pairings     = imgpair.lr_pairing_data[octave].matches_lr;
			vector<size_t>             & octave_pairings_idx = imgpair.lr_pairing_data[octave].matches_lr_row_index;
			const mrpt::vision::TSimpleFeatureList & octave_feats_left = imgpair.left.pyr_feats[octave];

			size_t idx=0;
			const size_t imgH   = imgpair.left.pyr.images[octave].getHeight();
			const size_t nFeats = octave_pairings.size();

			octave_pairings_idx.resize(imgH+1);  // the last entry, [nRows] = total # of feats
			for (size_t y=0;y<imgH;y++)
			{
				octave_pairings_idx[y] = idx;
				// Move on "idx" until we reach (at least) the next row:
				while (idx<nFeats && octave_feats_left[octave_pairings[idx].first].pt.y<=int(y)) { idx++; }
			}
			octave_pairings_idx[imgH]=octave_feats_left.size();

		}
		m_profiler.leave("stg3.pairings-row-index");

    } // end method faster+sad

	m_profiler.leave("stg3.find_pairings");


	// Draw pairings -----------------------------------------------------
	if (params_gui.draw_lr_pairings)
	{
	    m_profiler.enter("stg3.send2gui");
	    // FASTER
	    if( params_detect.detect_method == TDetectParams::dmFASTER )
	    {
            m_next_gui_info->draw_pairings_all.clear(); // "soft" clear (without dealloc)
            for (size_t octave=0;octave<nOctaves;octave++)
            {
                const TSimpleFeatureList &fL = imgpair.left.pyr_feats[octave];
                const TSimpleFeatureList &fR = imgpair.right.pyr_feats[octave];
                const vector_index_pairs_t & octave_pairings = imgpair.lr_pairing_data[octave].matches_lr;
                const size_t nFeats = octave_pairings.size();

                const size_t i0 = m_next_gui_info->draw_pairings_all.size();
                m_next_gui_info->draw_pairings_all.resize(i0 + nFeats);
                for(size_t i=0;i<nFeats;++i)
                {
                    // (X,Y) coordinates in the left/right images for this pairing:
                    m_next_gui_info->draw_pairings_all[i0+i].first = fL[ octave_pairings[i].first ].pt;
                    m_next_gui_info->draw_pairings_all[i0+i].second = fR[ octave_pairings[i].second].pt;
                }
            }
	    } // end-if
	    // ORB
	    else
	    {
	        m_next_gui_info->draw_pairings_all.clear();     // "soft" clear (without dealloc)
	        m_next_gui_info->draw_pairings_ids.clear();     // "soft" clear (without dealloc)
	        m_next_gui_info->draw_pairings_all.resize(imgpair.orb_matches.size());
	        m_next_gui_info->draw_pairings_ids.resize(imgpair.orb_matches.size());

	        vector<cv::DMatch>::iterator it;
	        size_t i = 0;
	        for( it = imgpair.orb_matches.begin(); it != imgpair.orb_matches.end(); ++it, ++i )
	        {
	            m_next_gui_info->draw_pairings_all[i].first.x   = imgpair.left.orb_feats[ it->queryIdx ].pt.x;
	            m_next_gui_info->draw_pairings_all[i].first.y   = imgpair.left.orb_feats[ it->queryIdx ].pt.y;
	            m_next_gui_info->draw_pairings_all[i].second.x  = imgpair.right.orb_feats[ it->trainIdx ].pt.x;
	            m_next_gui_info->draw_pairings_all[i].second.y  = imgpair.right.orb_feats[ it->trainIdx ].pt.y;

                m_next_gui_info->draw_pairings_ids[i] = imgpair.left.orb_feats[ it->queryIdx ].class_id;
	        } // end-for
	    } // end-else
        m_profiler.leave("stg3.send2gui");
	}

	m_profiler.leave("_stg3");

    if( params_detect.detect_method == TDetectParams::dmORB || params_detect.detect_method == TDetectParams::dmFAST_ORB )
        m_next_gui_info->text_msg_from_lr_match = mrpt::format(
                "L/R match: %lu ORB matchings ", imgpair.orb_matches.size() );
    else
    {
    	m_next_gui_info->text_msg_from_lr_match = mrpt::format(
		"L/R match: %u potential pairings | %u accepted (octaves: ",
		static_cast<unsigned int>(nPotentialMatches),
		static_cast<unsigned int>(nMatches)
		);

        for (size_t i=0;i<nOctaves;i++)
            m_next_gui_info->text_msg_from_lr_match+=mrpt::format("%u,",static_cast<unsigned int>(nMatchesPerOctave[i]));
        m_next_gui_info->text_msg_from_lr_match+=string(")");
	}

}


