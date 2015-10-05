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
using namespace mrpt::utils;
using namespace mrpt::vision;
using namespace mrpt::system;

void CStereoOdometryEstimator::stage4_track(
		CStereoOdometryEstimator::TTrackingData  & out_tracked_feats,
		CStereoOdometryEstimator::TImagePairData & prev_imgpair,		
		CStereoOdometryEstimator::TImagePairData & cur_imgpair )					// not const because the ids of the matches will change!
{
	m_profiler.enter("_stg4");
	const size_t nOctaves = prev_imgpair.left.pyr_feats.size();
	size_t nTracked = 0; // number of tracked features    

	// --------------------------------------------------------
	// ORB METHOD
	// --------------------------------------------------------
	if( params_detect.detect_method == TDetectParams::dmORB || params_detect.detect_method == TDetectParams::dmFAST_ORB )
	{
		/** /
		CImage imend( prev_imgpair.left.pyr.images[0].getWidth(), prev_imgpair.left.pyr.images[0].getHeight()+cur_imgpair.left.pyr.images[0].getHeight() );
		imend.drawImage(0,0,prev_imgpair.left.pyr.images[0]);
		imend.drawImage(0,prev_imgpair.left.pyr.images[0].getHeight()-1,cur_imgpair.left.pyr.images[0]);
		/**/

		// -- auxiliar variables
	    // vector<size_t> preLKpsIdx,preRKpsIdx,curLKpsIdx,curRKpsIdx;
	    const size_t preNMatches = prev_imgpair.orb_matches.size();

        // -- previous frame
		cv::Mat preLDesc(preNMatches,32,prev_imgpair.left.orb_desc.type());
	    cv::Mat preRDesc(preNMatches,32,prev_imgpair.right.orb_desc.type());

		for(size_t k = 0; k < preNMatches; ++k)
	    {
            // create matrixes with the proper descriptors
            // preLDesc and preRDesc
            prev_imgpair.left.orb_desc.row( prev_imgpair.orb_matches[k].queryIdx ).copyTo( preLDesc.row(k) );
	        prev_imgpair.right.orb_desc.row( prev_imgpair.orb_matches[k].trainIdx ).copyTo( preRDesc.row(k) );

	        // save the IDs of the involved feats
	        // preLKpsIdx.push_back( id1 );
	        // preRKpsIdx.push_back( id2 );

			// imend.drawCircle( prev_imgpair.left.orb_feats[id1].pt.x, prev_imgpair.left.orb_feats[id1].pt.y, mrpt::utils::TColor::red );
	    }

		// -- current frame
		const size_t curNMatches = cur_imgpair.orb_matches.size();

	    cv::Mat curLDesc(curNMatches,32,cur_imgpair.left.orb_desc.type());
	    cv::Mat curRDesc(curNMatches,32,cur_imgpair.right.orb_desc.type());
		
		for(size_t k = 0; k < curNMatches; ++k)
	    {
            // create matrixes with the proper descriptors
            // curLDesc and curRDesc
            cur_imgpair.left.orb_desc.row( cur_imgpair.orb_matches[k].queryIdx ).copyTo( curLDesc.row(k) );
	        cur_imgpair.right.orb_desc.row( cur_imgpair.orb_matches[k].trainIdx ).copyTo( curRDesc.row(k) );

	        // save the IDs of the involved feats
	        // curLKpsIdx.push_back( id1 );
	        // curRKpsIdx.push_back( id2 );

			// imend.drawCircle( cur_imgpair.left.orb_feats[id1].pt.x, cur_imgpair.left.orb_feats[id1].pt.y+prev_imgpair.left.pyr.images[0].getHeight(), mrpt::utils::TColor::red );
	    }

		// -- match the features
	    cv::BFMatcher matcher(cv::NORM_HAMMING,false);

        // -- match the left-left features and the right-right features
        vector<cv::DMatch> matL, matR;
        matcher.match( preLDesc /*query*/, curLDesc /*train*/, matL /* size of query */);
		matcher.match( preRDesc /*query*/, curRDesc /*train*/, matR /* size of query */);

        /** /
		for(size_t k = 0; k < matL.size(); ++k )
		{
			imend.line( prev_imgpair.left.orb_feats[matL[k].queryIdx].pt.x, prev_imgpair.left.orb_feats[matL[k].queryIdx].pt.y,
				cur_imgpair.left.orb_feats[matL[k].trainIdx].pt.x, 
				cur_imgpair.left.orb_feats[matL[k].trainIdx].pt.y+prev_imgpair.left.pyr.images[0].getHeight(),mrpt::utils::TColor::red );
		}

		imend.saveToFile("left-left.jpg");
        /**/

		// -- filter out by distance and avoid collisions for both 'Mats' at the same time
		vector<bool> left_train_matched( curNMatches, false ), right_train_matched( curNMatches, false );
		vector<cv::DMatch>::iterator itL = matL.begin(), itR = matR.begin();
		while( itL != matL.end() )
		{
			if( itL->distance > m_current_orb_th || itR->distance > m_current_orb_th || 
				left_train_matched[itL->trainIdx] || right_train_matched[itR->trainIdx] )	
			{ 
				itL = matL.erase( itL ); 
				itR = matR.erase( itR ); 
			}
			else
			{ 
				left_train_matched[itL->trainIdx] = right_train_matched[itR->trainIdx] = true;
				++itL; ++itR;
			}
		} // end

		ASSERTDEB_( matL.size() == matR.size() )
		
		// -- filter by fundmatrix LEFT MATCHES
        vector<cv::DMatch>::iterator itM;

        cv::Mat p1(matL.size(),2,cv::DataType<float>::type),p2(matL.size(),2,cv::DataType<float>::type);
        
		FILE *fprev = NULL;
		if( this->params_general.vo_save_files )
		{
			fprev = os::fopen( mrpt::format("%s/l-l_%04d.txt",params_general.vo_out_dir.c_str(),m_it_counter).c_str(), "wt");
			ASSERTDEB_( fprev!=NULL )
		}

		unsigned int k;
        for(k = 0, itM = matL.begin(); itM != matL.end(); ++itM, ++k)
        {
			const size_t preIdx = prev_imgpair.orb_matches[itM->queryIdx].queryIdx;
			const size_t curIdx = cur_imgpair.orb_matches[itM->trainIdx].queryIdx;

            p1.at<float>(k,0) = static_cast<float>( prev_imgpair.left.orb_feats[preIdx].pt.x );
            p1.at<float>(k,1) = static_cast<float>( prev_imgpair.left.orb_feats[preIdx].pt.y );
            p2.at<float>(k,0) = static_cast<float>( cur_imgpair.left.orb_feats[curIdx].pt.x );
            p2.at<float>(k,1) = static_cast<float>( cur_imgpair.left.orb_feats[curIdx].pt.y );

			if( this->params_general.vo_save_files )
			{
				os::fprintf(fprev, "%.2f %.2f %.2f %.2f %.2f\n",
					prev_imgpair.left.orb_feats[preIdx].pt.x, prev_imgpair.left.orb_feats[preIdx].pt.y,
					cur_imgpair.left.orb_feats[curIdx].pt.x, cur_imgpair.left.orb_feats[curIdx].pt.y, itM->distance );
			}
        } // end-for

		if( fprev ) os::fclose(fprev);
        
		vector<uchar> inliersLeft;
        cv::findFundamentalMat(p1,p2,cv::FM_RANSAC,1.0,0.99,inliersLeft);
		
		const int numInliersLeft = cv::countNonZero(inliersLeft);
		const bool goodFL = numInliersLeft >= 8;
		VERBOSE_LEVEL(2) << endl << "	Number of inliers left-left: " << numInliersLeft << endl;

        // -- filter by fundmatrix RIGHT MATCHES
		FILE *fcur = NULL;
		if( this->params_general.vo_save_files )
		{
			fcur = os::fopen(mrpt::format("%s/r-r_%04d.txt",params_general.vo_out_dir.c_str(),m_it_counter).c_str(),"wt");
			ASSERTDEB_( fcur!=NULL )
		}
	
        for(k = 0, itM = matR.begin(); itM != matR.end(); ++itM, ++k)
        {
			const size_t preIdx = prev_imgpair.orb_matches[itM->queryIdx].trainIdx;
			const size_t curIdx = cur_imgpair.orb_matches[itM->trainIdx].trainIdx;

            p1.at<float>(k,0) = static_cast<float>(prev_imgpair.right.orb_feats[preIdx].pt.x);
            p1.at<float>(k,1) = static_cast<float>(prev_imgpair.right.orb_feats[preIdx].pt.y);
            p2.at<float>(k,0) = static_cast<float>(cur_imgpair.right.orb_feats[curIdx].pt.x);
            p2.at<float>(k,1) = static_cast<float>(cur_imgpair.right.orb_feats[curIdx].pt.y);

			if( this->params_general.vo_save_files )
			{
				os::fprintf(fcur, "%.2f %.2f %.2f %.2f %.2f\n",
					prev_imgpair.right.orb_feats[preIdx].pt.x, prev_imgpair.right.orb_feats[preIdx].pt.y,
					cur_imgpair.right.orb_feats[curIdx].pt.x, cur_imgpair.right.orb_feats[curIdx].pt.y, itM->distance );
			}
        }

		if( fcur ) os::fclose(fcur);

		vector<uchar> inliersRight;
        cv::findFundamentalMat(p1,p2,cv::FM_RANSAC,1.0,0.99,inliersRight);
		
		const int numInliersRight = cv::countNonZero(inliersRight);
		const bool goodFR = numInliersRight >= 8;
		VERBOSE_LEVEL(2) << "	Number of inliers right-right: " << numInliersRight << endl;

		if( !goodFL || !goodFR )
		{
			VERBOSE_LEVEL(1) << " Fundamental matrix not found! left(" << goodFL <<") and right(" << goodFR << ")" << endl;
		}

        // -- delete outliers
        itL = matL.begin(); 
		itR = matR.begin();
        k = 0;
		while( itL != matL.end() && itR != matR.end() )
        {
            if( (goodFL && inliersLeft[k] == 0) || (goodFR && inliersRight[k] == 0) ) { itL = matL.erase(itL); itR = matR.erase(itR); }
            else { ++itL; ++itR; }
            ++k;
        }

        // -- save the tracking in the TTrackingData
        out_tracked_feats.prev_imgpair  = & prev_imgpair;
        out_tracked_feats.cur_imgpair   = & cur_imgpair;
        out_tracked_feats.tracked_pairs.resize( nOctaves );
        out_tracked_feats.tracked_pairs[0].reserve( matL.size() );

		// tracked_pairs[i].first  := idx of the match in the previous feature set
		// tracked_pairs[i].second := idx of the match in the current feature set

        // -- save the tracked features
        // const bool reset = _reset || /*m_tracked_pairs.size() == 0*/ m_usedIds.size() == 0; // start a new set of IDs

        // prepare ID vector
        //if( reset )
        //{
        //    m_usedIds.clear();
//            m_tracked_pairs.clear();
            // cout << "RESET TRACKED IDS!" << endl;
        //}

		// -- reset counter of tracked matches from last KF
		/*
		std::fill( this->m_idx_tracked_pairs_from_last_kf.begin(), this->m_idx_tracked_pairs_from_last_kf.end(), false );
		this->m_num_tracked_pairs_from_last_kf = 0;*/

		/*if( this->params_general.vo_use_matches_ids )
			std::fill( this->m_kf_ids.begin(), this->m_kf_ids.end(), false );*/

		if( this->params_general.vo_use_matches_ids )
		{
			m_kf_ids.clear();
			m_kf_ids.reserve( matL.size() );
			cur_imgpair.orb_matches_ID.resize( curNMatches );
		}

		vector<bool> c_tracked(curNMatches,false);

        // -- for each of the matched pairings...
		for( size_t k = 0; k < matL.size(); ++k )
        {
            // get the ID from the left feature in the previous set of features
            // const size_t featIdxP = preLKpsIdx[matL[k].queryIdx];
            // const int thisID = prev_imgpair.left.orb_feats[featIdxP].class_id;

            // is it already in our list of IDs?
            //vector<size_t>::iterator it = std::find(m_usedIds.begin(),m_usedIds.end(),thisID);

            // -- if we have a prev-cur match both left and right ids must be the same [CONSISTENCY CHECK]
			const size_t pre_match_idx = matL[k].queryIdx;		// idx of the MATCH (not the feature) in the previous frame
			const size_t cur_match_idx = matL[k].trainIdx;		// idx of the MATCH (not the feature) in the current frame

			if( matL[k].queryIdx == matR[k].queryIdx && matL[k].trainIdx == matR[k].trainIdx )
            {
                // we've got a tracked feature (only octave 0)
                out_tracked_feats.tracked_pairs[0].push_back( make_pair( pre_match_idx, cur_match_idx ) );	// store the match indexes

				// manage ids
				if( this->params_general.vo_use_matches_ids )
				{
					const size_t pre_match_ID = prev_imgpair.orb_matches_ID[ pre_match_idx ];
					
					if( pre_match_ID <= this->m_kf_max_match_ID )
						m_kf_ids.push_back( pre_match_ID );							// update

					/** /
					vector<size_t>::iterator it = std::find( this->m_ID_matches_kf.begin(), this->m_ID_matches_kf.end(), pre_match_ID );

					// fill a new map from match id to tracked status
					if( this->m_map_id2track.find( pre_match_idx ) != this->m_map_id2track.end() )
						n_map_id2track[pre_match_idx] = true;

					if( it != this->m_ID_matches_kf.end() )
					{
						const size_t idx = it-this->m_ID_matches_kf.begin();
						this->m_idx_tracked_pairs_from_last_kf[ idx ] = true;
						this->m_num_tracked_pairs_from_last_kf++;
					}
					/**/

					/** /
					if( pre_match_ID <= this->m_kf_max_match_ID )
					{
						this->m_idx_tracked_pairs_from_last_kf[ pre_match_ID ] = true;
						this->m_num_tracked_pairs_from_last_kf++;
					}
					/**/

					// set the ID of the tracked feats (keep the ID of the previous match)
					//const size_t featIdxC = curLKpsIdx[matL[k].trainIdx];
					//cur_imgpair.left.orb_feats[featIdxC].class_id = thisID;
					cur_imgpair.orb_matches_ID[cur_match_idx] = pre_match_ID;
					c_tracked[cur_match_idx] = true;
				}
				//nTrackedLastKF++;
                
//				if( reset )
//                {
//                    m_usedIds.push_back(thisID);        // first time here: add the IDs
//                    // m_tracked_pairs[thisID] = 0;
//                }
//                else
//                {
//                    if( it != /*m_tracked_pairs.end()*/m_usedIds.end() )
//                    {
////                        cout << thisID << ","
////                             << prev_imgpair.left.orb_feats[featIdxP].pt.x << ","
////                             << prev_imgpair.left.orb_feats[featIdxP].pt.y << ","
////                             << cur_imgpair.left.orb_feats[featIdxC].pt.x << ","
////                             << cur_imgpair.left.orb_feats[featIdxC].pt.y << endl;
//
////                        m_tracked_pairs[thisID] = 0;
//                        nTrackedLastKF++;
//                    }
//                } // end-else
            } // end-if
            //else
            //{
				//if( params_general.vo_use_matches_ids )
					//cur_imgpair.orb_matches_ID[ cur_match_idx ] = this->m_last_match_ID++;	// new ID

				// if( it != /*m_tracked_pairs.end()*/ m_usedIds.end() )
//                {
//                     // m_usedIds.erase(it);
////                    if( m_tracked_pairs[thisID] > 1 )
////                        m_tracked_pairs.erase(it2);      // lost feat (not seen more than the limit!)
////                    else
////                        m_tracked_pairs[thisID]++;       // give it another chance
////                    cout << "BORRAR" << endl;
////                    mrpt::system::pause();
//                }
            //}
        } // end for

		// add new ids to those current matches with no tracking info:
		if( this->params_general.vo_use_matches_ids )
		{
			for( size_t k = 0; k < c_tracked.size(); ++k )
			{
				if( !c_tracked[k] )
					cur_imgpair.orb_matches_ID[k] = ++this->m_last_match_ID;
			} // end-k
		} // end-if

		this->m_num_tracked_pairs_from_last_frame = out_tracked_feats.tracked_pairs[0].size();
		this->m_num_tracked_pairs_from_last_kf = this->m_kf_ids.size();

		// store the tracked features
		/** /
		this->m_map_id2track.swap( n_map_id2track );
		this->m_counter_tracked_last_keyframe = this->m_map_id2track.size();
		/**/

        // nTracked = out_tracked_feats.tracked_pairs[0].size();
        // this->m_tracked_pairs_from_last_reset = reset ? /*m_tracked_pairs.size()*/ m_usedIds.size() : nTrackedLastKF;
        // this->m_tracked_pairs_from_last_kf = num_tracked_last_kf;

		// save inter-frame matching to file if asked
		if( this->params_general.vo_save_files )
		{
			FILE *f = os::fopen( mrpt::format("%s/inter_frame_matching_%04d.txt",params_general.vo_out_dir.c_str(),m_it_counter).c_str(),"wt");
			for( size_t k = 0; k < out_tracked_feats.tracked_pairs[0].size(); ++k )
			{
				const size_t & preLIdx = prev_imgpair.orb_matches[out_tracked_feats.tracked_pairs[0][k].first].queryIdx;
				const size_t & preRIdx = prev_imgpair.orb_matches[out_tracked_feats.tracked_pairs[0][k].first].trainIdx;
				const size_t & curLIdx = cur_imgpair.orb_matches[out_tracked_feats.tracked_pairs[0][k].second].queryIdx;
				const size_t & curRIdx = cur_imgpair.orb_matches[out_tracked_feats.tracked_pairs[0][k].second].trainIdx;

				// os::fprintf(f,"%d %d\n", out_tracked_feats.tracked_pairs[0][k].first, out_tracked_feats.tracked_pairs[0][k].second );
				os::fprintf( f, "%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n",
					prev_imgpair.left.orb_feats[preLIdx].pt.x, prev_imgpair.left.orb_feats[preLIdx].pt.y, prev_imgpair.right.orb_feats[preRIdx].pt.x, prev_imgpair.right.orb_feats[preRIdx].pt.y,
					cur_imgpair.left.orb_feats[curLIdx].pt.x,  cur_imgpair.left.orb_feats[curLIdx].pt.y,  cur_imgpair.right.orb_feats[curRIdx].pt.x,  cur_imgpair.right.orb_feats[curRIdx].pt.y );
			}
			os::fclose(f);
		}

	} // end method ORB
	else
	{
        // Tracking strategy #1:
        //  Only try to match nearby FAST features via minimal SAD:
#if 0
		// UNMAINTAINED: CONSIDER REMOVAL
        // Tracking window size in pixels (from [-W,+W])
        const int WIN_W = 16;
        const int WIN_H = 16;

        const int PATCHSIZE_L = 3; // 8x8 patches are [-3,4] wrt the center point
        const int PATCHSIZE_R = 4;

        const uint32_t MAX_SAD = 1000;

        const uint32_t invalid_pairing_id = std::string::npos;

        out_tracked_feats.prev_imgpair = &prev_imgpair;
        out_tracked_feats.cur_imgpair = &cur_imgpair;
        out_tracked_feats.tracked_pairs.assign(nOctaves, vector_index_pairs_t() );

        // FILE *ftrack = os::fopen("ftracks.txt", "wt");
        m_profiler.enter("stg4.track");
        for (size_t octave=0;octave<nOctaves;octave++)
        {
    //	    cout << "octave: " << octave << endl;

            // For each octave:
            const TImagePairData::img_pairing_data_t &prev_pairs = prev_imgpair.lr_pairing_data[octave];
            const TImagePairData::img_pairing_data_t &cur_pairs  = cur_imgpair.lr_pairing_data[octave];

            // Get the maximum (x,y) for window searching:
            const mrpt::utils::TImageSize img_size = prev_imgpair.left.pyr.images[octave].getSize();
            const int absolute_wx_max = img_size.x - 1 - PATCHSIZE_R;
            const int absolute_wy_max = img_size.y - 1 - PATCHSIZE_R;

            // get references to the L&R prev/cur images of this octave:
            const CImage &pImgL = prev_imgpair.left.pyr.images[octave];
            const CImage &pImgR = prev_imgpair.right.pyr.images[octave];
            const CImage &cImgL = cur_imgpair.left.pyr.images[octave];
            const CImage &cImgR = cur_imgpair.right.pyr.images[octave];

            // Get pointers to the image data:
            const unsigned char *prev_img_data_L = pImgL.get_unsafe(0,0);
            const unsigned char *prev_img_data_R = pImgR.get_unsafe(0,0);

            const unsigned char *cur_img_data_L = cImgL.get_unsafe(0,0);
            const unsigned char *cur_img_data_R = cImgR.get_unsafe(0,0);

            const size_t img_stride = pImgL.getRowStride();
            ASSERTDEB_(img_stride==pImgR.getRowStride() && img_stride==cImgL.getRowStride() && img_stride==cImgR.getRowStride())

            // Go thru all the paired features in the y'th row of the previous image,
            //  and compare them with all the paired features in a window on the current image.
            for (int y=0;y<img_size.y-1;y++)
            {
                // Make a list with all the paired features in the previous image in this row:
                const size_t prev_idx0 = prev_pairs.matches_lr_row_index[y];
                const size_t prev_idx1 = prev_pairs.matches_lr_row_index[y+1];  // the last entry, [nRows] = total # of feats
                const size_t prev_num_feats = prev_idx1-prev_idx0;

                if (!prev_num_feats)
                    continue; // There're NO paired features in this row, go on!

                // Set the vertical limits of the search window (common to all features in this line)
                const int wy_min = std::max(PATCHSIZE_L     , y-WIN_W);
                const int wy_max = std::min(absolute_wy_max , y+WIN_W);

                // Check all the candidate features in "cur_imgpair" within the vertical window:
                const size_t cur_idx0 = cur_pairs.matches_lr_row_index[wy_min];
                const size_t cur_idx1 = cur_pairs.matches_lr_row_index[wy_max+1];  // the last entry, [nRows] = total # of feats
                const size_t cur_num_feats = cur_idx1-cur_idx0;

                if (!cur_num_feats)
                    continue; // There're NO paired features in this row, go on!

                // Check each feature in this row:
                for (size_t pi=prev_idx0;pi<prev_idx1;pi++)
                {
                    // The two indices of the left & right features in the previous stereo img:
                    const size_t pidx_l = prev_pairs.matches_lr[pi].first;
                    const size_t pidx_r = prev_pairs.matches_lr[pi].second;

                    size_t   best_pairing_in_curimg = invalid_pairing_id; // Index of paired feature ( as in cur_pairs.matches_lr )
                    uint32_t best_pairing_SAD=std::numeric_limits<uint32_t>::max(); // Sum of the SAD of both L&R tracked feats

                    // Set the horz limits of the search window for these feats:
                    const mrpt::vision::TSimpleFeature &p_ft_l = prev_imgpair.left.pyr_feats[octave][pidx_l];
                    const mrpt::vision::TSimpleFeature &p_ft_r = prev_imgpair.right.pyr_feats[octave][pidx_r];

                    // Each Left/right feature has its own horz window:
                    const int wx_min_l = std::max(PATCHSIZE_L     , p_ft_l.pt.x-WIN_H );
                    const int wx_max_l = std::min(absolute_wx_max , p_ft_l.pt.x+WIN_H );
                    const int wx_min_r = std::max(PATCHSIZE_L     , p_ft_r.pt.x-WIN_H );
                    const int wx_max_r = std::min(absolute_wx_max , p_ft_r.pt.x+WIN_H );

                    for (size_t ci=cur_idx0;ci<cur_idx1;ci++)
                    {
                        // The two indices of the left & right features in the current stereo img:
                        const size_t cidx_l = cur_pairs.matches_lr[ci].first;
                        const size_t cidx_r = cur_pairs.matches_lr[ci].second;

                        const mrpt::vision::TSimpleFeature &ft_l = cur_imgpair.left.pyr_feats[octave][cidx_l];
                        const mrpt::vision::TSimpleFeature &ft_r = cur_imgpair.right.pyr_feats[octave][cidx_r];

                        // Check if it falls within the horz window:
                        if (ft_l.pt.x<wx_min_l || ft_l.pt.x>wx_max_l ||
                            ft_r.pt.x<wx_min_r || ft_r.pt.x>wx_max_r )
                            continue; // Feature is out of the window, skip!

                        const uint32_t sad_l = rso::compute_SAD8(prev_img_data_L,cur_img_data_L,img_stride,p_ft_l.pt,ft_l.pt);
                        if (sad_l>MAX_SAD)
                            continue; // Bad match on left img, skip

                        const uint32_t sad_r = rso::compute_SAD8(prev_img_data_R,cur_img_data_R,img_stride,p_ft_r.pt,ft_r.pt);
                        if (sad_r>MAX_SAD)
                            continue; // Bad match on right img, skip

                        const uint32_t sad = sad_l+sad_r;

                        // OK, match below the threshold: keep the best of all of them:
                        if (sad<best_pairing_SAD)
                        {
                            best_pairing_SAD = sad;
                            best_pairing_in_curimg = ci;
                        }
                    } // end for each feature in current imgpair at row "y"

                    if (best_pairing_in_curimg!=invalid_pairing_id)
                    {
                        // we've got a match!
    //					cout << "track: " << pi << "<->" << best_pairing_in_curimg << " minSAD:" << best_pairing_SAD << endl;
                        out_tracked_feats.tracked_pairs[octave].push_back( std::make_pair(pi,best_pairing_in_curimg) );

                        const size_t idL1 = prev_pairs.matches_lr[pi].first;
                        const size_t idR1 = prev_pairs.matches_lr[pi].second;
                        const size_t idL2 = cur_pairs.matches_lr[best_pairing_in_curimg].first;
                        const size_t idR2 = cur_pairs.matches_lr[best_pairing_in_curimg].second;
                        os::fprintf(ftrack,"%d %d %d %d %d %d %d %d\n",
                            prev_imgpair.left.pyr_feats[octave][idL1].pt.x,  prev_imgpair.left.pyr_feats[octave][idL1].pt.y,
                            prev_imgpair.right.pyr_feats[octave][idR1].pt.x, prev_imgpair.right.pyr_feats[octave][idR1].pt.y,
                            cur_imgpair.left.pyr_feats[octave][idL2].pt.x,   cur_imgpair.left.pyr_feats[octave][idL2].pt.y,
                            cur_imgpair.right.pyr_feats[octave][idR2].pt.x,  cur_imgpair.right.pyr_feats[octave][idR2].pt.y);
                        //pause();

                        // update the identificator (keep the previous one)
                        cur_imgpair.left.pyr_feats[octave][idL2].ID =
                            prev_imgpair.left.pyr_feats[octave][idL1].ID;

                        cur_imgpair.right.pyr_feats[octave][idR2].ID =
                            prev_imgpair.right.pyr_feats[octave][idR1].ID;
                    }

                } // end for each feature in previous imgpair at row "y"

            } // end for each row "y"

            nTracked+=out_tracked_feats.tracked_pairs[octave].size();
        } // end for each octave
        cout << "Tracked features: " << nTracked << endl;
        //os::fclose(ftrack);
        m_profiler.leave("stg4.track");

    #endif // Strategy #1
	// Tracking strategy #2:
	// Use SAD matching in a window (???)

	}// end-method FASTER

    // Draw pairings -----------------------------------------------------
    if (params_gui.draw_tracking)
    {
        m_profiler.enter("stg4.send2gui");
        // FASTER
        if( params_detect.detect_method == TDetectParams::dmFASTER )
        {
            m_next_gui_info->stats_tracked_feats.clear();
            m_next_gui_info->stats_tracked_feats.reserve(nTracked);
            for (size_t octave=0;octave<nOctaves;octave++)
            {
                const TSimpleFeatureList &pfL = prev_imgpair.left.pyr_feats[octave];
                const TSimpleFeatureList &pfR = prev_imgpair.right.pyr_feats[octave];
                const TSimpleFeatureList &cfL = cur_imgpair.left.pyr_feats[octave];
                const TSimpleFeatureList &cfR = cur_imgpair.right.pyr_feats[octave];

                const vector_index_pairs_t & pPairings = prev_imgpair.lr_pairing_data[octave].matches_lr;
                const vector_index_pairs_t & cPairings = cur_imgpair.lr_pairing_data[octave].matches_lr;

                const vector_index_pairs_t &tracked_feats = out_tracked_feats.tracked_pairs[octave];
                const size_t nFeats = tracked_feats.size();
                for (size_t i=0;i<nFeats;i++)
                {
                    const size_t p_idx_l = pPairings[tracked_feats[i].first].first;
                    const size_t p_idx_r = pPairings[tracked_feats[i].first].second;
                    const size_t c_idx_l = cPairings[tracked_feats[i].second].first;
                    const size_t c_idx_r = cPairings[tracked_feats[i].second].second;

                    m_next_gui_info->stats_tracked_feats.resize( m_next_gui_info->stats_tracked_feats.size()+1 );
                    TTrackedPixels  & tp = *m_next_gui_info->stats_tracked_feats.rbegin();

                    tp.px_pL = pfL[p_idx_l].pt;
                    tp.px_pR = pfR[p_idx_r].pt;
                    tp.px_cL = cfL[c_idx_l].pt;
                    tp.px_cR = cfR[c_idx_r].pt;
                }
            }
        } // end if
        // ORB
        else
        {
            m_next_gui_info->stats_tracked_feats.clear();
			m_next_gui_info->stats_tracked_feats.reserve(this->m_num_tracked_pairs_from_last_frame);

            const vector<cv::KeyPoint> & pfL = prev_imgpair.left.orb_feats;
            const vector<cv::KeyPoint> & pfR = prev_imgpair.right.orb_feats;
            const vector<cv::KeyPoint> & cfL = cur_imgpair.left.orb_feats;
            const vector<cv::KeyPoint> & cfR = cur_imgpair.right.orb_feats;

            const vector<cv::DMatch>   & pPairings = prev_imgpair.orb_matches;
            const vector<cv::DMatch>   & cPairings = cur_imgpair.orb_matches;

            const vector_index_pairs_t & tracked_feats = out_tracked_feats.tracked_pairs[0];

            for( size_t i = 0; i < this->m_num_tracked_pairs_from_last_frame; ++i )
            {
                const size_t p_idx_l = pPairings[tracked_feats[i].first].queryIdx;
                const size_t p_idx_r = pPairings[tracked_feats[i].first].trainIdx;
                const size_t c_idx_l = cPairings[tracked_feats[i].second].queryIdx;
                const size_t c_idx_r = cPairings[tracked_feats[i].second].trainIdx;

                m_next_gui_info->stats_tracked_feats.resize( m_next_gui_info->stats_tracked_feats.size()+1 );
                TTrackedPixels  & tp = *m_next_gui_info->stats_tracked_feats.rbegin();

                tp.px_pL.x = pfL[p_idx_l].pt.x;     tp.px_pL.y = pfL[p_idx_l].pt.y;
                tp.px_pR.x = pfR[p_idx_r].pt.x;     tp.px_pR.y = pfR[p_idx_r].pt.y;
                tp.px_cL.x = cfL[c_idx_l].pt.x;     tp.px_cL.y = cfL[c_idx_l].pt.y;
                tp.px_cR.x = cfR[c_idx_r].pt.x;     tp.px_cR.y = cfR[c_idx_r].pt.y;
            } // end-for
        } // end-else
        m_profiler.leave("stg4.send2gui");
    }

	// infor for the GUI
    m_next_gui_info->text_msg_from_conseq_match = mrpt::format(
        "Tracked: %u features",
        static_cast<unsigned int>(this->m_num_tracked_pairs_from_last_frame)
    );
    // cout << m_next_gui_info->text_msg_from_conseq_match << endl;
	m_profiler.leave("_stg4");
}


