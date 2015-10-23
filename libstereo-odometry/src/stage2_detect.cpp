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
using namespace cv;

CStereoOdometryEstimator::TDetectParams::TDetectParams() :
	target_feats_per_pixel (10./1000.),
	initial_FAST_threshold(20/*6*/),
	fast_min_th(5), fast_max_th(30),
	non_maximal_suppression(true),
	KLT_win(4),
	minimum_KLT(10),
	detect_method(dmFASTER),
	nmsMethod(nmsmStandard),
	orb_nfeats(500),
	orb_nlevels(8),
	min_distance(3)
{
}

void CStereoOdometryEstimator::m_update_indexes( 
						TImagePairData::img_data_t	& data,					// I/O	-- Structure with features and feature index vector (which is filled here)
						size_t						  octave,				// I	-- Octave to process
						const vector<size_t>		& sorted_indices )		// oI	-- Vector containing the order of the features to be processed (ordered by row)
{
	bool use_sorted_indices = sorted_indices.size() > 0;
	if( use_sorted_indices )
	{
		ASSERT_(data.pyr_feats[octave].size() == sorted_indices.size() )
	}

	vector<size_t>::iterator fromRow = data.pyr_feats_index[octave].begin(), toRow;

    size_t feats_till_now = 0;
    size_t current_row = 0;
    for( size_t idx_feats = 0; idx_feats < data.pyr_feats[octave].size(); ++idx_feats)
    {
		const size_t idx = use_sorted_indices ? sorted_indices[idx_feats] : idx_feats;
        const TSimpleFeature &feat = data.pyr_feats[octave][idx];
        if( idx_feats == 0 )
        {
            current_row = feat.pt.y;
            toRow = data.pyr_feats_index[octave].begin()+current_row;
            fill( fromRow, toRow, 0 );  // fill with zeros
            fromRow = toRow;
            continue;
        }

        if( feat.pt.y == int(current_row) )
        {
            ++feats_till_now;
            continue;
        }
        current_row = feat.pt.y;

        toRow = data.pyr_feats_index[octave].begin()+current_row;
        fill( fromRow, toRow, ++feats_till_now );
        fromRow = toRow;
    }
}

void CStereoOdometryEstimator::m_adaptive_non_max_sup( 
					const size_t				& num_out_points, 
					const vector<KeyPoint>		& keypoints, 
					const Mat					& descriptors,
					vector<KeyPoint>			& out_kp_rad,
					Mat							& out_kp_desc,
					const double				& min_radius_th )
{
	// TO DO : if detection method != ORB or FAST+ORB --> throw exception
	// do it for both images

	// save before
	/** /
	{
		FILE *f = mrpt::system::os::fopen("antes.txt","wt");
		for( size_t k = 0; k < keypoints.size(); ++k )
			mrpt::system::os::fprintf(f,"%.2f %.2f\n",keypoints[k].pt.x,keypoints[k].pt.y);
		mrpt::system::os::fclose(f);
	}
	/**/

	const size_t actual_num_out_points = min(num_out_points,keypoints.size());
	
	const double CROB = 0.9;
	const size_t N = keypoints.size();

	// First: order them by response
	vector<size_t> sorted_indices( N );
	for (size_t i=0;i<N;i++)  sorted_indices[i]=i;
	std::sort( sorted_indices.begin(), sorted_indices.end(), KeypointResponseSorter< vector<KeyPoint> >(keypoints) );

	// insert the global maximum 
	const cv::KeyPoint & strongest_kp = keypoints[sorted_indices[0]];
	
	// create the radius vector	
	vector<double> radius( N );
	radius[sorted_indices[0]] = std::numeric_limits<double>::infinity();	// set the radius for this keypoint to infinity

	// for the rest of the keypoints:
	double min_ri, this_ri;
	for( size_t k1 = 1; k1 < N; ++k1 )
	{
		// cout << k1 << "->" << sorted_indices[k1] << ",";
		const cv::KeyPoint & kp1 = keypoints[sorted_indices[k1]];

		// the min_ri is at most the distance to the strongest keypoint
		min_ri = std::fabs( (kp1.pt.x-strongest_kp.pt.x)*(kp1.pt.x-strongest_kp.pt.x)+(kp1.pt.y-strongest_kp.pt.y)*(kp1.pt.y-strongest_kp.pt.y) );; // distance to the strongest
		
		// compute the ri value for all the previous keypoints
		for( int k2 = k1-1; k2 > 0; --k2 )
		{
			const cv::KeyPoint & kp2 = keypoints[sorted_indices[k2]];

			if( kp1.response < CROB*kp2.response )
			{
				this_ri = std::fabs((kp1.pt.x-kp2.pt.x)*(kp1.pt.x-kp2.pt.x)+(kp1.pt.y-kp2.pt.y)*(kp1.pt.y-kp2.pt.y));
				if( this_ri < min_ri ) min_ri = this_ri;
			}
		} // end-for-k2
		radius[sorted_indices[k1]] = min_ri;
	} // end-for-k1
	// cout << endl;

	// sort again according to the radius
	const double min_radius_th_2 = min_radius_th*min_radius_th;
	for( size_t i = 0; i < N; i++ ) sorted_indices[i] = i;
	std::sort( sorted_indices.begin(), sorted_indices.end(), KpRadiusSorter(radius) );
	
	// fill ouput
	// cout << "fill output " << endl;
	out_kp_rad.clear();
	out_kp_rad.reserve( N );
	// cout << "Radius: " << radius.size() << " and num_out_points: " << num_out_points << " actual: " << actual_num_out_points << endl;
	// cout << "Descriptor size: " << descriptors.size() << endl;
	// cout << "Keypoints size: " << keypoints.size() << endl;
	// cout << "N: " << N << endl;
	for( size_t i = 0; i < actual_num_out_points; i++ ) 
	{
		if( radius[sorted_indices[i]] > min_radius_th_2 ) 
		{
			out_kp_rad.push_back( keypoints[sorted_indices[i]] );
			out_kp_desc.push_back( descriptors.row( sorted_indices[i] ) );
		}
	}
	// cout << "done" << endl;
	// save after
	/** /
	{
		FILE *f = mrpt::system::os::fopen("despues.txt","wt");
		for( size_t k = 0; k < out_kp_rad.size(); ++k )
			mrpt::system::os::fprintf(f,"%.2f %.2f\n",out_kp_rad[k].pt.x,out_kp_rad[k].pt.y);
		mrpt::system::os::fclose(f);
	}
	/**/

} // end-m_adaptive_non_max_suppression

void CStereoOdometryEstimator::m_non_max_sup( TImagePairData::img_data_t &data, size_t octave )	// <-- useless?? consider to remove
{
    /** /
    {
        FILE *fb = mrpt::system::os::fopen("fbefore.txt","wt");
        for (size_t i=0;i<data.pyr_feats[octave].size();i++)
        {
            mrpt::system::os::fprintf(fb,"%d %d %.3f\n",
                                      data.pyr_feats[octave][i].pt.x,
                                      data.pyr_feats[octave][i].pt.y,
                                      data.pyr_feats[octave][i].response
                                      );
        }
        mrpt::system::os::fclose(fb);
    }
	/**/

    const mrpt::vector_size_t &idxL	= data.pyr_feats_index[octave];				// the index of the feat
    const size_t nColsMax			= data.pyr.images[octave].getWidth()-1;		// max number of columns
    const size_t nRowsMax			= data.pyr.images[octave].getHeight()-1;	// max number of rows

    vector<bool> featsToDelete(data.pyr_feats[octave].size(), false);
    for (size_t i=0;i<data.pyr_feats[octave].size();i++)
    {
        const TSimpleFeature &featL		= data.pyr_feats[octave][i];			// this feat
        if( featL.response == 0 ) // delete this (was too close to get the KLT value)
        {
            // cout << " [delete] (response == 0)" << endl;
            featsToDelete[i] = true;
            continue;
        }

        const size_t col_min			= std::max(0,featL.pt.x-2);						// from this col
        const size_t col_max			= std::min(int(nColsMax),featL.pt.x+2);			// ... to this col
        const size_t idx_feats_L0		= idxL[max(0,featL.pt.y-3)];					// from this row
        const size_t idx_feats_L1		= idxL[min(int(nRowsMax),featL.pt.y+2)];		// ... to this row

        if( featsToDelete[i] )
        {
            // cout << " [already set to be deleted] " << endl;
            continue;
        }

        // feats to test
		// cout << "--> From " << idx_feats_L0 << " to " << idx_feats_L1 << endl;

        // search for the maximum in a 5x5 window
        float max_response      = featL.response;
        //size_t max_response_idx = i;

        for (size_t idx_feats_L = idx_feats_L0; idx_feats_L < idx_feats_L1; ++idx_feats_L)
        {
            if(idx_feats_L == i)        // not check with itself
                continue;

		    // cout << "   with " << idx_feats_L;

            if( featsToDelete[idx_feats_L] )
            {
                // cout << " [already set to be deleted] " << endl;
                continue;
            }

            // this feature hasn't been visited yet
            TSimpleFeature &ofeatL = data.pyr_feats[octave][idx_feats_L];

            if( ofeatL.pt.x < int(col_min) || ofeatL.pt.x > int(col_max) )
            {
		        // cout << " [out of range] -- (" << ofeatL.pt.x << "," << ofeatL.pt.y << ")" << endl;
                continue;
            }

            // cout << " [res = " << ofeatL.response << " at " << ofeatL.pt.x << "," << ofeatL.pt.y << "]";

            if( ofeatL.response > max_response )
            {
                // cout << " -- max (" << ofeatL.response << " vs " << max_response << ")" << endl;
                max_response        = ofeatL.response;
                // max_response_idx    = idx_feats_L;
                featsToDelete[i]    = true;
            }
            else
            {
                featsToDelete[idx_feats_L] = true;
            }

        } // end for search maximun in window
    // mrpt::system::pause();
    } // end-for-feature

//         mrpt::gui::CDisplayWindow win_before, win_after;
//         win_before.showImageAndPoints(data.pyr.images[octave],data.pyr_feats[octave]);

    // remove bad features
    TSimpleFeatureList::iterator it = data.pyr_feats[octave].begin();
    size_t count = 0;
    while( it != data.pyr_feats[octave].end() )
    {
        if( featsToDelete[count] )
        {
            it = data.pyr_feats[octave].erase( it );
        }
        else
        {
            ++it;
        }
        count++;
    }

    /**/
    {
        FILE *fa = mrpt::system::os::fopen("fafter.txt","wt");
        for (size_t i=0;i<data.pyr_feats[octave].size();i++)
        {
            mrpt::system::os::fprintf(fa,"%d %d %.3f\n",
                                      data.pyr_feats[octave][i].pt.x,
                                      data.pyr_feats[octave][i].pt.y,
                                      data.pyr_feats[octave][i].response
                                      );
        }
        mrpt::system::os::fclose(fa);
    }
    /**/

//         win_after.showImageAndPoints(data.pyr.images[octave],data.pyr_feats[octave]);
//        mrpt::system::pause();

	// order by row coordinate
	const size_t N = data.pyr_feats[octave].size();
	vector<size_t> sorted_indices( N );
	for (size_t i=0;i<N;i++)  sorted_indices[i]=i;
	std::sort( sorted_indices.begin(), sorted_indices.end(), KpRowSorter(data.pyr_feats[octave]) );

    // update indexes
	m_update_indexes( data, octave, sorted_indices );

    /** /
    {
        FILE *fia = mrpt::system::os::fopen("fiafter.txt","wt");
        for (size_t i=0;i<data.pyr_feats_index[octave].size();i++)
            mrpt::system::os::fprintf(fia,"%lu\n", data.pyr_feats_index[octave][i]);
        mrpt::system::os::fclose(fia);
    }
    /**/

    //const vector_size_t & idxL = data.pyr_feats_index[octave];
    //const size_t nRowsMax = idxL.size()-1;
    //for (size_t y=0;y<nRowsMax;y++)
    //{
    //	size_t idx_feats_L0, idx_feats_L1;
    //	y == 0 ? idx_feats_L0 = idxL[y] : idx_feats_L0 = idxL[y-1];					// one less row
    //	y == nRowsMax-1 ? idx_feats_L1 = idxL[y] : idx_feats_L1 = idxL[y+1];		// one more row
    //
    //	for (size_t idx_feats_L=idx_feats_L0;idx_feats_L<idx_feats_L1;idx_feats_L++)
    //	{
    //		const TSimpleFeature &featL = data.pyr_feats[octave][idx_feats_L];
    //		if( featL.response )


    //	} // end idfeats

    //} // end for nRowsMax

} // end m_non_max_sup
void CStereoOdometryEstimator::m_non_max_sup(
					const size_t				& num_out_points, 
					const vector<KeyPoint>		& keypoints, 
					const Mat					& descriptors,
					vector<KeyPoint>			& out_kp,
					Mat							& out_kp_desc,
					const size_t				& imgH, 
					const size_t				& imgW )
{				
	//  1) Sort the features by "response": It's ~100 times faster to sort a list of
	//      indices "sorted_indices" than sorting directly the actual list of features "cv_feats"
	const size_t n_feats = keypoints.size();
			
	//prepare output
	out_kp.clear();
	out_kp.reserve( n_feats );
	out_kp_desc.reserve( n_feats );

	std::vector<size_t> sorted_indices(n_feats);
	for( size_t i = 0; i < n_feats; i++ )  sorted_indices[i]=i;
	std::sort( sorted_indices.begin(), sorted_indices.end(), KeypointResponseSorter< vector<KeyPoint> >(keypoints) );

	//  2) Filter by "min-distance" (in options.ORBOptions.min_distance)
	// The "min-distance" filter is done by means of a 2D binary matrix where each cell is marked when one
	// feature falls within it. This is not exactly the same than a pure "min-distance" but is pretty close
	// and for large numbers of features is much faster than brute force search of kd-trees.
	// (An intermediate approach would be the creation of a mask image updated for each accepted feature, etc.)
	const unsigned int occupied_grid_cell_size = params_detect.min_distance/2.0;
	const float occupied_grid_cell_size_inv = 1.0f/occupied_grid_cell_size;

	unsigned int grid_lx = (unsigned int)(1 + imgW * occupied_grid_cell_size_inv);
	unsigned int grid_ly = (unsigned int)(1 + imgH * occupied_grid_cell_size_inv );

	mrpt::math::CMatrixB occupied_sections(grid_lx,grid_ly);  // See the comments above for an explanation.
	occupied_sections.fillAll(false);

	size_t k = 0;
	size_t c_feats = 0;
	while( c_feats < num_out_points && k < n_feats )
	{
		const size_t idx = sorted_indices[k++];
		const KeyPoint & kp = keypoints[idx];

		// Check the min-distance:
		const size_t section_idx_x = size_t(kp.pt.x * occupied_grid_cell_size_inv);
		const size_t section_idx_y = size_t(kp.pt.y * occupied_grid_cell_size_inv);

		if (occupied_sections(section_idx_x,section_idx_y))
			continue; // Already occupied! skip.

		// Mark section as occupied
		occupied_sections.set_unsafe(section_idx_x,section_idx_y, true);
		if (section_idx_x>0)			occupied_sections.set_unsafe(section_idx_x-1,section_idx_y, true);
		if (section_idx_y>0)			occupied_sections.set_unsafe(section_idx_x,section_idx_y-1, true);
		if (section_idx_x<grid_lx-1)	occupied_sections.set_unsafe(section_idx_x+1,section_idx_y, true);
		if (section_idx_y<grid_ly-1)	occupied_sections.set_unsafe(section_idx_x,section_idx_y+1, true);

		// Add it to the output vector
		out_kp.push_back( kp );
		out_kp_desc.push_back( descriptors.row(idx) );
		++c_feats;
	} // end-while
}

/**  Stage2 operations:
  *   - Detect features on each image and on each scale.
  */
void CStereoOdometryEstimator::stage2_detect_features(
	CStereoOdometryEstimator::TImagePairData::img_data_t & img_data,
	mrpt::utils::CImage & gui_image,
	bool update_dyn_thresholds )
{
	using namespace mrpt::vision;

	m_profiler.enter("_stg2");

	// Resize containers:
	const size_t nPyrs = img_data.pyr.images.size();		// this will be 1 
	vector<size_t> nFeatsPassingKLTPerOctave(nPyrs);        // for stats only
    img_data.pyr_feats.resize(nPyrs);
    img_data.pyr_feats_index.resize(nPyrs);
    img_data.pyr_feats_desc.resize(nPyrs);

	if( params_detect.detect_method == TDetectParams::dmORB )
	{
        // ***********************************
	    // ORB method --> output feats are not ordered either by position nor by response
	    // ***********************************
        CImage & input_im = img_data.pyr.images[0];	// shortcut
		const size_t n_feats_to_extract = params_detect.non_maximal_suppression ? 1.5*params_detect.orb_nfeats : params_detect.orb_nfeats; // if non-max-sup is ON extract more features to get approx the number of desired output feats.
		
		// detect points
		// ORB orbDetector( n_feats_to_extract, 1.2, params_detect.orb_nlevels );	// user-defined number of feats, 'orb_nlevels' levels for pyramid, 1.2 distance between pyramid levels
		ORB orbDetector( n_feats_to_extract, 1.2, params_detect.orb_nlevels, 
			31 /*edgeThreshold*/, 
			0 /*firstLevel*/,
			2 /*WTA_K*/,
			ORB::HARRIS_SCORE /*scoreType*/,
			31 /*patchSize*/,
			m_current_fast_th /*params_detect.initial_FAST_threshold*/ );

        Mat img( input_im.getAs<IplImage>() );

        // detect keypoints and desc
		vector<KeyPoint> feat_vector;
		Mat				 desc_aux;
		orbDetector( img, Mat(), feat_vector, desc_aux );  // all the scales in the same call

		// non-maximal suppression
		if( params_detect.non_maximal_suppression )
		{
			if( params_detect.nmsMethod == TDetectParams::nmsmStandard )
			{
				const size_t imgH = input_im.getHeight();
				const size_t imgW = input_im.getWidth();
				this->m_non_max_sup( params_detect.orb_nfeats, feat_vector, desc_aux, img_data.orb_feats, img_data.orb_desc, imgH, imgW );
			}
			else if( params_detect.nmsMethod == TDetectParams::nmsmAdaptive )
				this->m_adaptive_non_max_sup( params_detect.orb_nfeats, feat_vector, desc_aux, img_data.orb_feats, img_data.orb_desc );
			else
				THROW_EXCEPTION("	[sVO -- Stg2: Detect] Invalid non-maximal-suppression method." );
		} // end-if-non-max-sup
		else
		{
			feat_vector.swap(img_data.orb_feats);
			img_data.orb_desc = desc_aux;					// this should copy just the header
		}

		FILE *f = mrpt::system::os::fopen("feats.txt","wt");
		for(int i = 0; i < img_data.orb_feats.size(); ++i) mrpt::system::os::fprintf(f,"%.2f,%.2f,%.5f\n",img_data.orb_feats[i].pt.x,img_data.orb_feats[i].pt.y,img_data.orb_feats[i].response);
		mrpt::system::os::fclose(f);

#if 0
		// perform subpixel
		{
			FILE *fo = mrpt::system::os::fopen("fbef.txt","wt");
			for( vector<KeyPoint>::iterator it = img_data.orb_feats.begin(); it != img_data.orb_feats.end(); ++it )
				mrpt::system::os::fprintf(fo,"%.2f %.2f\n", it->pt.x, it->pt.y );
			mrpt::system::os::fclose(fo);
		}

		CTimeLogger tictac;
		tictac.enter("kp->p");
		
		std::vector<Point2f> ofeats(img_data.orb_feats.size());
		for( int k = 0; k < img_data.orb_feats.size(); ++k )
			ofeats[k] = img_data.orb_feats[k].pt;

		tictac.leave("kp->p");

		tictac.enter("subpx");
		cv::cornerSubPix( img, ofeats/*img_data.orb_feats*/, Size(5,5), Size(1,1), TermCriteria( cv::TermCriteria::COUNT, 20, 1e-3 ) );
		tictac.leave("subpx");

		tictac.enter("p->kp");

		for( int k = 0; k < img_data.orb_feats.size(); ++k )
			img_data.orb_feats[k].pt = ofeats[k];
		
		tictac.leave("p->kp");

		{
			FILE *fo = mrpt::system::os::fopen("faft.txt","wt");
			for( vector<KeyPoint>::iterator it = img_data.orb_feats.begin(); it != img_data.orb_feats.end(); ++it )
				mrpt::system::os::fprintf(fo,"%.2f %.2f\n", it->pt.x, it->pt.y );
			mrpt::system::os::fclose(fo);
		}
#endif

		VERBOSE_LEVEL(2) << "	[sVO -- Stg2: Detect] Detected: " << img_data.orb_feats.size() << " feats" << endl;
	}
	else if( params_detect.detect_method == TDetectParams::dmFAST_ORB )
	{
		CImage & input_im = img_data.pyr.images[0];						// shortcut
        Mat img( input_im.getAs<IplImage>() )/*image*/, desc_aux/*descriptors*/;
		vector<KeyPoint> feat_vector;
		cv::FastFeatureDetector(5).detect( img, feat_vector );			// detect keypoints
		ORB().operator()(img, Mat(), feat_vector, desc_aux, true );		// extract descriptors

		// non-maximal suppression
		if( params_detect.non_maximal_suppression )
		{
			if( params_detect.nmsMethod == TDetectParams::nmsmStandard )
			{
				const size_t imgH = input_im.getHeight();
				const size_t imgW = input_im.getWidth();
				this->m_non_max_sup( params_detect.orb_nfeats, feat_vector, desc_aux, img_data.orb_feats, img_data.orb_desc, imgH, imgW );
			}
			else if( params_detect.nmsMethod == TDetectParams::nmsmAdaptive )
				this->m_adaptive_non_max_sup( params_detect.orb_nfeats, feat_vector, desc_aux, img_data.orb_feats, img_data.orb_desc );
			else
				THROW_EXCEPTION("	[sVO -- Stg2: Detect] Invalid non-maximal-suppression method." );
		} // end-if-non-max-sup
		else
		{
			feat_vector.swap(img_data.orb_feats);
			img_data.orb_desc = desc_aux;					// this should copy just the header
		}
		
		VERBOSE_LEVEL(2) << "	[sVO -- Stg2: Detect] Detected: " << img_data.orb_feats.size() << " feats" << endl;
	}
	else if( params_detect.detect_method == TDetectParams::dmFASTER )
	{
	    // ***********************************
	    // FASTER method
	    // ***********************************
        // Use a dynamic threshold to maintain a target number of features per square pixel.
        if (m_threshold.size()!=nPyrs) m_threshold.assign(nPyrs, params_detect.initial_FAST_threshold);

        m_next_gui_info->stats_feats_per_octave.resize(nPyrs); // Reserve size for stats
        m_next_gui_info->stats_FAST_thresholds_per_octave.resize(nPyrs);

        // Evaluate the KLT response of all features to discard those in texture-less zones:
        const unsigned int KLT_win	= params_detect.KLT_win;
        const double minimum_KLT	= params_detect.minimum_KLT;

        for (size_t octave=0;octave<nPyrs;octave++)
        {
            const std::string sProfileName = mrpt::format("stg2.detect.oct=%u",static_cast<unsigned int>(octave));
            m_profiler.enter(sProfileName.c_str());

            CFeatureExtraction::detectFeatures_SSE2_FASTER12(
                img_data.pyr.images[octave],
                img_data.pyr_feats[octave],
                m_threshold[octave],
                false /* don't append to list, overwrite it */,
                octave,
                &img_data.pyr_feats_index[octave] /* row-indexed list of features */
                );

//            cout << endl << "(" << octave << "): " << img_data.pyr_feats[octave].size() << endl;
            const size_t nFeats = img_data.pyr_feats[octave].size();

            // fill in the identifiers of the features
            for( size_t id = 0; id < nFeats; ++id )
                img_data.pyr_feats[octave][id].ID = this->m_lastID++;

            if (update_dyn_thresholds)
            {
                // Compute feature density & adjust dynamic threshold:
                const size_t nFeats= img_data.pyr_feats[octave].size();
                const mrpt::utils::TImageSize img_size = img_data.pyr.images[octave].getSize();
                const double feats_density = nFeats / static_cast<double>( img_size.x * img_size.y);

                if (feats_density<0.8*params_detect.target_feats_per_pixel)
                    m_threshold[octave] = std::max(1, m_threshold[octave]-1);
                else if (feats_density>1.2*params_detect.target_feats_per_pixel)
                    m_threshold[octave] = m_threshold[octave]+1;

                // Save stats for the GUI:
                m_next_gui_info->stats_feats_per_octave[octave] = nFeats;
                m_next_gui_info->stats_FAST_thresholds_per_octave[octave] = m_threshold[octave];
            }

            // compute the KLT response
            const std::string subSectionName = mrpt::format("stg2.detect.klt.oct=%u",static_cast<unsigned int>(octave));
            m_profiler.enter(subSectionName.c_str());

            const TImageSize img_size = img_data.pyr.images[octave].getSize();
            const TImageSize img_size_min( KLT_win+1, KLT_win+1 );
            const TImageSize img_size_max( img_size.x-KLT_win-1, img_size.y-KLT_win-1 );

            size_t nPassed = 0; // Number of feats in this octave that pass the KLT threshold (for stats only)

            for (size_t i=0;i<img_data.pyr_feats[octave].size();i++)
            {
                TSimpleFeature &f = img_data.pyr_feats[octave][i];
                const TPixelCoord pt = f.pt;
                if (pt.x>=img_size_min.x && pt.y>=img_size_min.y && pt.x<img_size_max.x && pt.y<img_size_max.y) {
                     f.response = img_data.pyr.images[octave].KLT_response(pt.x,pt.y,KLT_win);
                     if (f.response>=minimum_KLT) nPassed++;
                }
                else f.response = 0;
            } // end-for

            nFeatsPassingKLTPerOctave[octave] = nPassed;
            m_profiler.leave(subSectionName.c_str());

            // perform non-maximal suppression [5x5] window (if enabled)
            if( params_detect.non_maximal_suppression )
            {
                // Non-maximal supression using KLT response
                const string subSectionName2 = mrpt::format("stg2.detect.non-max.oct=%u",static_cast<unsigned int>(octave));
                m_profiler.enter(subSectionName2.c_str());
                m_non_max_sup( img_data, octave );
                m_profiler.leave(subSectionName2.c_str());
            } // end non-maximal suppression

            m_profiler.leave(sProfileName.c_str()); // end detect
        }
	}
    else
        THROW_EXCEPTION("	[sVO -- Stg2: Detect] ERROR: Unknown detection method")

	if (params_gui.show_gui && params_gui.draw_all_raw_feats)
	{
		// (It's almost as efficient to directly draw these small feature marks at this point
		// rather than send all the info to the gui thread and then draw there. A quick test shows
		// a gain of 75us -> 50us only, so don't optimize unless efficiency pushes really hard).
		m_profiler.enter("stg2.draw_feats");

		if( params_detect.detect_method == TDetectParams::dmFASTER )
		{
            for (size_t octave=0;octave<nPyrs;octave++)
            {
                const TSimpleFeatureList &f1 = img_data.pyr_feats[octave];
                const size_t n1 = f1.size();

                const bool org_img_color = gui_image.isColor();
                unsigned char* ptr1 = gui_image.get_unsafe(0,0);
                const size_t img1_stride = gui_image.getRowStride();
                for(size_t i=0;i<n1;++i)
                {
                    const int x=f1[i].pt.x; const int y=f1[i].pt.y;
                    unsigned char* ptr = ptr1 + img1_stride*y + (org_img_color ? 3*x:x);
                    if (org_img_color) {
                        *ptr++ = 0x00;
                        *ptr++ = 0x00;
                        *ptr++ = 0xFF;
                    }
                    else {
                        *ptr = 0xFF;
                    }
                } // end-for
            } // end-for
		} // end-if
		else
		{
		    const bool org_img_color = gui_image.isColor();
            unsigned char* ptr1 = gui_image.get_unsafe(0,0);
            const size_t img1_stride = gui_image.getRowStride();
            const vector<cv::KeyPoint> &f1 = img_data.orb_feats;
            for( vector<cv::KeyPoint>::const_iterator it = f1.begin(); it != f1.end(); ++it )
            {
                const int x = it->pt.x; const int y = it->pt.y;
                unsigned char* ptr = ptr1 + img1_stride*y + (org_img_color ? 3*x:x);
                if (org_img_color) {
                    *ptr++ = 0x00;
                    *ptr++ = 0x00;
                    *ptr++ = 0xFF;
                }
                else {
                    *ptr = 0xFF;
                } // end-else
            } // end-for
		} // end-else

		m_profiler.leave("stg2.draw_feats");
	}

    if( params_detect.detect_method == TDetectParams::dmORB || params_detect.detect_method == TDetectParams::dmFAST_ORB )
    {
        m_next_gui_info->text_msg_from_detect += mrpt::format("%lu ", img_data.orb_feats.size());
    }
    else
    {
        // for the GUI thread
        string sPassKLT;
        for (size_t i=0;i<nPyrs;i++)
            sPassKLT+=mrpt::format("%u/",static_cast<unsigned int>(nFeatsPassingKLTPerOctave[i]));

        string sDetect;
        for (size_t i=0;i<nPyrs;i++)
            sDetect += mrpt::format("%u/",static_cast<unsigned int>(img_data.pyr_feats[i].size()));

        string aux = mrpt::format( "\n%s feats (%s passed KLT)", sDetect.c_str(), sPassKLT.c_str() );
        m_next_gui_info->text_msg_from_detect += aux;
    } // end-else

	m_profiler.leave("_stg2");
}
