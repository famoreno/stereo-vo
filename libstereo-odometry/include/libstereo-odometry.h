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

/** \file libstereo-odometry.h
  * \brief This file exposes the public C++ API and data types of the stereo-odometry library
  */

// opencv
#include <cv.h>
#include <highgui.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>

// mrpt
#include <mrpt/utils/CImage.h>
#include <mrpt/utils/CConfigFile.h>
#include <mrpt/utils/CTimeLogger.h>

#include <mrpt/vision/CStereoRectifyMap.h>
#include <mrpt/vision/CImagePyramid.h>
#include <mrpt/vision/TSimpleFeature.h>
#include <mrpt/vision/CFeatureExtraction.h>

#include <mrpt/poses/CPose3D.h>
#include <mrpt/poses/CPose3DRotVec.h>

#include <mrpt/math/CMatrixB.h>
#include <mrpt/math/lightweight_geom_data.h>

#include <mrpt/system/filesystem.h>
#include <mrpt/system/os.h>
#include <mrpt/system/threads.h>

#include <mrpt/opengl/CSetOfObjects.h>
#include <mrpt/synch/CThreadSafeVariable.h>
#include <mrpt/obs/CObservationStereoImages.h>
#include <mrpt/gui/CDisplayWindow3D.h>

using namespace mrpt::utils;
using namespace mrpt::math;
using namespace mrpt::poses;
using namespace mrpt::system;

#include <fstream>

#define SHOW_WARNING(MSG) cout << "[Visual Odometry] " << MSG << endl;
#define DUMP_VO_ERROR_CODE( _CODE ) \
	cout << "[sVO] ERROR: "; \
	switch( _CODE ) {	case voecBadCondNumber    : cout << "(LS) Bad Condition Number" << endl; break; \
						case voecIncrFuncCostStg1 : cout << "(LS) (Initial) Function cost increased too many times" << endl; break; \
						case voecIncrFuncCostStg2 : cout << "(LS) (Final) Function cost increased too many times" << endl; break; \
						case voecFirstIteration   : cout << "First iteration, no change in pose can be computed" << endl; break; \
						case voecBadTracking	  : cout << "Number of tracked features is low " << endl; break; \
						case voecNone : default   : cout << "None" << endl; break; }

struct KpRadiusSorter : public std::binary_function<size_t,size_t,bool>
{
	const std::vector<double> & m_radius_data;
	KpRadiusSorter( const std::vector<double> & radius_data ) : m_radius_data( radius_data ) { }
	bool operator() (size_t k1, size_t k2 ) const {
		return ( m_radius_data[k1] > m_radius_data[k2] );
	}
};

typedef struct t_change_in_pose_output {
	mrpt::poses::CPose3D	change_in_pose;
	std::vector<double>		out_residual;
	unsigned int			num_gn_iterations;
	unsigned int			num_gn_final_iterations;

	t_change_in_pose_output() : change_in_pose( mrpt::poses::CPose3D() ), out_residual( std::vector<double>() ), num_gn_iterations( 0 ), num_gn_final_iterations( 0 ) {}

} t_change_in_pose_output;

// Namespace of the Robust Stereo Odometry (RSO) library
namespace rso
{
	using namespace std;
	using namespace mrpt::vision;

	typedef std::vector<std::pair<size_t,size_t> > vector_index_pairs_t;
	typedef std::vector<std::pair<mrpt::utils::TPixelCoord,mrpt::utils::TPixelCoord> > vector_pixel_pairs_t;

	enum VOErrorCode { voecNone, voecBadCondNumber, voecIncrFuncCostStg1, voecIncrFuncCostStg2, voecFirstIteration, voecBadTracking };	//!< Error code for the minimization process in the visual odometry engine

	/** The main class of libstereo-odometry: it implements the Stereo Odometry problem and holds all the related
	  *   options and parameters.
	  */
	class CStereoOdometryEstimator
	{
	public:
		struct TStereoOdometryRequest;  // Forward decl. (declaration below)
		struct TStereoOdometryResult;   // Forward decl. (declaration below)

	//---------------------------------------------------------------
	/** @name Main API methods
		@{
		/** The main entry point: process a new pair of images and returns the estimated camera pose increment.
		  *
		  */
		void processNewImagePair(
			TStereoOdometryRequest & request_data,
			TStereoOdometryResult & result );

		// this custom method gets info from two different frames and computes the change in pose between them
		bool getChangeInPose(
					const vector_index_pairs_t			& tracked_pairs,
					const vector<cv::DMatch>			& pre_matches,
					const vector<cv::DMatch>			& cur_matches,
					const vector<cv::KeyPoint>			& pre_left_feats,
					const vector<cv::KeyPoint>			& pre_right_feats,
					const vector<cv::KeyPoint>			& cur_left_feats,
					const vector<cv::KeyPoint>			& cur_right_feats,
					const mrpt::utils::TStereoCamera	& stereo_camera,
					TStereoOdometryResult				& result,									// output
					const vector<double>				& ini_estimation = vector<double>(6,0) );	// initial estimation of the pose (if any)	

		// this custom method computes the coords of a set of features after a change in pose
		// TO DO: CONSIDER REMOVE THIS
		/*void getProjectedCoords(
					const vector<cv::DMatch>			& pre_matches,						// all the matches in the previous keyframe
					const vector<cv::KeyPoint>			& pre_left_feats,
					const vector<cv::KeyPoint>			& pre_right_feats,					// all the features in the previuous keyframe
					const vector<bool>					& other_matches_tracked,			// tracking info for OTHER matches [size=A]
					const mrpt::utils::TStereoCamera	& stereo_camera,					// stereo camera intrinsic and extrinsic parameters
					const CPose3D						& change_pose,						// the estimated change in pose between keyframes
					vector< pair<TPixelCoordf,TPixelCoordf> > & pro_pre_feats );			// [out] coords of the features in the left image [size=B (only for those whose 'other_matches_tracked' entry is false]
		*/

		// this custom method computes the coords of a set of features after a change in pose
		void getProjectedCoords(
					const vector<cv::DMatch>			& pre_matches,						// all the matches in the previous keyframe
					const vector<cv::KeyPoint>			& pre_left_feats,
					const vector<cv::KeyPoint>			& pre_right_feats,					// all the features in the previuous keyframe
					const vector< pair<int,float> >		& other_matches_tracked,			// tracking info for OTHER matches [size=A]
					const mrpt::utils::TStereoCamera	& stereo_camera,					// stereo camera intrinsic and extrinsic parameters
					const CPose3D						& change_pose,						// the estimated change in pose between keyframes
					vector< pair<TPixelCoordf,TPixelCoordf> > & pro_pre_feats );			// [out] coords of the features in the left image [size=B (only for those whose 'other_matches_tracked' entry is false]

		bool saveStateToFile( const string & filename );	// returns false on error
		bool loadStateFromFile( const string & filename );	// returns false on error

		void inline dumpToConsole()
		{
			cout << "---------------------------------------------------------" << endl;
			cout << " Visual Odometry parameters" << endl;
			cout << "---------------------------------------------------------" << endl;
			params_general.dumpToConsole();
			params_rectify.dumpToConsole();
			params_detect.dumpToConsole();
			params_lr_match.dumpToConsole();
			params_least_squares.dumpToConsole();
			params_gui.dumpToConsole();
		}

		/*void insertDataInEngine(
			const vector<cv::KeyPoint>	& pre_left_feats,
			const cv::Mat				& pre_right_desc,
			const vector<cv::KeyPoint>	& pre_right_feats,
			const cv::Mat				& pre_left_desc,
			const vector<cv::DMatch>	& pre_matches,
			const vector<size_t>		& pre_matches_ID,
			const vector<cv::KeyPoint>	& cur_left_feats,
			const cv::Mat				& cur_right_desc,
			const vector<cv::KeyPoint>	& cur_right_feats,
			const cv::Mat				& cur_left_desc,
			const vector<cv::DMatch>	& cur_matches,
			const vector<size_t>		& cur_matches_ID );*/

	/** @} */  // End of main API methods
	//---------------------------------------------------------------
	/** @name Public data fields
		@{ */

		struct TStereoOdometryRequest
		{
			/** The input images */
			mrpt::obs::CObservationStereoImagesPtr  stereo_imgs;

			/** The camera params */
			mrpt::utils::TStereoCamera stereo_cam;

			/** Precomputed data */
			bool use_precomputed_data;
			vector<cv::KeyPoint> *precomputed_left_feats, *precomputed_right_feats;
			cv::Mat *precomputed_left_desc, *precomputed_right_desc;
			vector<cv::DMatch> *precomputed_matches;
			vector<size_t> *precomputed_matches_ID;

			/** Repeat */
			bool repeat;

			TStereoOdometryRequest() :  stereo_imgs(),
                                        stereo_cam(),
                                        use_precomputed_data(false),
                                        precomputed_left_feats(NULL),
                                        precomputed_right_feats(NULL),
                                        precomputed_left_desc(NULL),
                                        precomputed_right_desc(NULL),
                                        precomputed_matches(NULL),
										precomputed_matches_ID(NULL),
										repeat(false) {}
		};

		struct TStereoOdometryResult
		{
			// least square results
		    CPose3D			outPose;
			vector< pair<size_t,size_t> > outliers;
			vector<double>	out_residual;
			int				num_it, num_it_final;

			bool			valid;
		    VOErrorCode		error_code;
			size_t			tracked_feats_from_last_KF, tracked_feats_from_last_frame;

			pair<size_t,size_t> detected_feats;
			size_t			stereo_matches;

		    TStereoOdometryResult() : 
				outPose(CPose3D()), 
				outliers(vector< pair<size_t,size_t> >()),
				valid(false), 
				tracked_feats_from_last_KF(0), 
				tracked_feats_from_last_frame(0), 
				out_residual(vector<double>()), 
				num_it(0), 
				num_it_final(0), 
				error_code(voecNone), 
				stereo_matches(0)
				{ 
					detected_feats.first = detected_feats.second = 0; 
				}
		};

		struct TGeneralParams
		{
			TGeneralParams();
			bool vo_use_matches_ids;	//!< Indicates if we should track the IDs of the matches through time (Default = false)
			bool vo_save_files;			//!< Indicates if we should store some information of the system in files as the process goes (Default = false)
			bool vo_debug;				//!< Indicates if we are debugging the application (auxiliary) (Default = false)
			bool vo_pause_it;			//!< Indicates if we want to pause the application after each iteration (Default = false)
			string vo_out_dir;			//!< Sets the output directory for saving debug files (Default = "out")

			void dumpToConsole()
			{
				cout << "	[GENERAL]	Track the IDs of the matches?: ";
				vo_use_matches_ids ? cout << "Yes" : cout << "No"; cout << endl;
				cout << "	[GENERAL]	Save information files?: ";
				vo_save_files ? cout << "Yes" : cout << "No"; cout << endl;
				cout << "	[GENERAL]	Debug?: ";
				vo_debug ? cout << "Yes" : cout << "No"; cout << endl;
				cout << "	[GENERAL]	Pause after each iteration?: ";
				vo_pause_it ? cout << "Yes" : cout << "No"; cout << endl;
				cout << "	[GENERAL]	Output directory: " << vo_out_dir << endl;
			}
		};

		/** Parameters for the LS optimization stage */
		struct TLeastSquaresParams
		{
			TLeastSquaresParams();

			// Parameters for optimize_*()
			// -------------------------------------
			bool	use_robust_kernel;
			double	kernel_param;
			size_t	max_iters;
			size_t	initial_max_iters;
			double	max_error_per_obs_px;	//!< default: 1e-3
			double	std_noise_pixels;		//!< default: 1, the standard deviation assumed for feature coordinates (this parameter is only needed to scale the uncertainties of reconstructed LMs with unknown locations).
			size_t	max_incr_cost;
			double	residual_threshold;
			size_t	bad_tracking_th;
			bool	use_previous_pose_as_initial;
			bool	use_custom_initial_pose;		// use the (input) custom initial pose but do not save it for later use (useful when getChangeInPose is called). This setting has priority over 'use_previous_pose_as_initial'
			// -------------------------------------

			void dumpToConsole()
			{
				cout << "	[LS]		Use Robust Kernel?: ";
				use_robust_kernel ? cout << "Yes" : cout << "No"; cout << endl;
				if( use_robust_kernel )
					cout << "	[LS]		Robust kernel param: " << kernel_param << endl;
				cout << "	[LS]		Use previous estimated pose as initial pose?: ";
				use_previous_pose_as_initial ? cout << "Yes" : cout << "No"; cout << endl;
				cout << "	[LS]		Initial maximum number of iterations: " << initial_max_iters << endl;
				cout << "	[LS]		Final maximum number of iterations: " << max_iters << endl;
				cout << "	[LS]		Maximum error in pixels for observation: " << max_error_per_obs_px << endl;
				cout << "	[LS]		STD noise for the feature coordinates: " << std_noise_pixels << endl;
				cout << "	[LS]		Maximum allowed number of times the cost can grow: " << max_incr_cost << endl;
				cout << "	[LS]		Residual threshold for detecting outliers: " << residual_threshold << endl;
				cout << "	[LS]		Bad tracking threshold: " << bad_tracking_th << endl;
			}
		};

		/** Parameters for the image rectification & preprocessing stage */
		struct TRectifyParams
		{
			TRectifyParams();
			int nOctaves;					//!< Number of octaves to create

			void dumpToConsole()
			{
				cout << "	[RECTIFY]	Number of octaves: " << nOctaves << endl;
			}
		};

		/** Parameters for the GUI windows and visualization */
		struct TGUIParams
		{
			TGUIParams();
			bool show_gui;					//!< Show GUI? (Default = true)
			bool draw_all_raw_feats;		//!< Default = false
			bool draw_lr_pairings;			//!< Default = false
			bool draw_tracking;				//!< Default = true

			void dumpToConsole()
			{
				cout << "	[GUI]		Show GUI?: "; show_gui ? cout << "Yes" : cout << "No"; cout << endl;
				cout << "	[GUI]		Draw Raw Features?: "; draw_all_raw_feats ? cout << "Yes" : cout << "No"; cout << endl;
				cout << "	[GUI]		Draw Stereo Matches?: "; draw_lr_pairings ? cout << "Yes" : cout << "No"; cout << endl;
				cout << "	[GUI]		Draw Tracked Matches?: "; draw_tracking ? cout << "Yes" : cout << "No"; cout << endl;
			}
		};

		/** Parameters for the keypoint detection */
		struct TDetectParams
		{
			enum NMSMethod { nmsmStandard, nmsmAdaptive };	//!< Non-maximal suppression method: Standard or Adaptive
		    enum TDMethod { dmORB, dmFAST_ORB, dmFASTER, dmKLT };	//!< Feature detection method: FASTER (not implemented), ORB or FAST+ORB

			TDetectParams();
			double		target_feats_per_pixel;		//!< Desired number of features per square pixel (Default: 10/1000)
			int			initial_FAST_threshold;		//!< The initial threshold of the FAST feature detector, which will be dynamically adapted to fulfill \a target_feats_per_pixel (Default=15)
			bool		non_maximal_suppression;	//!< Enable/disable the non-maximal suppression after the detection (5x5 windows is used)
			int			KLT_win;					//!< Window for the KLT response evaluation (Default: 4)
			double		minimum_KLT;				//!< Minimum KLT response not to discard a feature as being in a textureless zone (Default: 10)
			TDMethod	detect_method;				//!< Method to detect features (also affects to the matching process)
			NMSMethod	nmsMethod;					//!< Method to perform non maximal suppression
			size_t		orb_nfeats;					//!< Number of features to be detected (only for ORB)
			size_t		orb_nlevels;				//!< The number of pyramid levels
			size_t		min_distance;				//!< The allowed minimun distance between features

			int			fast_min_th, fast_max_th;	//!< Limits for the FAST detector threshold

			void dumpToConsole()
			{
				cout << "	[DETECT]	Detection method: ";
				switch( detect_method )
				{
					case dmFASTER : cout << "FASTER" << endl; break;
					case dmORB : cout << "ORB" << endl; break;
					case dmFAST_ORB : cout << "FAST + ORB" << endl; break;
				}
				if( detect_method == dmORB )
				{
					cout << "	[DETECT]	Number of desired ORB features: " << orb_nfeats << endl;
					cout << "	[DETECT]	Number of desired ORB levels in scale space: " << orb_nlevels << endl;
				}
				if( detect_method == dmFASTER || detect_method == dmFAST_ORB )
				{				
					cout << "	[DETECT]	Desired number of features per square pixel: " << target_feats_per_pixel << endl;
					cout << "	[DETECT]	Initial FAST threshold: " << initial_FAST_threshold << endl;
					cout << "	[DETECT]	FAST threshold limits: " << fast_min_th << ":" << fast_max_th << endl;
					cout << "	[DETECT]	Window for the KLT response evaluation: " << KLT_win << endl;
					cout << "	[DETECT]	Minimum KLT response to consider a feature: " << minimum_KLT << endl;
				}
				cout << "	[DETECT]	Perform Non Maximal Suppression (NMS)?: "; 
				non_maximal_suppression ? cout << "Yes" : cout << "No"; cout << endl;
				cout << "	[DETECT]	NMS Method: ";
				switch( nmsMethod )
				{
					case nmsmStandard : cout << "Standard" << endl; break;
					case nmsmAdaptive : cout << "Adaptive" << endl; break;
				}
				cout << "	[DETECT]	Allowed minimum distance between features: " << min_distance << endl;
			}
		};

		struct TLeftRightMatchParams
		{
			TLeftRightMatchParams();
			uint32_t maximum_SAD;				//!< The maximum SAD value to consider a pairing as a potential match (Default: ~400)
			bool     enable_robust_1to1_match;	//!< Only match if a pair of L/R features have the best match score to each other (default: false)
			double	 max_SAD_ratio;				//!< The maximum ratio between the two smallest SAD when searching for pairings (Default: 0.5)
			bool     rectified_images;			//!< Indicates if the stereo pair has parallel optical axes
			double   max_y_diff;				//!< Maximum allowed distance in pixels from the same row in the images for corresponding feats
			double   orb_max_distance;			//!< Maximum allowed Hamming distance between a pair of features to be considered a match
			double   min_z, max_z;				//!< Min/Max value for the Z coordinate of 3D feature to be considered (reject too close/far features)

			int		 orb_min_th, orb_max_th;	//!< Limits for the ORB matching threshold

			void dumpToConsole()
			{
				cout << "	[MATCH]		Enable robust 1 to 1 match?: ";
				enable_robust_1to1_match ? cout << "Yes" : cout << "No"; cout << endl;
				cout << "	[MATCH]		Maximum Sum of Absolute Differences (SAD): " << maximum_SAD << endl;
				cout << "	[MATCH]		Maximum SAD ratio: " << max_SAD_ratio << endl;
				cout << "	[MATCH]		Stereo pair has parallel optical axes?: ";
				rectified_images ? cout << "Yes" : cout << "No"; cout << endl;
				if( rectified_images )
					cout << "	[MATCH]		Maximum 'y' distance allowed between matched features: " << max_y_diff << endl;
				cout << "	[MATCH]		Maximum ORB distance: " << orb_max_distance << endl;
				cout << "	[MATCH]		ORB distance limits: " << orb_min_th << "/" << orb_max_th << endl;
				cout << "	[MATCH]		Min/Max value for the Z coordinate of 3D feature to be considered: " << min_z << "/" << max_z << endl;
			}
		};

		/** Different parameters for the SRBA methods */
		TRectifyParams			params_rectify;
		TDetectParams			params_detect;
		TLeastSquaresParams		params_least_squares;
		TLeftRightMatchParams	params_lr_match;
		TGUIParams				params_gui;
		TGeneralParams			params_general;

	/** @} */  // End of data fields
	//---------------------------------------------------------------

		/** Default constructor */
		CStereoOdometryEstimator();

		/** Destructor: close windows & clean up  */
		~CStereoOdometryEstimator();

		/** Enable or disables time profiling of all operations (default=enabled), which will be reported upon destruction */
		void inline enable_time_profiler(bool enable=true) { m_profiler.enable(enable); }

		/** Access to the time profiler */
		inline mrpt::utils::CTimeLogger & get_time_profiler() { return m_profiler; }

		/** Changes the verbosity level: 0=None (only critical msgs), 1=verbose, 2=so verbose you'll have to say "Stop!" */
		inline void setVerbosityLevel(int level) { m_verbose_level = level; }

		/** Sets and gets FAST threshold */
		//inline int getFASTThreshold( ) { return params_detect.initial_FAST_threshold; }
		//inline void setFASTThreshold( int value ) { params_detect.initial_FAST_threshold = std::min(30,std::max(5,value)); }
		//inline void resetFASTThreshold( ) { params_detect.initial_FAST_threshold = 20;}

		inline int getFASTThreshold( ) { return m_current_fast_th; }
		inline void setFASTThreshold( int value ) { m_current_fast_th = std::min(params_detect.fast_max_th,std::max(params_detect.fast_min_th,value)); }
		inline void resetFASTThreshold( ) { m_current_fast_th = params_detect.initial_FAST_threshold; }
		inline bool isFASTThMin( ) { return m_current_fast_th == params_detect.fast_min_th; }
		inline bool isFASTThMax( ) { return m_current_fast_th == params_detect.fast_max_th; }

		/** Sets and gets ORB matching threshold */
		inline int getORBThreshold( ) { return m_current_orb_th; }
		inline void setORBThreshold( int value ) { m_current_orb_th = std::min(params_lr_match.orb_max_th,std::max(params_lr_match.orb_min_th,value)); }
		inline void resetORBThreshold( ) { m_current_orb_th = params_lr_match.orb_max_distance;}
		inline bool isORBThMin( ) { return m_current_orb_th == params_lr_match.orb_min_th; }
		inline bool isORBThMax( ) { return m_current_orb_th == params_lr_match.orb_max_th; }

		bool keyPressedOnGUI(); //!< Return true if any key was pressed on the GUI window \sa  getKeyPressedOnGUI()

		/** Return the key code of the last key pressed on the GUI window:
		  * ==0: means no key pressed on window; >0: is the key code, -1: the window was closed by the user.
		  * \sa getKeyPressedOnGUI(), keyPressedOnGUI()
		  */
		int getKeyPressedOnGUI();

		/** Loads configuration from an INI file
		  * Sections must be (in this order) related to: RECTIFY, DETECT, MATCH, LEAST_SQUARES, GUI, GENERAL
		  */
		void loadParamsFromConfigFile( const mrpt::utils::CConfigFile &iniFile, const std::vector<std::string> &sections)
		{
			ASSERT_(sections.size() == 6)	// one section for type of params:

			if( sections[0].size() > 0 )	// rectify
				params_rectify.nOctaves						= iniFile.read_int(sections[0], "nOctaves", params_rectify.nOctaves, false);

			if( sections[1].size() > 0 )	// detection
			{
				params_detect.target_feats_per_pixel		= iniFile.read_double(sections[1], "target_feats_per_pixel", params_detect.target_feats_per_pixel, false);
				params_detect.initial_FAST_threshold		= iniFile.read_int(sections[1], "initial_FAST_threshold", params_detect.initial_FAST_threshold, false);
				params_detect.KLT_win						= iniFile.read_int(sections[1], "KLT_win", params_detect.KLT_win, false);
				params_detect.minimum_KLT					= iniFile.read_double(sections[1], "minimum_KLT", params_detect.minimum_KLT, false);
				params_detect.non_maximal_suppression	    = iniFile.read_bool(sections[1], "non_maximal_suppression", params_detect.non_maximal_suppression, false);
				int aux										= iniFile.read_int(sections[1], "detect_method", params_detect.detect_method, false);
				switch(aux)
				{
					case 0:
					default: params_detect.detect_method = TDetectParams::dmORB; break;
					case 1:  params_detect.detect_method = TDetectParams::dmFAST_ORB; break;
					case 2:  params_detect.detect_method = TDetectParams::dmFASTER; break;
				}
                params_detect.nmsMethod						= iniFile.read_int(sections[1], "non_max_supp_method", params_detect.nmsMethod, false) == 0 ? TDetectParams::nmsmStandard : TDetectParams::nmsmAdaptive;
                params_detect.orb_nfeats                    = iniFile.read_int(sections[1],"orb_nfeats",params_detect.orb_nfeats,false);
				params_detect.orb_nlevels					= iniFile.read_int(sections[1],"orb_nlevels",params_detect.orb_nlevels,false);
				params_detect.min_distance					= iniFile.read_int(sections[1],"min_distance",params_detect.min_distance,false);

				params_detect.fast_min_th					= iniFile.read_int(sections[1],"fast_min_th",params_detect.fast_min_th,false);
				params_detect.fast_max_th					= iniFile.read_int(sections[1],"fast_max_th",params_detect.fast_max_th,false);
			}

			if( sections[2].size() > 0 )	// left right matching
			{
				params_lr_match.enable_robust_1to1_match	= iniFile.read_bool(sections[2], "enable_robust_1to1_match", params_lr_match.enable_robust_1to1_match, false);
				params_lr_match.rectified_images	        = iniFile.read_bool(sections[2], "rectified_images", params_lr_match.rectified_images, false);
				params_lr_match.max_SAD_ratio				= iniFile.read_double(sections[2], "max_SAD_ratio", params_lr_match.max_SAD_ratio, false);
				params_lr_match.maximum_SAD					= iniFile.read_int(sections[2], "maximum_SAD", params_lr_match.maximum_SAD, false);

				params_lr_match.orb_min_th					= iniFile.read_int(sections[2], "orb_min_th", params_lr_match.orb_min_th, false);
				params_lr_match.orb_max_th					= iniFile.read_int(sections[2], "orb_max_th", params_lr_match.orb_max_th, false);

				params_lr_match.max_y_diff				    = iniFile.read_double(sections[2], "max_y_diff", params_lr_match.max_y_diff, false);
                params_lr_match.orb_max_distance            = iniFile.read_double(sections[2], "orb_max_distance", params_lr_match.orb_max_distance, false);;
				params_lr_match.min_z						= iniFile.read_double(sections[2], "min_z", params_lr_match.min_z, false);
				params_lr_match.max_z						= iniFile.read_double(sections[2], "max_z", params_lr_match.max_z, false);
			}

			if( sections[3].size() > 0 )	// least squares
			{
				params_least_squares.kernel_param			= iniFile.read_double(sections[3], "kernel_param", params_least_squares.kernel_param, false);
				params_least_squares.max_error_per_obs_px	= iniFile.read_double(sections[3], "max_error_per_obs_px", params_least_squares.max_error_per_obs_px, false);
				params_least_squares.max_iters				= iniFile.read_int(sections[3], "max_iters", params_least_squares.max_iters, false);
				params_least_squares.initial_max_iters      = iniFile.read_int(sections[3], "initial_max_iters", params_least_squares.initial_max_iters, false);
				params_least_squares.max_incr_cost          = iniFile.read_int(sections[3], "max_incr_cost", params_least_squares.max_incr_cost, false);
				params_least_squares.std_noise_pixels		= iniFile.read_double(sections[3], "std_noise_pixels", params_least_squares.std_noise_pixels, false);
				params_least_squares.residual_threshold		= iniFile.read_double(sections[3], "residual_threshold", params_least_squares.residual_threshold, false);
				params_least_squares.use_robust_kernel		= iniFile.read_bool(sections[3], "use_robust_kernel", params_least_squares.use_robust_kernel, false);
				params_least_squares.bad_tracking_th        = iniFile.read_int(sections[3], "bad_tracking_th", params_least_squares.bad_tracking_th, false);
				params_least_squares.use_previous_pose_as_initial = iniFile.read_bool(sections[3], "use_previous_pose_as_initial", params_least_squares.use_previous_pose_as_initial, false);
			}

			if( sections[4].size() > 0 )	// GUI
			{
				params_gui.show_gui							= iniFile.read_bool(sections[4], "show_gui", params_gui.show_gui, false);
				params_gui.draw_all_raw_feats				= iniFile.read_bool(sections[4], "draw_all_raw_feats", params_gui.draw_all_raw_feats, false);
				params_gui.draw_lr_pairings					= iniFile.read_bool(sections[4], "draw_lr_pairings", params_gui.draw_lr_pairings, false);
				params_gui.draw_tracking					= iniFile.read_bool(sections[4], "draw_tracking", params_gui.draw_tracking, false);
			}

			if( sections[5].size() > 0 )	// GENERAL
			{
				params_general.vo_use_matches_ids		= iniFile.read_bool(sections[5], "vo_use_matches_ids", params_general.vo_use_matches_ids, false);
				params_general.vo_save_files			= iniFile.read_bool(sections[5], "vo_save_files", params_general.vo_save_files, false);
				params_general.vo_debug					= iniFile.read_bool(sections[5], "vo_debug", params_general.vo_debug, false);
				params_general.vo_pause_it				= iniFile.read_bool(sections[5], "vo_pause_it", params_general.vo_pause_it, false);
				params_general.vo_out_dir				= iniFile.read_string(sections[5], "vo_out_dir", params_general.vo_out_dir, false );
			}

			resetFASTThreshold();
			resetORBThreshold();
		}

		/** Loads configuration from an INI file from its name */
		void loadParamsFromConfigFileName( const std::string &fileName, const std::vector<std::string> &sections )
		{
			// ORDER: Rectify, detect, match, least_squares, gui
			ASSERT_(mrpt::system::fileExists(fileName))
			mrpt::utils::CConfigFile iniFile(fileName);
			loadParamsFromConfigFile(iniFile,sections);
		} // end loadParamsFromConfigFileName

		/** Sets/resets the match IDs generator */
		void inline resetIds( const bool reset = true ) { this->m_reset = reset; }
		void inline setIds( const vector<size_t> & ids ) { 
			if( this->m_current_imgpair ) {
				this->m_current_imgpair->orb_matches_ID.resize(ids.size()); 
				std::copy(ids.begin(), ids.end(), this->m_current_imgpair->orb_matches_ID.begin()); 
			}
			else {
				this->m_prev_imgpair->orb_matches_ID.resize(ids.size()); 
				std::copy(ids.begin(), ids.end(), this->m_prev_imgpair->orb_matches_ID.begin()); 
			}
		} // end-setIds

		CImage & getRefCurrentImageLeft() { return params_gui.show_gui ? m_gui_info->img_left : m_next_gui_info->img_left; }
		CImage & getRefCurrentImageRight() { return params_gui.show_gui ? m_gui_info->img_right : m_next_gui_info->img_right; }
		CImage getCopyCurrentImageLeft() { CImage aux; params_gui.show_gui ? aux.copyFromForceLoad(m_gui_info->img_left) : aux.copyFromForceLoad(m_next_gui_info->img_left); return aux; }
		CImage getCopyCurrentImageRight() { CImage aux; params_gui.show_gui ? aux.copyFromForceLoad(m_gui_info->img_right) : aux.copyFromForceLoad(m_next_gui_info->img_right); return aux; }

		/** Returns copies to the inner structures */
		void getValues( vector<cv::KeyPoint> & leftKP, vector<cv::KeyPoint> & rightKP,
                        cv::Mat &leftDesc, cv::Mat &rightDesc,
                        vector<cv::DMatch> & matches )
		{
		    leftKP.resize( m_current_imgpair->left.orb_feats.size() );
		    std::copy( m_current_imgpair->left.orb_feats.begin(), m_current_imgpair->left.orb_feats.end(), leftKP.begin() );

		    rightKP.resize( m_current_imgpair->right.orb_feats.size() );
		    std::copy( m_current_imgpair->right.orb_feats.begin(), m_current_imgpair->right.orb_feats.end(), rightKP.begin() );

            m_current_imgpair->left.orb_desc.copyTo( leftDesc );
		    m_current_imgpair->right.orb_desc.copyTo( rightDesc );

		    matches.resize( m_current_imgpair->orb_matches.size() );
		    std::copy( m_current_imgpair->orb_matches.begin(), m_current_imgpair->orb_matches.end(), matches.begin() );
		} // end getValues

	private:
		/** Profiler for all SRBA operations
		  *  Enabled by default, can be disabled with \a enable_time_profiler(false)
		  */
		mutable mrpt::utils::CTimeLogger  m_profiler;
        size_t                            m_lastID;								//!< Identificator of the last tracked feature

		// matches id management (if 'params_general.vo_use_matches_ids' is set)
        size_t                            m_num_tracked_pairs_from_last_kf;
		size_t							  m_num_tracked_pairs_from_last_frame;
		bool							  m_reset;
		vector<size_t>					  m_kf_ids;
		size_t                            m_last_match_ID;						//!< Identificator of the last match ID
		size_t							  m_kf_max_match_ID;

		// orb method: fast detector and orb matching thresholds
		int								m_current_fast_th;
		int								m_current_orb_th;

		bool							m_error_in_tracking;
		VOErrorCode						m_error;
		// ------------------------------------------------------------------------------------------------------------

		int m_verbose_level;													//!< 0: None (only critical msgs), 1: verbose, 2:even more verbose
		unsigned int m_it_counter;												//!< Iteration counter
		vector<double> m_last_computed_pose; 

		struct TImagePairData
		{
			struct img_data_t
			{
				mrpt::vision::CImagePyramid                   pyr;              //!< Pyramid of grayscale images
				std::vector<mrpt::vision::TSimpleFeatureList> pyr_feats;        //!< Features in each pyramid
				std::vector<mrpt::vector_size_t>              pyr_feats_index;  //!< Index of feature indices per row
				std::vector<cv::Mat>                          pyr_feats_desc;   //!< ORB Descriptors of the features

				// orb based:
				std::vector<cv::KeyPoint>                     orb_feats;        //!< ORB based feats, one vector for all the octaves
				cv::Mat                                       orb_desc;         //!< ORB based descriptors
			};

			mrpt::system::TTimeStamp  timestamp;
			img_data_t left, right;

			struct img_pairing_data_t
			{
				/** For this octave, the list of pairings of features: L&R indices as in \a pyr_feats
				  * \note It is assumed that pairings are in top-bottom, left-right order
				  */
				vector_index_pairs_t matches_lr;

				/** For this octave, a vector with length of number of rows in the images, containing the index of the first
				  * matched pairs of features in that row. The indices are those found in \a matches_lr
				  */
				std::vector<size_t> matches_lr_row_index;

			};

			/** For each octave, all the information about L/R pairings */
			std::vector<img_pairing_data_t>  lr_pairing_data;

			/** The ORB matches */
			std::vector<cv::DMatch> orb_matches;

			/** The idx of the ORB matches */
			std::vector<size_t> orb_matches_ID;

			TImagePairData() : timestamp(INVALID_TIMESTAMP)  { }
		};

		typedef stlplus::smart_ptr<TImagePairData> TImagePairDataPtr;

		struct TTrackingData
		{
			/** Pointers to the previous & current image pairs orginal structures. */
			const TImagePairData *prev_imgpair, *cur_imgpair;

			/** For each octave, the paired indices of matched stereo features.
			  * Indices are as seen in \a matches_lr
			  */
			std::vector<vector_index_pairs_t>  tracked_pairs;

			/** Number of features tracked from the last reset (useful for slam applications) */
		};

        /** At any time step, the current (latest) and previous pair of stereo images,
		  *  already processed as pyramids, detectors.
		  */
		TImagePairDataPtr m_current_imgpair, m_prev_imgpair;
		mrpt::vision::CStereoRectifyMap  m_stereo_rectifier;
		std::vector<int> m_threshold;

		/** Updates the row index matrix
        */
		void m_update_indexes( TImagePairData::img_data_t &data, size_t octave );
        
		/** Performs m_non_max_suppression for the detected features
        */
        void m_non_max_sup( TImagePairData::img_data_t &data, size_t octave );	// <-- useless?? consider remove
		void m_adaptive_non_max_sup(
					const size_t				& num_out_points,
					const vector<cv::KeyPoint>	& keypoints,
					const cv::Mat				& descriptors,
					vector<cv::KeyPoint>		& out_kp_rad,
					cv::Mat						& out_kp_desc,
					const double				& min_radius_th = 0 );

		void m_non_max_sup(
					const size_t				& num_out_points,
					const vector<cv::KeyPoint>	& keypoints,
					const cv::Mat				& descriptors,
					vector<cv::KeyPoint>		& out_kp_rad,
					cv::Mat						& out_kp_desc,
					const size_t				& imgH,
					const size_t				& imgW );

        /** Performs one iteration of a robust Gauss-Newton minimization for the visual odometry
        */
        bool m_evalRGN(
                const CStereoOdometryEstimator::TTrackingData	& tracked_feats,
                const mrpt::utils::TStereoCamera				& cam,
                const vector<double>							& deltaPose,
				const int										& cnt,				// input iteration counter
				const vector<TPoint3D>							& lmks,
                Eigen::MatrixXd									& out_newPose,
                Eigen::MatrixXd									& out_gradient,
                vector<double>									& out_residual,
                double											& out_cost,
				VOErrorCode										& out_error_code );

        void m_pinhole_stereo_projection(
                const vector<TPoint3D> &lmks,                           // [input]  the input 3D landmarks
                const TStereoCamera &cam,                               // [input]  the stereo camera
                const vector<double> &delta_pose,                       // [input]  the tested movement of the camera
                vector< pair<TPixelCoordf,TPixelCoordf> > &out_pixels,  // [output] the pixels of the landmarks in the (left & right) images
                vector<Eigen::MatrixXd> &out_jacobian);

		// Save to stream methods
		/** Saves a vector of keypoints to a strem */
		bool m_dump_keypoints_to_stream(
				std::ofstream				& stream,
				const vector<cv::KeyPoint>	& keypoints,
				const cv::Mat				& descriptors );

		/** Saves the matche sto a strem */
		bool m_dump_matches_to_stream(
				std::ofstream				& stream,
				const vector<cv::DMatch>	& matches,
				const vector<size_t>		& matches_ids );

		bool m_load_keypoints_from_stream(
				std::ifstream				& stream,
				vector<cv::KeyPoint>		& keypoints,
				cv::Mat						& descriptors );

		bool m_load_matches_from_stream(
				std::ifstream				& stream,
				vector<cv::DMatch>			& matches,
				vector<size_t>				& matches_ids );

		inline bool m_jacobian_is_good( Eigen::MatrixXd & jacobian )
		{
			for( int r = 0; r < jacobian.rows(); ++r ) {
				for( int c = 0; c < jacobian.cols(); ++c ) {
					if( mrpt::math::isNaN(jacobian(r,c)) || !mrpt::math::isFinite(jacobian(r,c)) )
						return false;
				}
			}
			return true;
		}

	//---------------------------------------------------------------
	/** @name GUI stuff
	    @{ */

		/** The thread for updating the GUI */
		void thread_gui();
		mrpt::system::TThreadHandle  m_thread_gui;
		bool  m_threads_must_close;  //!< set to true at destruction to signal all threads that they must end.

		/** ==0: means no key pressed on window; >0: is the key code, -1: the window was closed by the user.
		  * \sa getKeyPressedOnGUI(), keyPressedOnGUI()
		  */
		int   m_win_keyhit;

		struct TTrackedPixels
		{
			mrpt::utils::TPixelCoord px_pL, px_pR, px_cL, px_cR;
		};

		struct TInfoForTheGUI
		{
			mrpt::system::TTimeStamp    timestamp;                          //!< This is the key field that signals when the whole structure's changed & GUI must be refreshed
			mrpt::utils::CImage         img_left, img_right;                //!< Images to be shown on the left & right panels of the display
			std::vector<int>            stats_feats_per_octave;
			std::vector<int>            stats_FAST_thresholds_per_octave;
			std::vector<TTrackedPixels> stats_tracked_feats;
			std::string                 text_msg_from_lr_match;
			std::string                 text_msg_from_detect;
			std::string                 text_msg_from_conseq_match;
			std::string                 text_msg_from_optimization;
			CPose3D                     inc_pose;

			rso::vector_pixel_pairs_t   draw_pairings_all;
			vector<size_t>              draw_pairings_ids;

			/** Quick swap with another structure */
			void swap(TInfoForTheGUI &o)
			{
				std::swap(timestamp, o.timestamp);
				img_left.swap(o.img_left);
				img_right.swap(o.img_right);
				stats_feats_per_octave.swap(o.stats_feats_per_octave);
				stats_FAST_thresholds_per_octave.swap(o.stats_FAST_thresholds_per_octave);
				std::swap(stats_tracked_feats, o.stats_tracked_feats);
				text_msg_from_lr_match.swap(o.text_msg_from_lr_match);
				text_msg_from_detect.swap(o.text_msg_from_detect);
				text_msg_from_conseq_match.swap(o.text_msg_from_conseq_match);
				text_msg_from_optimization.swap(o.text_msg_from_optimization);
				draw_pairings_all.swap(o.draw_pairings_all);
				draw_pairings_ids.swap(o.draw_pairings_ids);
				std::swap(inc_pose, o.inc_pose);
			}
			TInfoForTheGUI();
		};

		/** Each stage may add or modify this struct, which upon complete processing of
		  * one iteration, if sent to the GUI thread via \a m_gui_info
		  */
		TInfoForTheGUI  *m_next_gui_info;
		TInfoForTheGUI   m_gui_info_cache[2]; //!< We only need two instances of this struct at once: one for writing the new data, and the "final" one sent to the GUI thread.
		int  m_gui_info_cache_next_index; //!< Will alternate between 0 and 1.

		/** This variable is set by \a processNewImagePair() */
		TInfoForTheGUI*  m_gui_info;
		mrpt::synch::CCriticalSection m_gui_info_cs;

	/** @} */
	//---------------------------------------------------------------

		/**  Stage1 operations:
		  *   - Convert to grayscale (if input is RGB)
		  *   - Rectify stereo images (if input not already rectified)
		  *   - Build pyramid
		  */
		void stage1_prepare_rectify(
			TStereoOdometryRequest & in_imgs,
			TImagePairData & out_imgpair
			);

		/**  Stage2 operations:
		  *   - Detect features on each image and on each scale.
		  */
		void stage2_detect_features(
			TImagePairData::img_data_t & img_data,
			mrpt::utils::CImage & gui_image,
			bool update_dyn_thresholds = false );

		/** Stage3 operations:
		  *   - Match features between L/R images.
		  */
		void stage3_match_left_right( CStereoOdometryEstimator::TImagePairData & imgpair, const TStereoCamera & stereoCamera );

		/** Stage4 operations:
		  *   - Track features in both L/R images between two consecutive time steps.
		  *   - Robustness checks for tracking.
		  */
		void stage4_track(
			TTrackingData &out_tracked_feats,
			TImagePairData &prev_imgpair,
			TImagePairData &cur_imgpair );

		/** Stage5 operations:
		  *   - Estimate the optimal change in pose between time steps.
		  */
        void stage5_optimize(
            TTrackingData					& out_tracked_feats,
            const mrpt::utils::TStereoCamera	& stereoCam,
            TStereoOdometryResult				& result,
			const vector<double>				& initial_estimation = vector<double>(6,0) );

	}; // end of class "CStereoOdometryEstimator"

} // end of namespace
