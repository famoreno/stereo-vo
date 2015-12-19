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
#include <mrpt/hwdrivers/CCameraSensor.h>
#include <mrpt/utils/CConfigFile.h>
#include <mrpt/utils/CConfigFileMemory.h>

#include <mrpt/otherlibs/tclap/CmdLine.h>

using namespace std;
using namespace rso;
using namespace mrpt::system;

#include <mrpt/version.h>
#if MRPT_VERSION>=0x130
using mrpt::obs::CObservationPtr;
#else
using mrpt::slam::CObservationPtr;
#endif


int main(int argc, char**argv)
{
	try
	{
		// USAGE EXAMPLE: 
		// [LIVE]		demo-stereo-odometry --sensor stereo-camera.ini --cfg-section-name section --opt demo-stereo-odometry-config.ini
		// [RAWLOG]		demo-stereo-odometry --input dataset.rawlog --opt demo-stereo-odometry-config.ini
		// [IMAGE DIR]	demo-stereo-odometry --img_dir img-dir-cfg.ini --cam stereo-camera.ini --opt demo-stereo-odometry-config.ini

		// Declare the supported options.
		TCLAP::CmdLine cmd("demo-stereo-odometry", ' ', mrpt::system::MRPT_getVersion().c_str());

		// for rawlog input
		TCLAP::ValueArg<std::string> arg_rawlog_file(
			"i","input",
			"Input dataset to load",false,"","dataset.rawlog",cmd);

		// for live sensor
		TCLAP::ValueArg<std::string> arg_sensor_cfg_file(
			"","sensor",
			"Configuration file for stereo camera sensor to open.\n"
			"This file must be in rawlog-grabber format and the sensor section name "
			"will be [CAMERA] unless specified otherwise with --cfg-section-name",
			false,"","stereo-camera.ini",cmd);
		
		TCLAP::ValueArg<std::string> arg_section_name(
			"","cfg-section-name",
			"The section name to load camera parameters from. See --sensor.",false,"","CAMERA",cmd);

		// for image directory input
		TCLAP::ValueArg<std::string> arg_img_dir_cfg_file(
			"d","img_dir",
			"Image directory configuration",false,"","img_dir_cfg.ini",cmd);

		TCLAP::ValueArg<std::string> arg_cam_cfg_file(
			"c","cam",
			"Configuration file for the camera.\n",
			false,"","camera.ini",cmd);

		// application parameters
		TCLAP::ValueArg<std::string> arg_app_cfg_file(
			"o","opt",
			"Configuration file for the application.\n",
			false,"","test.ini",cmd);

		TCLAP::SwitchArg arg_pause("p","pause","Pause with each frame",cmd, false);

		// Parse arguments:
		if (!cmd.parse( argc, argv ))
			return 1;

		if( (!arg_rawlog_file.isSet() && !arg_sensor_cfg_file.isSet() && !arg_img_dir_cfg_file.isSet() ) ||
			(arg_rawlog_file.isSet() && arg_sensor_cfg_file.isSet() ) || 
			(arg_rawlog_file.isSet() && arg_img_dir_cfg_file.isSet() ) || 
			(arg_img_dir_cfg_file.isSet() && arg_sensor_cfg_file.isSet() ) )
		{
			cerr <<
				"Error: Expected exactly one argument of either '--input', '--sensor' or '--img_dir'\n"
				"Use --help to see the complete usage documentation.\n";
			return 1;
		}

		// Parse params:
		bool pause_each_frame = arg_pause.getValue();

		// ---------------------------------------------------
		// select source of images:
		mrpt::hwdrivers::CCameraSensor myCam;  // The generic image source

		const bool run_live_from_sensor = arg_sensor_cfg_file.isSet();
		const bool run_from_rawlog		= arg_rawlog_file.isSet();

		if (run_live_from_sensor)
		{
			// Run with live sensor:
			// MRPT_TODO("In mrpt::hwdrivers stereo camera classes: allow loading the full rectification parameters from the .ini file")
			const string sCfgFile = arg_sensor_cfg_file.getValue();
			const string sSection = arg_section_name.getValue();
			myCam.loadConfig(mrpt::utils::CConfigFile(sCfgFile),sSection);

			cout << "Running from live camera..." << endl;
		}
		else
		if (run_from_rawlog)
		{
			// Run from a virtual stereo camera sensor parsing a dataset:
			const string sDatasetFile = arg_rawlog_file.getValue();

			const string str = string(
				"[CONFIG]\n"
				"grabber_type=rawlog\n"
				"capture_grayscale=false\n"
				"rawlog_file=") + sDatasetFile +
				string("\n");

			myCam.loadConfig(mrpt::utils::CConfigFileMemory(str),"CONFIG");

			cout << "Running on rawlog file... " << sDatasetFile << endl;
		}
		else
		{
			// Run from a set of images in a directory
			myCam.loadConfig( mrpt::utils::CConfigFile(arg_img_dir_cfg_file.getValue()), "IMG_SOURCE" );

			cout << "Running from image directory... " << endl;
		}

		// Try to start grabbing images: (will raise an exception on any error)
		myCam.initialize();

		// -----------------------------------------------
		// Declare Stereo Odometry object:
		CStereoOdometryEstimator stereo_odom_engine;

		std::vector<std::string> paramSections;
		paramSections.push_back("RECTIFY");
		paramSections.push_back("DETECT");
		paramSections.push_back("MATCH");
		paramSections.push_back("IF-MATCH");
		paramSections.push_back("LEAST_SQUARES");
		paramSections.push_back("GUI");
		paramSections.push_back("GENERAL");

		// Get .ini file
		CConfigFile app_iniFile( arg_app_cfg_file.getValue() );

		stereo_odom_engine.loadParamsFromConfigFileName( arg_app_cfg_file.getValue(), paramSections );
		stereo_odom_engine.setVerbosityLevel( app_iniFile.read_int("GENERAL","vo_verbosity",0,false) );
		
		// read pose of the camera on the robot
		vector<double> v_pose(6,0.0);
		app_iniFile.read_vector( "GENERAL", "camera_pose_on_robot", v_pose, v_pose, false);

		if( stereo_odom_engine.params_general.vo_debug || stereo_odom_engine.params_general.vo_save_files )
			createDirectory( stereo_odom_engine.params_general.vo_out_dir );

		CPose3D pose;
		// CPose3D poseOnRobot(0,0,0,DEG2RAD(-90),DEG2RAD(0),DEG2RAD(-100) );	// read this from app config file
		CPose3D poseOnRobot(v_pose[0],v_pose[1],v_pose[2],DEG2RAD(v_pose[3]),DEG2RAD(v_pose[4]),DEG2RAD(v_pose[5]));

		FILE *f = mrpt::system::os::fopen( mrpt::format("%s/camera_pose.txt", stereo_odom_engine.params_general.vo_out_dir.c_str()).c_str(),"wt");

		CStereoOdometryEstimator::TStereoOdometryRequest odom_request;
		if( run_live_from_sensor )
		{
			CConfigFile camera_cfgFile( arg_sensor_cfg_file.getValue() );
			odom_request.stereo_cam.loadFromConfigFile( "CAMERA", camera_cfgFile );
		}
		else
		if( run_from_rawlog )
		{
			CObservationPtr obs = myCam.getNextFrame();
			if( obs.present() )
				CObservationStereoImagesPtr(obs)->getStereoCameraParams( odom_request.stereo_cam );
			else
				THROW_EXCEPTION("No observations in rawlog");

			// reset the camera
			myCam.initialize();
		}
		else
		{
			CConfigFile camera_cfgFile( arg_cam_cfg_file.getValue() );
			odom_request.stereo_cam.loadFromConfigFile( "CAMERA", camera_cfgFile );
		}

		CObservationPtr obs;
		bool end = false; // Signal for closing if the user command so.
        unsigned int count = 0;
		while ( !end && (obs=myCam.getNextFrame()).present() )
		{
			cout << "Frame: " << ++count << endl;

			CStereoOdometryEstimator::TStereoOdometryResult  odom_result;

			// We need the observation type to be stereo images: (if it's the wrong type, an exception will be raised)
			odom_request.stereo_imgs = CObservationStereoImagesPtr(obs);

			// Estimate visual odometry:
			stereo_odom_engine.processNewImagePair( odom_request, odom_result );

			// Compute the current position
			if( odom_result.valid )
			{
				//CPose3D pri_pose(pose);

				//// composition
				//CPose3D aux;
				//aux.composeFrom(poseOnRobot,odom_result.outPose);
				//aux.inverseComposeFrom(aux,poseOnRobot);
				//pose.composeFrom(pose,aux);
				//cout << "CPose3D: " << pose << endl;

				// matrix version
				CMatrixDouble44 mat01,mat02,mat03,mat04,mat05;
				pose.getHomogeneousMatrix(mat01);					// pose
				poseOnRobot.getHomogeneousMatrix(mat02);			// k
				odom_result.outPose.getHomogeneousMatrix(mat03);	// deltaPose

				mat04 = mat02*mat03*mat02.inverse();
				mat05.multiply(mat01,mat04);
				pose = CPose3D(mat05);
				// cout << "Matrix: " << pose << endl;
			}
			else
			{
			    DUMP_VO_ERROR_CODE( odom_result.error_code )
			}

            // save the pose
            mrpt::system::os::fprintf(f,"%.3f %.3f %.3f %.3f %.3f %.3f\n",
                                      pose.m_coords[0],pose.m_coords[1],pose.m_coords[2],
                                      pose.yaw(),pose.pitch(),pose.roll());

			// GUI is updated automatically. Keep iterating.
			if (pause_each_frame) {
				while (!stereo_odom_engine.keyPressedOnGUI()) {
					mrpt::system::sleep(5);
				}
			}

			// Process different key commands:
			if (stereo_odom_engine.keyPressedOnGUI())
			{
				const int key = stereo_odom_engine.getKeyPressedOnGUI();
				switch (key)
				{
				// Close the window, press ESC or 'q' -> quit:
				case -1:
				case 27:
				case 'Q':
				case 'q':
					end=true;
					break;

				// pause/resume with 'p' or spacebar:
				case 'p':
				case 'P':
				case ' ':
					pause_each_frame= !pause_each_frame;
					break;

				}
			} // end process keys
			cout << "Frame: " << count << " finished." << endl;
			cout << "-----------------------------------------" << endl << endl;
		} // end main while() loop.

        mrpt::system::os::fclose(f);

		return 0;
	}
	catch (std::exception &e)
	{
		cerr << "**EXCEPTION**:\n" << e.what() << endl;
		return 1;
	}
}

