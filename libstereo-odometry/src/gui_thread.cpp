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

#include <mrpt/opengl.h>
#include <mrpt/utils/CObserver.h>

using namespace rso;
using namespace std;
using namespace mrpt::opengl;
using namespace mrpt::utils;

// GUI params
CStereoOdometryEstimator::TGUIParams::TGUIParams() :
	show_gui(true),
	draw_all_raw_feats(false),
	draw_lr_pairings(false),
	draw_tracking(true)
{
}
// Info sent to the GUI for displaying:
CStereoOdometryEstimator::TInfoForTheGUI::TInfoForTheGUI() :
	timestamp(INVALID_TIMESTAMP),
	img_left(UNINITIALIZED_IMAGE),
	img_right(UNINITIALIZED_IMAGE)
{
}

// Auxiliary struct for subscribing to events of the GUI and
// immediately send them to the main object.
struct TObserverWindowEvents : public mrpt::utils::CObserver
{
	int &m_trg_key_code;

	TObserverWindowEvents(int &trg_key_code) : m_trg_key_code(trg_key_code)
	{
	}

	void OnEvent(const mrptEvent &e)
	{
		if (e.isOfType<mrpt::gui::mrptEventWindowChar>())
		{
			const mrpt::gui::mrptEventWindowChar *ev = e.getAs<mrpt::gui::mrptEventWindowChar>();
			m_trg_key_code = ev->char_code;
		}
		else if (e.isOfType<mrpt::gui::mrptEventWindowClosed>())
		{
//			mrpt::gui::mrptEventWindowClosed *ev = e.getAsNonConst<mrpt::gui::mrptEventWindowClosed>();
			//ev->allow_close = false;
			m_trg_key_code = -1; // -1 means the window has been closed by user.
		}
	}
};

// The thread in charge of creating and updating the GUI windows.
void CStereoOdometryEstimator::thread_gui()
{
	VERBOSE_LEVEL(1) << "[CStereoOdometryEstimator::thread_gui] Thread alive.\n";

	mrpt::gui::CDisplayWindow3DPtr  win;
	CPose3D current_pose(0,0,0,DEG2RAD(-90),DEG2RAD(0),DEG2RAD(-100) );

	try
	{
		// Initialize a 3D view with 2 panels for L & R images:
		win = mrpt::gui::CDisplayWindow3D::Create("Stereo Odometry",1200,465);

		// Subscribe to key events:
		TObserverWindowEvents my_observer( m_win_keyhit );
		my_observer.observeBegin(*win);

		vector<COpenGLViewportPtr>  gl_views(3);
		{
			COpenGLScenePtr &theScene = win->get3DSceneAndLock();
			gl_views[0] = theScene->getViewport("main");
			ASSERT_(gl_views[0])
			gl_views[1] = theScene->createViewport("right_image");
			ASSERT_(gl_views[1])
			gl_views[2] = theScene->createViewport("pose3D");

			// Assign sizes:
			gl_views[0]->setViewportPosition(  0, 0, .33,1.);
			gl_views[1]->setViewportPosition(.33, 0, .33,1.);
			gl_views[2]->setViewportPosition(.66, 0, .33,1.);

			// Prepare 3D view
			CSetOfObjectsPtr cam = stock_objects::BumblebeeCamera();
			cam->setPose(CPose3D());
			cam->setName("bumblebee");

			CSetOfLinesPtr path = CSetOfLines::Create();
			path->setName("path");
			path->setColor(0,1,0);
			path->appendLine(0,0,0,0,0,0);

			gl_views[2]->insert( stock_objects::CornerXYZ() );
			gl_views[2]->insert( CGridPlaneXY::Create(-100,100,-100,100) );
			gl_views[2]->insert( cam );
			gl_views[2]->insert( path );

			theScene->enableFollowCamera(true);

			// IMPORTANT!!! IF NOT UNLOCKED, THE WINDOW WILL NOT BE UPDATED!
			win->unlockAccess3DScene();
		}

		mrpt::system::TTimeStamp  last_timestamp = INVALID_TIMESTAMP;

		// Main loop in this GUI thread
		while (!m_threads_must_close)
		{
			// Check if we have new stuff to show:
			m_gui_info_cs.enter();

			if (m_gui_info->timestamp==last_timestamp || m_gui_info->timestamp==INVALID_TIMESTAMP)
			{	// Not a new or valid structure, keep waiting.
				m_gui_info_cs.leave();
				mrpt::system::sleep(2);
				continue;
			}
			last_timestamp = m_gui_info->timestamp;

			// We have new stuff:
			// Firstly, make a quick copy of the data to release the critical section ASAP and without
			// letting any chance of raising an exception in the way:
			TInfoForTheGUI  info;
			info.swap(*m_gui_info);
			m_gui_info_cs.leave();
			// From now on, we can safely work on our copy "info"
			//----------------------------------------------------

            // Set the image to color mode
            // ----------------------------------------------------
			info.img_left.colorImageInPlace();
			info.img_right.colorImageInPlace();

			// Update camera position
			current_pose += info.inc_pose;

			// Draw optional stuff on the base images:
			// ----------------------------------------------------
			m_profiler.enter("gui.draw_lr_pairings");
			if (!info.draw_pairings_all.empty())
			{
				const size_t nFeats = info.draw_pairings_all.size();

				static TColor colors[8] = {
					TColor(0xFF,0x00,0x00),
					TColor(0xFF,0xFF,0x00),
					TColor(0x00,0x00,0xFF),
					TColor(0x00,0xFF,0xFF),
					TColor(0x00,0x00,0x00),
					TColor(0xFF,0xFF,0xFF),
					TColor(0xFF,0x00,0xFF),
					TColor(0xFF,0x80,0x80)
				};

				const int rect_w = 2;
				for(size_t i=0;i<nFeats;++i)
				{
					// (X,Y) coordinates in the left/right images for this pairing:
					const TPixelCoord &ptL = info.draw_pairings_all[i].first;
					const TPixelCoord &ptR = info.draw_pairings_all[i].second;

					TColor thisColor;
					if( params_detect.detect_method == TDetectParams::dmORB )
                        thisColor = colors[info.draw_pairings_ids[i]&0x07];
                    else
                        thisColor = colors[i&0x07];

					info.img_left.rectangle( ptL.x-rect_w, ptL.y-rect_w, ptL.x+rect_w, ptL.y+rect_w, thisColor );
					info.img_right.rectangle( ptR.x-rect_w, ptR.y-rect_w, ptR.x+rect_w, ptR.y+rect_w, thisColor );
				}
			}
			m_profiler.leave("gui.draw_lr_pairings");

			// Draw tracked feats as lines in L/R
			m_profiler.enter("gui.draw_tracking");
			if (!info.stats_tracked_feats.empty())
			{
				static TColor colors[8] = {
					TColor(0xFF,0x00,0x00),
					TColor(0xFF,0xFF,0x00),
					TColor(0x00,0x00,0xFF),
					TColor(0x00,0xFF,0xFF),
					TColor(0x00,0x00,0x00),
					TColor(0xFF,0xFF,0xFF),
					TColor(0xFF,0x00,0xFF),
					TColor(0xFF,0x80,0x80)
				};

				const size_t N = info.stats_tracked_feats.size();
				for (size_t i=0;i<N;i++)
				{
					info.img_left.line(
						info.stats_tracked_feats[i].px_pL.x,
						info.stats_tracked_feats[i].px_pL.y,
						info.stats_tracked_feats[i].px_cL.x,
						info.stats_tracked_feats[i].px_cL.y,
						colors[i&0x07] );

					info.img_right.line(
						info.stats_tracked_feats[i].px_pR.x,
						info.stats_tracked_feats[i].px_pR.y,
						info.stats_tracked_feats[i].px_cR.x,
						info.stats_tracked_feats[i].px_cR.y,
						colors[i&0x07] );
				}
			}
			m_profiler.leave("gui.draw_tracking");

			// ------------------------------------------------------------
			// LOCK 3D scene & update images & text labels:
			m_profiler.enter("gui.locked-update");
			win->get3DSceneAndLock();

			gl_views[0]->setImageView_fast(info.img_left);   // "_fast()" has "move semantics" and destroy origin objects
			gl_views[1]->setImageView_fast(info.img_right);
            CRenderizablePtr obj = gl_views[2]->getByName("bumblebee");
            if(obj)
            {
                CSetOfObjectsPtr bb = static_cast<CSetOfObjectsPtr>(obj);
                bb->setPose( current_pose ); // Update camera position:

                CCamera &theCam = gl_views[2]->getCamera();
                theCam.setPointingAt( bb->getPoseX(), bb->getPoseY(), bb->getPoseZ() );
            }
			
			obj = gl_views[2]->getByName("path");
            if(obj)
            {
				CSetOfLinesPtr p = static_cast<CSetOfLinesPtr>(obj);
				p->appendLineStrip( current_pose.x(), current_pose.y(), current_pose.z() ); // Update camera path
            }

			// Text:
			win->addTextMessage(
					5,5,
					mrpt::system::dateTimeLocalToString(info.timestamp),
					TColorf(1,1,1),"mono",10, mrpt::opengl::FILL, 1
					);

			// Text:
			/** /
			{
				string sStatsFeats = string("Raw FAST features per octave: ");
				string sThresholds;
				for (size_t i=0;i<info.stats_feats_per_octave.size();i++) {
					sStatsFeats += mrpt::format("%u/",static_cast<unsigned int>(info.stats_feats_per_octave[i]) );
					sThresholds += mrpt::format("%i/",static_cast<int>(info.stats_FAST_thresholds_per_octave[i]) );
				}
				sStatsFeats += string(" | Thresholds: ") + sThresholds;

				win->addTextMessage(
						5,5+1*15,
						sStatsFeats,
						TColorf(1,1,1),"mono",10, mrpt::opengl::FILL, 2
						);
			}
			/ * */
			// Text:
			win->addTextMessage(
					5,5+1*15,
					"Detected features: " + info.text_msg_from_detect + "\n",
					TColorf(1,1,1),"mono",10, mrpt::opengl::FILL, 2
					);
			// Text:
			win->addTextMessage(
					5,5+2*15,
					info.text_msg_from_lr_match,
					TColorf(1,1,1),"mono",10, mrpt::opengl::FILL, 3
					);
			// Text:
			win->addTextMessage(
					5,5+3*15,
					info.text_msg_from_conseq_match,
					TColorf(1,1,1),"mono",10, mrpt::opengl::FILL, 4
					);

            // Text:
			win->addTextMessage(
					5,5+5*15,
					mrpt::format("Current pose (x,y,z,yaw,pitch,roll) = (%.2f,%.2f,%.2f,%.1fd,%.1fd,%.1fd)",
                        current_pose.x(), current_pose.y(), current_pose.z(),
                        RAD2DEG(current_pose.yaw()), RAD2DEG(current_pose.pitch()), RAD2DEG(current_pose.roll()))
                        + info.text_msg_from_optimization,
					TColorf(1,1,1),"mono",10, mrpt::opengl::FILL, 5
					);

			win->unlockAccess3DScene();
			m_profiler.leave("gui.locked-update");
			win->repaint();
		} // end while() main loop
	}
	catch(std::exception &e)
	{
		cerr << "[CStereoOdometryEstimator::thread_gui] Thread exit for exception:\n" << e.what() << endl;
	}

	if (win) {
		win.clear();
		mrpt::system::sleep(20); // Leave time to the wxWidgets thread to clean up
	}
	VERBOSE_LEVEL(1) << "[CStereoOdometryEstimator::thread_gui] Thread closed.\n";
}


bool CStereoOdometryEstimator::keyPressedOnGUI()
{
	return m_win_keyhit!=0;
}

int CStereoOdometryEstimator::getKeyPressedOnGUI()
{
	int r=m_win_keyhit;
	m_win_keyhit=0;
	return r;
}
