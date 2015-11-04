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

// computes the jacobian of a set of variables
void CStereoOdometryEstimator::m_pinhole_stereo_projection(
                const vector<TPoint3D> &lmks,                           // [input]  the input 3D landmarks
                const TStereoCamera &cam,                               // [input]  the stereo camera
                const vector<double> &delta_pose,                       // [input]  the tested movement of the camera (w1,w2,w3,t1,t2,t3)
                vector< pair<TPixelCoordf,TPixelCoordf> > &out_pixels,  // [output] the pixels of the landmarks in the (left & right) images
                vector<Eigen::MatrixXd> &out_jacobian                   // [output] the jacobians of the projections vector<matrix4x6>
                )
{
    const size_t nL = lmks.size();

    double w1,w2,w3,t1,t2,t3;                       // delta pose values
    double w12,w22,w32;
    double tt,tt2,tt3,tt4;                          // modulus of the angle
    double r00,r01,r02,r10,r11,r12,r20,r21,r22;     // rotation matrix values

    // get the pose parameters
    w1 = delta_pose[0]; w2 = delta_pose[1]; w3 = delta_pose[2];
    t1 = delta_pose[3]; t2 = delta_pose[4]; t3 = delta_pose[5];
    w12 = w1*w1; w22 = w2*w2; w32 = w3*w3;

    tt  = sqrt(w1*w1+w2*w2+w3*w3);
    tt2 = tt*tt;    tt3 = tt2*tt;   tt4 = tt3*tt;

    // precompute sin/cosvalues
    double sin_tt = sin(tt);
    double cos_tt = cos(tt);

    double dr00dw1,dr00dw2,dr00dw3,dr01dw1,dr01dw2,dr01dw3,dr02dw1,dr02dw2,dr02dw3;
    double dr10dw1,dr10dw2,dr10dw3,dr11dw1,dr11dw2,dr11dw3,dr12dw1,dr12dw2,dr12dw3;
    double dr20dw1,dr20dw2,dr20dw3,dr21dw1,dr21dw2,dr21dw3,dr22dw1,dr22dw2,dr22dw3;
    if( tt < 1e-5 )
    {
        r00 = 1;    r01 = -w3;  r02 = w2;
        r10 = w3;   r11 = 1;    r12 = -w1;
        r20 = -w2;  r21 = w1;   r22 = 1;

        // first row
        dr00dw1 = dr00dw2 = dr00dw3 = 0;

        dr01dw1 = dr01dw2 = 0;
        dr01dw3 = -1;

        dr02dw1 = dr02dw3 = 0;
        dr02dw2 = 1;

        // second row
        dr10dw1 = dr10dw2 = 0;
        dr10dw3 = 1;

        dr11dw1 = dr11dw2 = dr11dw3 = 0;

        dr12dw1 = -1;
        dr12dw2 = dr12dw3 = 0;

        // third row
        dr20dw1 = dr20dw3 = 0;
        dr20dw2 = -1;

        dr21dw1 = 1;
        dr21dw2 = dr21dw3 = 0;

        dr22dw1 = dr22dw2 = dr22dw3 = 0;
    }
    else
    {
        double u,v,dudw1,dudw2,dudw3,dvdw1,dvdw2,dvdw3;

        u = (cos_tt-1)/tt2;
        dudw1 = ((-sin_tt*w1/tt)*tt2-(cos_tt-1)*2*w1)/tt4;
        dudw2 = ((-sin_tt*w2/tt)*tt2-(cos_tt-1)*2*w2)/tt4;
        dudw3 = ((-sin_tt*w3/tt)*tt2-(cos_tt-1)*2*w3)/tt4;

        v = sin_tt/tt;
        dvdw1 = w1*(tt*cos_tt-sin_tt)/tt3;
        dvdw2 = w2*(tt*cos_tt-sin_tt)/tt3;
        dvdw3 = w3*(tt*cos_tt-sin_tt)/tt3;

        // rotation matrix and derivatives
        r00 = (w22 + w32)*u + 1;
        r01 = - w3*v - w1*w2*u;
        r02 = w2*v - w1*w3*u;

        r10 = w3*v - w1*w2*u;
        r11 = (w12 + w32)*u + 1;
        r12 = - w1*v - w2*w3*u;

        r20 = - w2*v - w1*w3*u;
        r21 = w1*v - w2*w3*u;
        r22 = (w12 + w22)*u + 1;

        // first row
        dr00dw1 = (w22 + w32)*dudw1;
        dr00dw2 = 2*w2*u+(w22+w32)*dudw2;
        dr00dw3 = 2*w3*u+(w22+w32)*dudw3;

        dr01dw1 = -w3*dvdw1-(w2*u+w1*w2*dudw1);
        dr01dw2 = -w3*dvdw2-(w1*u+w1*w2*dudw2);
        dr01dw3 = -(v+w3*dvdw3)-w1*w2*dudw3;

        dr02dw1 = w2*dvdw1-(w3*u+w1*w3*dudw1);
        dr02dw2 = (v+w2*dvdw2)-w1*w3*dudw2;
        dr02dw3 = w2*dvdw3-(w1*u+w1*w3*dudw3);

        // second row
        dr10dw1 = w3*dvdw1-(w2*u+w1*w2*dudw1);
        dr10dw2 = w3*dvdw2-(w1*u+w1*w2*dudw2);
        dr10dw3 = (v+w3*dvdw3)-w1*w2*dudw3;

        dr11dw1 = 2*w1*u+(w12+w32)*dudw1;
        dr11dw2 = (w12 + w32)*dudw2;
        dr11dw3 = 2*w3*u+(w12+w32)*dudw3;

        dr12dw1 = -(v+w1*dvdw1)-w2*w3*dudw1;
        dr12dw2 = -w1*dvdw2-(w3*u+w2*w3*dudw2);
        dr12dw3 = -w1*dvdw3-(w2*u+w2*w3*dudw3);

        // third row
        dr20dw1 = -w2*dvdw1-(w3*u+w1*w3*dudw1);
        dr20dw2 = -(v+w2*dvdw2)-w1*w3*dudw2;
        dr20dw3 = -w2*dvdw3-(w1*u+w1*w3*dudw3);

        dr21dw1 = (v+w1*dvdw1)-w2*w3*dudw1;
        dr21dw2 = w1*dvdw2-(w3*u+w2*w3*dudw2);
        dr21dw3 = w1*dvdw3-(w2*u+w2*w3*dudw3);

        dr22dw1 = 2*w1*u+(w12+w22)*dudw1;
        dr22dw2 = 2*w2*u+(w12+w22)*dudw2;
        dr22dw3 = (w22 + w32)*dudw3;
    }

    // resize output:
    out_jacobian.resize(nL);
    out_pixels.resize(nL);

    for( size_t m = 0; m < nL; ++m )
    {
        // individual jacobian
        out_jacobian[m] = Eigen::MatrixXd::Zero(4,6);

        // 'lmks[m]' is the 3D position of the landmark wrt the left camera in the previous pose
        const double X1p = lmks[m].x;
        const double Y1p = lmks[m].y;
        const double Z1p = lmks[m].z;

        // compute 3D point in current left coordinate system
        const double X1c = r00*X1p+r01*Y1p+r02*Z1p+t1;
        const double Y1c = r10*X1p+r11*Y1p+r12*Z1p+t2;
        const double Z1c = r20*X1p+r21*Y1p+r22*Z1p+t3;

        // 3D position of the landmark wrt the right coordinate system
        const double X2c = X1c-cam.rightCameraPose[0];  // baseline

        // outPoints:
        TPixelCoordf leftPixel, rightPixel;
        leftPixel.x  = cam.leftCamera.fx()*X1c/Z1c+cam.leftCamera.cx();
        leftPixel.y  = cam.leftCamera.fy()*Y1c/Z1c+cam.leftCamera.cy();

        rightPixel.x = cam.rightCamera.fx()*X2c/Z1c+cam.rightCamera.cx();
        rightPixel.y = cam.rightCamera.fy()*Y1c/Z1c+cam.rightCamera.cy();

        out_pixels[m] = make_pair(leftPixel,rightPixel);
        // cout << out_pixels[m].first << "," << out_pixels[m].second << endl;

        // the output jacobian
        // deltaPose = [w1 w2 w3 t1 t2 t3]
        double X1cd, Y1cd, Z1cd;
        for( uint8_t j = 0; j < 6; ++j)
        {
              // derivatives of 3d pt. in curr. left coordinates wrt. param j
              switch (j)
              {
                case 0: // w1
                    if(tt < 1e-5)
                    {
                        X1cd = 0;   Y1cd = -Z1p;   Z1cd = Y1p;
                    }
                    else
                    {
                        X1cd = dr00dw1*X1p+dr01dw1*Y1p+dr02dw1*Z1p;
                        Y1cd = dr10dw1*X1p+dr11dw1*Y1p+dr12dw1*Z1p;
                        Z1cd = dr20dw1*X1p+dr21dw1*Y1p+dr22dw1*Z1p;
                    }
                    break;
                case 1: // w2
                    if(tt < 1e-5)
                    {
                        X1cd = Z1p;   Y1cd = 0;   Z1cd = -X1p;
                    }
                    else
                    {
                        X1cd = dr00dw2*X1p+dr01dw2*Y1p+dr02dw2*Z1p;
                        Y1cd = dr10dw2*X1p+dr11dw2*Y1p+dr12dw2*Z1p;
                        Z1cd = dr20dw2*X1p+dr21dw2*Y1p+dr22dw2*Z1p;
                    }
                    break;
                case 2: // w3
                    if(tt < 1e-5)
                    {
                        X1cd = -Y1p;   Y1cd = X1p;   Z1cd = 0;
                    }
                    else
                    {
                        X1cd = dr00dw3*X1p+dr01dw3*Y1p+dr02dw3*Z1p;
                        Y1cd = dr10dw3*X1p+dr11dw3*Y1p+dr12dw3*Z1p;
                        Z1cd = dr20dw3*X1p+dr21dw3*Y1p+dr22dw3*Z1p;
                    }
                    break;
                case 3: // t1
                        X1cd = 1; Y1cd = 0; Z1cd = 0; break;
                case 4: // t2
                        X1cd = 0; Y1cd = 1; Z1cd = 0; break;
                case 5: // t3
                        X1cd = 0; Y1cd = 0; Z1cd = 1; break;
              } // end switch

              // set jacobian entries
              out_jacobian[m](0,j) = cam.leftCamera.fx()*(X1cd*Z1c-X1c*Z1cd)/(Z1c*Z1c);  // left u'
              out_jacobian[m](1,j) = cam.leftCamera.fy()*(Y1cd*Z1c-Y1c*Z1cd)/(Z1c*Z1c);  // left v'
              out_jacobian[m](2,j) = cam.rightCamera.fx()*(X1cd*Z1c-X2c*Z1cd)/(Z1c*Z1c); // right u'
              out_jacobian[m](3,j) = cam.rightCamera.fy()*(Y1cd*Z1c-Y1c*Z1cd)/(Z1c*Z1c); // right v'
        } // end-for j
    } // end-for m
} // end

bool CStereoOdometryEstimator::m_evalRGN(
	const TKeyPointList					& list1_l,			// input -- left image coords at time 't'
	const TKeyPointList					& list1_r,			// input -- right image coords at time 't'
	const TKeyPointList					& list2_l,			// input -- left image coords at time 't+1'
	const TKeyPointList					& list2_r,			// input -- right image coords at time 't+1'
	const vector<bool>					& mask,				// input -- vector with same size as input: true --> use point, false --> not use it
	const vector<double>				& deltaPose,        // input -- (w1,w2,w3,t1,t2,t3)
	const mrpt::utils::TStereoCamera	& stereoCam,		// input -- stereo camera parameters
    Eigen::MatrixXd						& out_newPose,		// output
    Eigen::MatrixXd						& out_gradient,
    vector<double>						& out_residual,
    double								& out_cost,
	VOErrorCode							& out_error_code )
{
	const size_t nL = list1_l.size();						// number of points to project
	const size_t n_non_masked = std::count( mask.begin(), mask.end(), true );

	ASSERTMSG_(n_non_masked>0,"Number of non masked points entering evaluation of Gauss Newton is zero!")

	// prepare output
	out_residual.resize(nL,std::numeric_limits<double>::max());						// one residual value for each landmark
    out_cost		= 0;							// the cost for this iteration
	out_gradient	= Eigen::MatrixXd::Zero(6,1);
	out_error_code	= voecNone;

    // prepare variables

    // gradient, jacobian and hessian (both individual and total)
    Eigen::MatrixXd gi(6,1);
    Eigen::MatrixXd Ji(4,6);
    Eigen::MatrixXd Hi(6,6);
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(6,6);

	// 1. Project 'list1' to 3D according to 'stereoCam'
	// 2. Back project them again to current frame according to 'deltaPose'

	// shortcut to camera parameters
	const double & cul = stereoCam.leftCamera.cx();
    const double & cvl = stereoCam.leftCamera.cy();
    const double & fl  = stereoCam.leftCamera.fx();
    const double & cur = stereoCam.rightCamera.cx();
    const double & fr  = stereoCam.rightCamera.fx();
	const double & baseline = stereoCam.rightCameraPose[0];
	
	MRPT_TODO("Optimize evaluation of Gauss-Newton")
	// 1. Project to 3D
    vector<TPoint3D> lmks(n_non_masked);
	const int octave = 0;
    for( size_t m = 0, i = 0; m < nL; ++m)
    {
		if( !mask[m] ) continue;

		const cv::KeyPoint & featL = list1_l[m];
		const cv::KeyPoint & featR = list1_r[m];

        const double ul  = featL.pt.x;
        const double vl  = featL.pt.y;
        const double ur  = featR.pt.x;
        const double b_d = baseline/(fl*(cur-ur)+fr*(ul-cul));

        lmks[i] = TPoint3D(b_d*fr*(ul-cul),b_d*fr*(vl-cvl),b_d*fl*fr);				// (X,Y,Z)
		++i;
    } // end for m

	// 2. Project back to current image
    vector< pair<TPixelCoordf,TPixelCoordf> > out_pixels;	// left and right coordinates
    vector<Eigen::MatrixXd> out_jacobian;
    m_pinhole_stereo_projection( lmks, stereoCam, deltaPose, out_pixels, out_jacobian );

	// 3. Build Gauss-Newton equation system 
	for( size_t m = 0, i = 0; m < nL; ++m)
    {
		if( !mask[m] ) continue;

		if( !m_jacobian_is_good(out_jacobian[i]) ) { ++i; continue; }

		const cv::KeyPoint & featL = list2_l[m];
		const cv::KeyPoint & featR = list2_r[m];

        // landmark projections on the image
        const TPixelCoordf & p2D_l = out_pixels[i].first;
        const TPixelCoordf & p2D_r = out_pixels[i].second;

        // get the jacobian of the 3D->2D projection
        Ji = out_jacobian[i];

        // residual
        const double r_left_x  = featL.pt.x-p2D_l.x;  // observation - prediction (left)
        const double r_left_y  = featL.pt.y-p2D_l.y;  // observation - prediction (left)
        const double r_right_x = featR.pt.x-p2D_r.x;  // observation - prediction (left)
        const double r_right_y = featR.pt.y-p2D_r.y;  // observation - prediction (left)

        // the residual vector
        Eigen::Vector4d ri; ri << r_left_x, r_left_y, r_right_x, r_right_y;

        // input value for the robust kernel function
        const double s = r_left_x*r_left_x + r_left_y*r_left_y + r_right_x*r_right_x + r_right_y*r_right_y;
        out_residual[m] = s;

		// use robust kernel
		// fi = b^2*(sqrt(1+r^2/b^2)-1)
        double rho_p    = 1;    // derivative of the robust kernel function
        double fi       = 0;    // individual cost
        if( params_least_squares.use_robust_kernel )
        {
            const double b2     = params_least_squares.kernel_param*params_least_squares.kernel_param;
            const double n      = sqrt(1+(s/b2));
            rho_p				= 1/n;         // rho derivative
            fi					= b2*(n-1);    // individual cost
        }
        else
        {
            fi = 0.5*s;         // individual cost
        }
        out_cost += fi;

        // compute the individual gradient and hessian
        gi = rho_p*(Ji.transpose()*ri);     // Wi  = eye(4); gi  = rho_p.*(Ji'*Wi*ri);
        Hi  = Ji.transpose()*Ji;            // Hi  = Ji'*pHi*Ji;

        // add it to the total gradient and hessian
        out_gradient += gi;
        H += Hi;
    } // end-for

    // 4. Solve Gauss-Newton equations
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(H,Eigen::ComputeThinU|Eigen::ComputeThinV);
    Eigen::VectorXd eValues(6);
    eValues = svd.singularValues();				// eigen values

    double condNumber = eValues(0)/eValues(5);  // H condition number
	if( mrpt::math::isNaN(condNumber) )
	{
		cout << "ERROR: Condition number is NaN" << endl;
		cout << "Hessian = [" << endl << H << "]" << endl;
		m_error = out_error_code = voecBadCondNumber;
        return false;
	} // end-if
	
	out_newPose = svd.solve(out_gradient); // solve the system H*dx = g
    return true;
} // end--m_evalRGN

// evaluate one step of a robust gauss-newton minimization
bool CStereoOdometryEstimator::m_evalRGN(
    const CStereoOdometryEstimator::TTrackingData	& tracked_feats,	// input
    const mrpt::utils::TStereoCamera				& cam,              // input
    const vector<double>							& deltaPose,        // input (w1,w2,w3,t1,t2,t3)
	const int										& cnt,				// input iteration counter
	const vector<TPoint3D>							& lmks,
    Eigen::MatrixXd									& out_newPose,
    Eigen::MatrixXd									& out_gradient,
    vector<double>									& out_residual,
    double											& out_cost,
	VOErrorCode										& out_error_code )
{
    const size_t octave = 0;	// (by now, just octave 0)

    // get the features in prev images
    // local variables
    const size_t nL = tracked_feats.tracked_pairs[octave].size();    // number of matches between (left-right)_p and (left-right)_c seen from this position

	// prepare output
    out_residual.resize(nL);										// one residual value for each landmark
    out_cost		= 0;											// the cost for this iteration
	out_gradient	= Eigen::MatrixXd::Zero(6,1);
	out_error_code	= voecNone;

    // prepare variables
    // get the prediction of the observation (uL',vL',uR',vR')
    Eigen::MatrixXd zPre( 4, nL );

    // gradient, jacobian and hessian (both individual and total)
    Eigen::MatrixXd gi(6,1);
    Eigen::MatrixXd Ji(4,6);
    Eigen::MatrixXd Hi(6,6);
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(6,6);

    // residuals r(k) = zObs(x) - zPred(x)
    Eigen::MatrixXd r_left  = Eigen::MatrixXd::Zero(nL,2);
    Eigen::MatrixXd r_right = Eigen::MatrixXd::Zero(nL,2);
    Eigen::MatrixXd r = Eigen::MatrixXd::Zero(4*nL,1);				// the vector of residuals

    /** / DEBUG
    for( size_t m = 0; m < nL; ++m)
    {
        const size_t mpreIdx = tracked_feats.tracked_pairs[octave][m].first;

        // left and right feature indexes
        const size_t lpreIdx = tracked_feats.prev_imgpair->lr_pairing_data[octave].matches_lr[mpreIdx].first;
        const size_t rpreIdx = tracked_feats.prev_imgpair->lr_pairing_data[octave].matches_lr[mpreIdx].second;

        const TSimpleFeature &featpreL = tracked_feats.prev_imgpair->left.pyr_feats[octave][lpreIdx];
        const TSimpleFeature &featpreR = tracked_feats.prev_imgpair->right.pyr_feats[octave][rpreIdx];

        cout << "FEAT PAIR 1 [" << mpreIdx << "]: " << featpreL.pt << " and " << featpreR.pt << endl;

        // left and right feature indexes
        const size_t mcurIdx = tracked_feats.tracked_pairs[octave][m].second;

        const size_t lcurIdx = tracked_feats.cur_imgpair->lr_pairing_data[octave].matches_lr[mcurIdx].first;
        const size_t rcurIdx = tracked_feats.cur_imgpair->lr_pairing_data[octave].matches_lr[mcurIdx].second;

        // observations: featL & featR
        const TSimpleFeature &featL = tracked_feats.cur_imgpair->left.pyr_feats[octave][lcurIdx];
        const TSimpleFeature &featR = tracked_feats.cur_imgpair->right.pyr_feats[octave][rcurIdx];

        cout << "FEAT PAIR 2 [" << mcurIdx << "]: " << featL.pt << " and " << featR.pt << endl;
        mrpt::system::pause();

    } // end for
    /**/
    
	// to do: filter by Z

    vector< pair<TPixelCoordf,TPixelCoordf> > out_pixels;
    vector<Eigen::MatrixXd> out_jacobian;
    // landmark projections on the image
    m_pinhole_stereo_projection( lmks, cam, deltaPose, out_pixels, out_jacobian );

	FILE *fpred  = NULL;
	if( params_general.vo_save_files )
		fpred = mrpt::system::os::fopen(mrpt::format("%s/predictions_%04d_it%d.txt",params_general.vo_out_dir.c_str(),m_it_counter,cnt).c_str(),"wt");

    for( size_t m = 0; m < nL; ++m)
    {
		if( !m_jacobian_is_good(out_jacobian[m]) )
			continue;

		// indexes of the matches in the current step
        const size_t mcurIdx = tracked_feats.tracked_pairs[octave][m].second;
        TSimpleFeature featL,featR;

        if( params_detect.detect_method == TDetectParams::dmORB || params_detect.detect_method == TDetectParams::dmFAST_ORB )
        {
            // left and right feature indexes
            const size_t lcurIdx = tracked_feats.cur_imgpair->orb_matches[mcurIdx].queryIdx;
            const size_t rcurIdx = tracked_feats.cur_imgpair->orb_matches[mcurIdx].trainIdx;

            featL.pt.x = tracked_feats.cur_imgpair->left.orb_feats[lcurIdx].pt.x;
            featL.pt.y = tracked_feats.cur_imgpair->left.orb_feats[lcurIdx].pt.y;
            featR.pt.x = tracked_feats.cur_imgpair->right.orb_feats[rcurIdx].pt.x;
            featR.pt.y = tracked_feats.cur_imgpair->right.orb_feats[rcurIdx].pt.y;
        }
        else
        {
            // left and right feature indexes
            const size_t lcurIdx = tracked_feats.cur_imgpair->lr_pairing_data[octave].matches_lr[mcurIdx].first;
            const size_t rcurIdx = tracked_feats.cur_imgpair->lr_pairing_data[octave].matches_lr[mcurIdx].second;

            featL = tracked_feats.cur_imgpair->left.pyr_feats[octave][lcurIdx];
            featR = tracked_feats.cur_imgpair->right.pyr_feats[octave][rcurIdx];
        }

        // landmark projections on the image
        TPixelCoordf & p2D_l = out_pixels[m].first;
        TPixelCoordf & p2D_r = out_pixels[m].second;

        // get the jacobian of the 3D->2D projection
        Ji = out_jacobian[m];

        // residual
        const double r_left_x  = featL.pt.x-p2D_l.x;  // observation - prediction (left)
        const double r_left_y  = featL.pt.y-p2D_l.y;  // observation - prediction (left)
        const double r_right_x = featR.pt.x-p2D_r.x;  // observation - prediction (left)
        const double r_right_y = featR.pt.y-p2D_r.y;  // observation - prediction (left)

		if( params_general.vo_save_files )
		{
			mrpt::system::os::fprintf(fpred, "%d %d %d %d %.2f %.2f %.2f %.2f\n", 
				featL.pt.x, featL.pt.y, featR.pt.x, featR.pt.y,
				p2D_l.x, p2D_l.y, p2D_r.x, p2D_r.y );
		}

        // the residual vector
        Eigen::Vector4d ri; ri << r_left_x, r_left_y, r_right_x, r_right_y;

        // input value for the robust kernel function
        const double s = r_left_x*r_left_x + r_left_y*r_left_y + r_right_x*r_right_x + r_right_y*r_right_y;
        out_residual[m] = s;

		// use robust kernel
		// fi = b^2*(sqrt(1+r^2/b^2)-1)
        double rho_p    = 1;    // derivative of the robust kernel function
        double fi       = 0;    // individual cost
		if( m_it_counter == 9 )
			params_least_squares.use_robust_kernel = false;
		else
			params_least_squares.use_robust_kernel = true;
        if( params_least_squares.use_robust_kernel )
        {
            const double b2     = params_least_squares.kernel_param*params_least_squares.kernel_param;
            const double n      = sqrt(1+(s/b2));
            rho_p   = 1/n;         // rho derivative
            fi      = b2*(n-1);    // individual cost

            // rho_pp  = -1/(2*b^2*n^3);   % rho second derivative
            // pHi     = rho_p.*Wi+2*rho_pp.*(Wi*ri)*(Wi*ri)';
            // pHi     = eye(4);
        }
        else
        {
            fi = 0.5*s;         // individual cost

            // rho_pp  = 0;
            // pHi     = eye(4);
        }
        out_cost += fi;

        // compute the individual gradient and hessian
        gi = rho_p*(Ji.transpose()*ri);     // Wi  = eye(4); gi  = rho_p.*(Ji'*Wi*ri);
        Hi  = Ji.transpose()*Ji;            // Hi  = Ji'*pHi*Ji;

        // add it to the total gradient and hessian
        out_gradient += gi;
        H += Hi;

    } // end-for

	if( params_general.vo_save_files )
		mrpt::system::os::fclose(fpred);

    // build the gauss newton equations
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(H,Eigen::ComputeThinU|Eigen::ComputeThinV);
    Eigen::VectorXd eValues(6);
    eValues = svd.singularValues();				// eigen values

    double condNumber = eValues(0)/eValues(5);  // H condition number
	if( mrpt::math::isNaN(condNumber) )
	{
		cout << "ERROR: Condition number is NaN" << endl;
		cout << "Hessian = [" << endl << H << "]" << endl;
		m_error = out_error_code = voecBadCondNumber;
        return false;
	} // end-if

	/*
	if( false && (condNumber < 1e-15 || mrpt::math::isNaN(condNumber) || !mrpt::math::isFinite(condNumber)) )
    {
        out_newPose = Eigen::MatrixXd::Zero(6,1);

		// save debug info: previous and current matches positions in images
		if( params_general.vo_debug )
		{
			FILE * fdebug = mrpt::system::os::fopen( mrpt::format("%s/debug_minimization_%d.txt",params_general.vo_out_dir.c_str(),m_it_counter).c_str(), "wt");
			for( size_t i = 0; i < nL; ++i )
			{
				const size_t mpreIdx = tracked_feats.tracked_pairs[octave][i].first;
				const size_t lpreIdx = tracked_feats.prev_imgpair->orb_matches[mpreIdx].queryIdx;
				const size_t rpreIdx = tracked_feats.prev_imgpair->orb_matches[mpreIdx].trainIdx;
				const cv::KeyPoint & pl_kp = tracked_feats.prev_imgpair->left.orb_feats[lpreIdx];
				const cv::KeyPoint & pr_kp = tracked_feats.prev_imgpair->right.orb_feats[rpreIdx];

				const size_t mcurIdx = tracked_feats.tracked_pairs[octave][i].second;
				const size_t lcurIdx = tracked_feats.cur_imgpair->orb_matches[mcurIdx].queryIdx;
				const size_t rcurIdx = tracked_feats.cur_imgpair->orb_matches[mcurIdx].trainIdx;
				const cv::KeyPoint & cl_kp = tracked_feats.cur_imgpair->left.orb_feats[lcurIdx];
				const cv::KeyPoint & cr_kp = tracked_feats.cur_imgpair->right.orb_feats[rcurIdx];

				mrpt::system::os::fprintf(fdebug, "%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n",
					pl_kp.pt.x, pl_kp.pt.y, pr_kp.pt.x, pr_kp.pt.y,
					cl_kp.pt.x, cl_kp.pt.y, cr_kp.pt.x, cr_kp.pt.y );
			}
			mrpt::system::os::fclose(fdebug);

			// save images
			m_prev_imgpair->left.pyr.images[0].saveToFile(     mrpt::format( "%s/impL_%d.jpg", params_general.vo_out_dir.c_str(), m_it_counter).c_str() );
			m_prev_imgpair->right.pyr.images[0].saveToFile(    mrpt::format( "%s/impR_%d.jpg", params_general.vo_out_dir.c_str(), m_it_counter).c_str() );
			m_current_imgpair->left.pyr.images[0].saveToFile(  mrpt::format( "%s/imcL_%d.jpg", params_general.vo_out_dir.c_str(), m_it_counter).c_str() );
			m_current_imgpair->right.pyr.images[0].saveToFile( mrpt::format( "%s/imcR_%d.jpg", params_general.vo_out_dir.c_str(), m_it_counter).c_str() );
		}

		out_error_code = voecBadCondNumber;
        return false;

    }
    else
    {*/
        out_newPose = svd.solve(out_gradient); // solve the system H*dx = g
        return true;
    //}
}

void CStereoOdometryEstimator::stage5_optimize(
	CStereoOdometryEstimator::TTrackingData			& out_tracked_feats,
	const mrpt::utils::TStereoCamera				& stereoCam,
	TStereoOdometryResult							& result,
	const vector<double>							& initial_estimation )	// [input] (w1,w2,w3,t1,t2,t3)
 {
    m_profiler.enter("_stg5");

	const size_t nOctaves = out_tracked_feats.tracked_pairs.size();

	// -- get total number of tracked matches for all the octaves
	size_t num_total_matches = 0;
	for( size_t octave = 0; octave < nOctaves; ++octave )
		num_total_matches += out_tracked_feats.tracked_pairs[octave].size();

	// -- reserve space for the keypoint lists
	TKeyPointList list1_l, list1_r, list2_l, list2_r;
	list1_l.reserve(num_total_matches);
	list1_r.reserve(num_total_matches);
	list2_l.reserve(num_total_matches);
	list2_r.reserve(num_total_matches);

	vector< pair<size_t,size_t> > octave_match_idx_vector;
	octave_match_idx_vector.reserve(num_total_matches);
		
	// -- process each octave
	for( size_t octave = 0; octave < nOctaves; ++octave )
	{
		// convert all keypoints to largest scale
		const size_t scale_norm = std::pow(2,octave);
		const size_t num_matches_this_octave = out_tracked_feats.tracked_pairs[octave].size();
		for( size_t i = 0; i < num_matches_this_octave; ++i )
		{
			// previous data
			const size_t pre_match_idx = out_tracked_feats.tracked_pairs[octave][i].first;
			const size_t pre_kp_left_idx = out_tracked_feats.prev_imgpair->lr_pairing_data[octave].matches_lr_dm[pre_match_idx].queryIdx;
			const size_t pre_kp_right_idx = out_tracked_feats.prev_imgpair->lr_pairing_data[octave].matches_lr_dm[pre_match_idx].trainIdx;

			list1_l.push_back( out_tracked_feats.prev_imgpair->left.pyr_feats_kps[octave][pre_kp_left_idx] );
			list1_r.push_back( out_tracked_feats.prev_imgpair->right.pyr_feats_kps[octave][pre_kp_right_idx] );
			
			list1_l.back().pt.x *= scale_norm; // scale normalization
			list1_l.back().pt.y *= scale_norm; // scale normalization
			list1_r.back().pt.x *= scale_norm; // scale normalization
			list1_r.back().pt.y *= scale_norm; // scale normalization

			// current data
			const size_t cur_match_idx = out_tracked_feats.tracked_pairs[octave][i].second;
			const size_t cur_kp_left_idx = out_tracked_feats.cur_imgpair->lr_pairing_data[octave].matches_lr_dm[cur_match_idx].queryIdx;
			const size_t cur_kp_right_idx = out_tracked_feats.cur_imgpair->lr_pairing_data[octave].matches_lr_dm[cur_match_idx].trainIdx;

			list2_l.push_back( out_tracked_feats.cur_imgpair->left.pyr_feats_kps[octave][cur_kp_left_idx] );
			list2_r.push_back( out_tracked_feats.cur_imgpair->right.pyr_feats_kps[octave][cur_kp_right_idx] );

			list2_l.back().pt.x *= scale_norm; // scale normalization
			list2_l.back().pt.y *= scale_norm; // scale normalization
			list2_r.back().pt.x *= scale_norm; // scale normalization
			list2_r.back().pt.y *= scale_norm; // scale normalization

			octave_match_idx_vector.push_back( make_pair(octave,i) );
		} // end-number-of-matches
	} // end-octaves

	// non-max-suppression (only left images)
	// -------------------------------------------
	vector<bool> survivors(num_total_matches);
	m_non_max_sup(	list1_l, 
					survivors,	// this will update 'survivors' vector
					out_tracked_feats.prev_imgpair->left.pyr.images[0].getHeight(), 
					out_tracked_feats.prev_imgpair->left.pyr.images[0].getWidth(), 
					num_total_matches ); 

	if( params_general.vo_save_files )
	{
		FILE *f_opt = os::fopen("optimization.txt","wt");
		for( size_t i = 0; i < survivors.size(); ++i )
		{
			if( survivors[i] )
			{
				os::fprintf(f_opt,"%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
					list1_l[i].pt.x,list1_l[i].pt.y,
					list1_r[i].pt.x,list1_r[i].pt.y,
					list2_l[i].pt.x,list2_l[i].pt.y,
					list2_r[i].pt.x,list2_r[i].pt.y );
			} // end-if
		} // end-for
		os::fclose(f_opt);
	} // end-if
	mrpt::system::pause();

    // Solve Gauss Newton
	vector<double>	out_residual;		
	double			pCost = 0,			// previous cost
					cCost = 0;			// current cost
    bool			done = false,		// flags for finishing ...
					abort = false;		// ... or aborting process
    vector<double>	deltaPose(6,0);
    Eigen::MatrixXd	out_newPose(6,1), 
					out_grad(6,1);

	if( params_least_squares.use_custom_initial_pose ) // this setting has priority
		deltaPose = initial_estimation;
	else if( params_least_squares.use_previous_pose_as_initial )
		deltaPose = m_last_computed_pose;

	//if( m_verbose_level >= 2 )
	//{
	//	cout << endl << "	Initial estimation (w1,w2,w3,t1,t2,t3) ";
	//	if( params_least_squares.use_custom_initial_pose )
	//	{
	//		cout << " (custom): " << endl;
	//		DUMP_VECTOR(initial_estimation)
	//	}
	//	else if( params_least_squares.use_previous_pose_as_initial )
	//	{
	//		cout << " (last): " << endl;
	//		DUMP_VECTOR(m_last_computed_pose)
	//	}
	//}
	
	VOErrorCode out_error_code;
    unsigned int timesInc = 0;		// number of times with incremental cost
    result.num_it = 0;				// iteration counter
	while( result.num_it < int(params_least_squares.initial_max_iters) && !done && !abort )
    {
        pCost = cCost;

        // perform one iteration of the gauss newton process
		bool cond = m_evalRGN( 
				list1_l, list1_r, list2_l, list2_r,
				survivors,				// in --> masks elements to use
                deltaPose,				// in
                stereoCam,				// in
                out_newPose,			// out	--> size 6x1
                out_grad,				// out	--> size 6x1
                out_residual,			// out	--> size listXX.size()
                cCost,					// out	--> scalar
                result.error_code );	// out	--> scalar
        
		if( m_verbose_level >= 2 ) printf( "\n	It %d -- COST [stg1]: %.10f\n", result.num_it, cCost );
        VERBOSE_LEVEL(2) << "	It " << result.num_it << " -- INCR_POSE [stg1] (w1,w2,w3,t1,t2,t3): " << out_newPose.transpose() << endl;

        if( !cond )
        {
            result.valid = false;
            return;
        }

        // update the pose
        for(uint8_t k = 0; k < deltaPose.size(); ++k)
            deltaPose[k] += out_newPose(k);

        // check ending condition
        if( result.num_it  > 0 )
        {
			/** /
            const double K   = 2*(pCost-cCost);
            Eigen::MatrixXd _aux = out_newPose.transpose()*out_grad;
            double aux = _aux(0,0);
            const double med = K/aux;
            if(med >= 0)
            {
                double m = 0;
                for(uint8_t c = 0; c < 6; ++c)
                    m += (out_newPose(c)*out_newPose(c));

                done = sqrt(m) < params_least_squares.max_error_per_obs_px;

//                cout << "SQRT: " << sqrt(m) << endl;
            }
            else
            {
                ++timesInc;
//                cout << "Function cost is not decreasing (" << timesInc << " vs " << params_least_squares.max_incr_cost << "). Aborting...";
                if( timesInc > params_least_squares.max_incr_cost )
                    abort = true;
            }
			/**/
			double m = 0;
			for(uint8_t c = 0; c < 6; ++c)	m += (out_newPose(c)*out_newPose(c));

			done = sqrt(m) < params_least_squares.max_error_per_obs_px;

			if( pCost < cCost )
			{
				if( ++timesInc > params_least_squares.max_incr_cost )
				{
					SHOW_WARNING("Function cost has increased too many times!");
					result.error_code = voecIncrFuncCostStg1;
					abort = true;
				}
			} // end-if
        }
        result.num_it ++;
    } // end while iters

    // keep only the inliers!
	for( size_t i = 0; i < out_residual.size(); ++i )
    {
		if( out_residual[i] < params_least_squares.residual_threshold )
			survivors[i] = false;
        else
		{
			const size_t octave = octave_match_idx_vector[i].first;		// octave
			const size_t t_idx = octave_match_idx_vector[i].second;		// idx
			result.outliers.push_back( out_tracked_feats.tracked_pairs[octave][t_idx].second );
		}
	} // end-for

#if 0
    CStereoOdometryEstimator::TTrackingData inliers;
    inliers.cur_imgpair     = out_tracked_feats.cur_imgpair;
    inliers.prev_imgpair    = out_tracked_feats.prev_imgpair;
    inliers.tracked_pairs.resize(1);    // one only octave!
    inliers.tracked_pairs[0].reserve(out_residual.size());
	result.outliers.reserve(out_residual.size());

    vector<TPoint3D> lmks2;
	lmks2.reserve(lmks.size());

	for( size_t obs = 0; obs < out_residual.size(); ++obs )
    {
        if( out_residual[obs] < params_least_squares.residual_threshold )
		{
			lmks2.push_back( lmks[obs] );
            inliers.tracked_pairs[0].push_back( out_tracked_feats.tracked_pairs[0][obs] );
		}
        else
		{
			const size_t idx = out_tracked_feats.tracked_pairs[0][obs].second;
			const size_t id  = params_general.vo_use_matches_ids ? 
				m_current_imgpair->lr_pairing_data[0].matches_IDs[ out_tracked_feats.tracked_pairs[0][obs].second /*current*/ ] : // m_current_imgpair->orb_matches_ID[ out_tracked_feats.tracked_pairs[0][obs].second /*current*/ ] : 
				0;
				result.outliers.push_back( make_pair(idx,id) );
		}
    }
    VERBOSE_LEVEL(2) << "	Inliers: " << inliers.tracked_pairs[0].size() << endl;
#endif
    // update the tracked IDs (remove outliers)
    // ----------------------------------------------------

    // final refinement starting at
    // ----------------------------------------------------
    // opt1: the final estimation in the previous stage
    // opt2: zero
    // opt3: the initial estimation
    done = false, abort = false;
    // for(uint8_t ii = 0; ii < 6; ++ii) deltaPose[ii] = 0;
	// deltaPose = initial_estimation;

	result.num_it_final = 0;
	vector<double> out_residual_final;
    while( result.num_it_final < int(params_least_squares.max_iters) && !done && !abort )
    {
        pCost = cCost;

        // perform one iteration of the gauss newton process
		bool cond = m_evalRGN( 
				list1_l, list1_r, list2_l, list2_r,
				survivors,			// in --> masks elements to use
                deltaPose,			// in
                stereoCam,			// in
                out_newPose,		// out	--> size 6x1
                out_grad,			// out	--> size 6x1
                out_residual,		// out	--> size listXX.size()
                cCost,				// out	--> scalar
                out_error_code );	// out	--> scalar

        if( m_verbose_level >= 2 ) printf( "\n	It %d -- COST [ref]: %.10f\n", result.num_it_final, cCost );
        VERBOSE_LEVEL(2) << "	It " << result.num_it_final << " -- INCR_POSE [ref]: " << out_newPose.transpose() << endl;

        if( !cond )
        {
			result.out_residual.swap( out_residual );
            result.valid = false;
			return;
        }

        // update the pose
        for(uint8_t k = 0; k < deltaPose.size(); ++k)
            deltaPose[k] += out_newPose(k);

        // check ending condition
        if( result.num_it_final > 0 )
        {
			/** /
            const double K   = 2*(pCost-cCost);
            Eigen::MatrixXd _aux = out_newPose.transpose()*out_grad;
            double aux = _aux(0,0);
            const double med = K/aux;
            if( med >= 0 )
            {
                double m = 0;
                for(uint8_t c = 0; c < 6; ++c)
                    m += (out_newPose(c)*out_newPose(c));

                done = sqrt(m) < params_least_squares.max_error_per_obs_px;

                // cout << "SQRT [ref]: " << sqrt(m) << endl;
            }
            else
            {
                if( pCost < cCost )
				{
					cout << "Function cost is not decreasing: ";
					cout << pCost << " vs " << cCost << endl; // " outpose: " << endl << out_newPose.transpose() << " and outGrad: " << endl << out_grad.transpose() << endl;
					if( ++timesInc > params_least_squares.max_incr_cost )
					{
						SHOW_WARNING("Function cost has increased too many times!");
						abort = true;
					}
				}
            }
			/**/

			double m = 0;
			for(uint8_t c = 0; c < 6; ++c)	m += (out_newPose(c)*out_newPose(c));

			done = sqrt(m) < params_least_squares.max_error_per_obs_px;

			if( pCost < cCost )
			{
				// cout << "Function cost is not decreasing: " << pCost << " vs " << cCost << endl;
				if( ++timesInc > params_least_squares.max_incr_cost )
				{
					SHOW_WARNING("Function cost has increased too many times!");
					abort = true;
					result.error_code = voecIncrFuncCostStg2;
				}
			} // end-if
        }
        result.num_it_final++;
    } // end while iters

	// save debug files
	if( params_general.vo_save_files )
	{
		FILE *fresidual = os::fopen(mrpt::format("%s/out_residual_%04d.txt",params_general.vo_out_dir.c_str(),m_it_counter).c_str(),"wt");
		for( size_t k = 0; k < out_residual.size(); ++k )
			mrpt::system::os::fprintf(fresidual,"%.3f\n", out_residual[k]);
		mrpt::system::os::fclose(fresidual);
		fresidual = os::fopen(mrpt::format("%s/out_residual_final_%04d.txt",params_general.vo_out_dir.c_str(),m_it_counter).c_str(),"wt");
		for( size_t k = 0; k < out_residual_final.size(); ++k )
			mrpt::system::os::fprintf(fresidual,"%.3f\n", out_residual_final[k]);
		mrpt::system::os::fclose(fresidual);
	}

    // at this point, deltaPose has the inverse of the change in pose between time steps
    // deltaPose = [w1,w2,w3,t1,t2,t3]
    CPose3DRotVec rvt(deltaPose[0],deltaPose[1],deltaPose[2],deltaPose[3],deltaPose[4],deltaPose[5]);
	result.outPose = CPose3D(rvt.getInverse()); // this is the pose of the current stereo frame wrt the previous one

    if( !params_least_squares.use_custom_initial_pose && params_least_squares.use_previous_pose_as_initial )
	{
		m_last_computed_pose = deltaPose;
		cout << "	Saving 'm_last_computed_pose': ";
		DUMP_VECTOR(m_last_computed_pose);
	}
	VERBOSE_LEVEL(2) << "   :: OUTPOSE: " << result.outPose << endl;

    // set output result
    result.tracked_feats_from_last_frame	= m_num_tracked_pairs_from_last_frame;
	result.out_residual.swap( out_residual );
    result.valid = !abort;

    m_profiler.leave("_stg5");

    m_next_gui_info->inc_pose = result.outPose;
    m_next_gui_info->text_msg_from_optimization = mrpt::format(
            "\nIncr. pose = (%.2f,%.2f,%.2f,%.1fd,%.1fd,%.1fd)",
            result.outPose.x(), result.outPose.y(), result.outPose.z(),
            RAD2DEG(result.outPose.yaw()), RAD2DEG(result.outPose.pitch()), RAD2DEG(result.outPose.roll()) );
} // end stage5_optimization
