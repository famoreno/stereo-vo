[![Build Status](https://travis-ci.org/famoreno/stereo-vo.svg?branch=master)](https://travis-ci.org/famoreno/stereo-vo)

# stereo-vo
Robust Stereo Visual Odometry

**Note:** *Preliminary version*

A C++ library for stereo visual odometry.

## Documentation
* [Doxygen API reference](http://famoreno.github.io/stereo-vo/)
* References:
  * Moreno, F.A. and Blanco, J.L. and Gonzalez, J. **A constant-time SLAM back-end in the continuum between global mapping and submapping: application to visual stereo SLAM**, International Journal of Robotics Research, 2016. (In Press)
  *  Moreno, F.A. **Stereo Visual SLAM for Mobile Robots Navigation**, PhD Thesis, 2015. ([PDF](http://mapir.isa.uma.es/famoreno/papers/thesis/FAMD_thesis.pdf))

## Building from sources

### Prerequisites

* CMake
* [MRPT](https://github.com/MRPT/mrpt) (>=1.3.0 ?)
* OpenCV (>=2.4.0 ?)

They can be installed in Debian or Ubuntu with:

    sudo apt-get install build-essential cmake libmrpt-dev libopencv-dev

**Note:** Better efficiency can be achieved if `MRPT` and `OpenCV` are compiled from sources instead of grabbed with `apt-get` by instructing the compiler to optimize for native architecture.

### Compiling

...

