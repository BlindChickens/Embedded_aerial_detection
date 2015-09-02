#ifndef VIDEOCAPTURE_H
#define VIDEOCAPTURE_H


#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv/cv.h"
#include "thermalprocessing.h"

using namespace cv;
using namespace std;

void capture_thermal_feed();
void capture_colour_feed();

#endif // VIDEOCAPTURE_H
