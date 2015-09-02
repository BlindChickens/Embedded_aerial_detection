#ifndef THERMALPROCESSING_H
#define THERMALPROCESSING_H

#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv/cv.h"
#include "opencv2/ml/ml.hpp"
#include "QDebug"
#include <cstdlib>
#include <stdlib.h>
#include <stdio.h>
#include <ctime>
#include <windows.h>

#include "colourprocessing.h"

using namespace cv;
using namespace std;

void test(int getal);

void onMouse( int event, int x, int y, int f, void* );
void initialize(Mat image);
Mat getimage();
vector<Rect> getblobs();
void resetblobs();
void segmentImage();


vector<vector<Point> > getContours(Mat image, int ret_mode);
vector<Rect> getBoundingBoxes(vector<vector<Point> > contourPoints);
//void drawContours(Scalar color, Mat image);
void drawBoundingBoxes(Mat image);


#endif // THERMALPROCESSING_H
