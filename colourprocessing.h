#ifndef COLOURPROCESSING_H
#define COLOURPROCESSING_H

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


using namespace cv;
using namespace std;

void initialize_colour_processing(Mat image, vector<Rect> new_blobs);
void convert_to_HSV(Mat image);
void convert_to_CIELUV(Mat image);
void analyse_blobs();

int countSetBits(int code);
int rightshift(int num, int shift, int width);
bool isUniform(int code);
int getMinimal(int code);
bool isMinimal(int code);
void initLookup();
Mat get_hist(Mat lbp_mat,int size);
void show_hist();

Mat HS_feature(Mat src, Rect blob);
Mat UV_feature(Mat src, Rect blob);
void LBP(const Mat& src, Mat& dst, int radius, int neighbors);

void MyFilledCircle( Mat img, Point center, Scalar colour,int size );

void segment(Mat orgimg);

#endif // COLOURPROCESSING_H
