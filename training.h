#ifndef TRAINING_H
#define TRAINING_H

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

void setParams(int svm_type, int kernel_type, double degree, double gamma, double coef0, double CValue, double nu, double p, CvTermCriteria term_crit);

void analyse_blobs_t();
void train();
void initialize_training(Mat image);
void LBP_t(const Mat& src, Mat& dst, int radius, int neighbors);
void initLookup_t();

#endif // TRAINING_H
