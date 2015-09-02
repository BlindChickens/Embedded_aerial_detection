#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QApplication>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv/cv.h"
#include "opencv2/ml/ml.hpp"
#include "QDebug"
#include "QObject"
#include "QWidget"
#include "QSpinBox"
#include <cstdlib>
#include <stdlib.h>
#include <stdio.h>
#include <ctime>
#include <QTime>
#include <windows.h>
#include <QPushButton>

//#include "colourprocessing.h"
//#include "thermalprocessing.h"
#include "feature_extraction.h"
#include "training.h"
#include "videocapture.h"

using namespace cv;
using namespace std;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:


    MainWindow(QWidget *parent = 0);
    Mat drawing;
    Mat drawing2;
    Mat features;
    Mat labels;


    QPushButton *start_thermal_processing;
    QPushButton *start_analasys;
    QPushButton *reset;
    QPushButton *load;

    QPushButton *initialise;
    QPushButton *add_btn;
    QPushButton *next_btn;
    QPushButton *train_btn;

    char* trackbar_type = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";
    char* trackbar_value = "Value";
    //static void Threshold_Demo( int, void* );

    vector<double> getFeaturesSet(Rect boundbox);
    double getAspectRatio(double width, double height);

    void delay();





    void MyFilledCircle(Mat img, Point center , Scalar colour, int size);

    Mat norm_0_255(Mat src);


    ~MainWindow();

public slots:
    void capture_video();
    void startThermalProcessing();
    void AnalyseFeatures();
    void reset_blobs();
    void start_training();
    void add_sample();
    void next();
    void train_svm();

};

#endif // MAINWINDOW_H
