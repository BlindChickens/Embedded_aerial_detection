#include "mainwindow.h"

#include <string>

//extern QApplication *app;
using namespace cv;
using namespace std;

int nr;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    this->setGeometry(100,100,400,400);
    initLookup();
    srand (static_cast <unsigned> (time(0)));
//    int pos = 0;
//    int false_positive =0;
//    int false_negative =0;
//    int width = 200;
//    int height = 200;
//    int posreg = 100;
//    Mat etter = Mat::zeros(width, height, CV_8UC3);
//    Mat etter2 = Mat::zeros(width, height, CV_8UC3);
//    Mat etter3 = Mat::zeros(width, height, CV_8UC3);


    //Mat test = Mat::ones((width*(posreg-1))+posreg,2,CV_32FC1);





        start_thermal_processing = new QPushButton("Start",this);
        start_thermal_processing->setGeometry(0,0,100,100);
        connect(start_thermal_processing, SIGNAL(clicked()),this, SLOT(startThermalProcessing()));

        start_analasys = new QPushButton("Analyse",this);
        start_analasys->setGeometry(100,0,100,100);
        connect(start_analasys, SIGNAL(clicked()),this, SLOT(AnalyseFeatures()));

        reset = new QPushButton("Reset",this);
        reset->setGeometry(200,0,100,100);
        connect(reset, SIGNAL(clicked()),this, SLOT(reset_blobs()));

        load = new QPushButton("load",this);
        load->setGeometry(300,0,100,100);
        connect(load, SIGNAL(clicked()),this, SLOT(capture_video()));

        initialise = new QPushButton("Initialise training",this);
        initialise->setGeometry(0,100,100,100);
        connect(initialise, SIGNAL(clicked()),this, SLOT(start_training()));

        add_btn = new QPushButton("Add",this);
        add_btn->setGeometry(100,100,100,100);
        connect(add_btn, SIGNAL(clicked()),this, SLOT(add_sample()));

        next_btn = new QPushButton("Next",this);
        next_btn->setGeometry(200,100,100,100);
        connect(next_btn, SIGNAL(clicked()),this, SLOT(next()));

        train_btn = new QPushButton("Finish",this);
        train_btn->setGeometry(300,100,100,100);
        connect(train_btn, SIGNAL(clicked()),this, SLOT(train_svm()));


//Lees training prente om features te extract
//    for(size_t k = 0;k<20;k++){
//        //stringstream ss;
//        //ss << k;
//        //string nr = ss.str();
//        //image = imread(std::string("C:\\Users\\Jacques\\Desktop\\Rhino Test Images\\Humans.jpg"));
//        //image = imread(std::string("C:\\Users\\Jacques\\Desktop\\Rhino Test Images\\Aerial\\")+nr+".jpg");
//        features.at<float>(k,0) = 10 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(30-10)));    // eerste feature
//        features.at<float>(k,1) = 40 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(85-40)));    // tweede feature

////        Mat contourDrawing = Mat::zeros(image.rows, image.cols, CV_8UC3);
////        getContours(image,100,1);
////        delay();
////        for (size_t i = 0; i<pointsMat.size();i++){
////            drawContours( contourDrawing, pointsMat, (int)i, Scalar (0,0,255), 1, 8, vector<Vec4i>(), 0, Point() );
////        }


////        imshow("drawing4",contourDrawing);
////        waitKey(0);
//        //delay();
//    }

//    for(size_t k = 100;k<200;k++){
//        features.at<float>(k,1) = 35 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(50-35)));    // eerste feature
//        features.at<float>(k,0) = 35 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(50-35)));    // tweede feature
//        labels.at<float>(k) = -1;
//    }

//    Mat prediction = Mat::zeros(1,2,CV_32FC1);
//    prediction.at<float>(0,0) = 226;
//    prediction.at<float>(0,1) = 11;
//    Mat results = Mat(4,1,CV_32FC1);
//    for(size_t j = 0;j<100;j++){
//        params.degree += 1;
//        params.nu = 0.1;
//        for(size_t i = 0;i<8;i++){

//            //train svm op die featureset
//            svm.train(features,labels,Mat(),Mat(),params);
//            //svm.save("C:\\Users\\Jacques\\Desktop\\svm_rhino.xml");
//            svm.predict(prediction,results);
//            //qDebug()<<results.at<float>(0);
//            params.nu +=0.1;
//            if(results.at<float>(0) == 0){
//                qDebug()<<params.nu << params.degree;
//                //break;
//            }
//        }


//    }

    //svm.save("C:\\Users\\Jacques\\Desktop\\svm_rhino.xml");

    //Was om image te maak waar mens die decision boudaries kan waarneem
//    CvSVM mySVM1;
//    mySVM1.load("C:\\Users\\Jacques\\Desktop\\svm_rhino.xml");
//    Mat boundaries = Mat::zeros(800*800,2,CV_32FC1);

//    for(size_t k = 0;k<800;k++){
//        for(size_t j = 0;j<800;j++){
//            boundaries.at<float>(800*k+j,0) = k;
//            boundaries.at<float>(800*k+j,1) = j;
//        }
//    }
//    Mat resultss = Mat(800*800,1,CV_32FC1);
//    svm.predict(boundaries,resultss);
//    for(size_t k = 0;k<800;k++){
//        for(size_t j = 0;j<800;j++){
//            if(resultss.at<float>(800*k+j) == 1){
//                if(     (boundaries.at<float>(800*k+j,0)>18) && (boundaries.at<float>(800*k+j,0)<25) &&  (boundaries.at<float>(800*k+j,1)<85) && (boundaries.at<float>(800*k+j,1)>40)){
//                    qDebug() << "Is true!!";
//                }


//                circle( atom_image,  Point( boundaries.at<float>(800*k+j,0), boundaries.at<float>(800*k+j,1)),   1,  Scalar(255, 0, 0), -1, 8);
//            }
//            else{
//                circle( atom_image,  Point( boundaries.at<float>(800*k+j,0), boundaries.at<float>(800*k+j,1)),   1,  Scalar(0, 0, 255), -1, 8);
//            }
//        }
//    }



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    //net om PCA te toets

//    //klas 1
//    for(size_t k = 0;k<400;k++){
//        features.at<float>(k,0) = 10 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(30-10)));         // eerste feature
//        features.at<float>(k,1) = 400 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(410-400)));      // tweede feature
//        features.at<float>(k,2) = 25 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(35-25)));         // derde feature
//        features.at<float>(k,3) = 140 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(160-140)));      // vierde feature
//    }
//    //klas 2
//    for(size_t k = 400;k<800;k++){
//        features.at<float>(k,0) = 40 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(62-40)));         // eerste feature
//        features.at<float>(k,1) = 60 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(82-60)));         // tweede feature
//        features.at<float>(k,2) = 200 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(220-200)));      // derde feature
//        features.at<float>(k,3) = 300 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(320-300)));      // vierde feature
//    }

//    //klas 3
//    for(size_t k = 800;k<1200;k++){
//        features.at<float>(k,0) = 10 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(25-10)));         // eerste feature
//        features.at<float>(k,1) = 400 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(410-400)));      // tweede feature
//        features.at<float>(k,2) = 25 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(35-25)));         // derde feature
//        features.at<float>(k,3) = 130 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(150-130)));      // vierde feature
//    }



    //qDebug() << pca.mean.rows << pca.mean.cols << pca.mean.depth();
    //qDebug() << resnorm.rows << resnorm.cols << resnorm.type();
//    //convertScaleAbs(res,resnorm);
//    for(size_t i = 0;i<10;i++){
//        qDebug() << 4*resnorm.at<uchar>(i,0);
//    }


    //Om mean te plot - initial data moet 2d wees
//    Point pos = Point(pca.mean.at<double>(0, 0),
//                      pca.mean.at<double>(0, 1));
//    qDebug()<< "pos: "<<pca.mean.at<double>(0, 0)<<pca.mean.at<double>(0, 1);
//    MyFilledCircle(atom_image,4*pos,Scalar (0,255,0));

    //normalize(features, resnorm, 0,250,NORM_MINMAX,CV_8UC1);
    Mat feat;
    double minVal, maxVal;
    minMaxLoc(features, &minVal, &maxVal);
    //features.convertTo(feat, CV_8U);
    //qDebug() << feat.at<uchar>(50,0);


    for(size_t i = 0;i<100;i++){
        //MyFilledCircle(atom_image,Point(4* resnorm.at<uchar>(i,0), 4*resnorm.at<uchar>(i,1)),Scalar (0,0,255));
        //MyFilledCircle(atom_image,Point(10* feat.at<uchar>(i,0), 10*feat.at<uchar>(i,1)),Scalar (0,0,255));
        //qDebug() << res.at<uchar>(i,2);
    }
    for(size_t i = 100;i<200;i++){
        //MyFilledCircle(atom_image,Point(4* resnorm.at<uchar>(i,0), 4*resnorm.at<uchar>(i,1)),Scalar (0,255,0));
        //MyFilledCircle(atom_image,Point(10* feat.at<uchar>(i,0), 10*feat.at<uchar>(i,1)),Scalar (0,255,0));
        //qDebug() << res.at<uchar>(i,2);
    }
//    for(size_t i = 800;i<1200;i++){
//        MyFilledCircle(atom_image,Point( 4*resnorm.at<uchar>(i,0), 4*resnorm.at<uchar>(i,1)),Scalar (255,0,0));
//        //qDebug() << res.at<uchar>(i,2);
//    }
//    qDebug() << resnorm.rows << resnorm.cols << resnorm.depth();
    //namedWindow("drawing");
    //imshow("drawing",atom_image);


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//    //image = imread("C:\\Users\\Jacques\\Desktop\\Rhino Test Images\\Humans.jpg");
//    drawing = Mat::zeros(image.size(),CV_8UC3);
//    drawing2 = Mat::zeros(image.size(),CV_8UC3);
//    spinny = new QSpinBox(this);
//    spinny->setRange(0,300);
//    spinny->setSingleStep(10);
//    connect(spinny, SIGNAL(valueChanged(int)),this, SLOT(drawBoundingBoxess(int)));
//    if (image.empty()){
//        qDebug()<<"YesYes";
//        //QApplication::quit();
//    }




       //Circle Detection (Heavy crap)

//        Mat gimage;
//        vector<Vec3f> CirclesArray;
//        cvtColor(image,gimage,CV_BGR2GRAY);
//        HoughCircles(gimage,CirclesArray,CV_HOUGH_GRADIENT,4,300,1,100,0,500);

//        for( int i=0; i < CirclesArray.size(); i++ )
//            {
//                 Point center(cvRound(CirclesArray[i][0]), cvRound(CirclesArray[i][1]));
//                 int radius = cvRound(CirclesArray[i][2]);
//                 // draw the circle center
//                 circle( gimage, center, 3, Scalar(0,255,0), -1, 5, 0 );
//                 // draw the circle outline
//                 circle( gimage, center, radius, Scalar(0,0,255), 2, 4, 0 );
//            }
}

void MainWindow::capture_video(){
    capture_thermal_feed();
    //capture_colour_feed();
}

void MainWindow::startThermalProcessing(){

    string p_str = "C:\\Users\\Jacques\\Desktop\\Rhino Test Images\\Original Images\\";
    //string p_str = "C:\\Users\\Jacques\\Desktop\\Rhino Test Images\\Bark\\T01_";
    //string p_str = "C:\\Users\\Jacques\\Desktop\\video\\output_";


    //Mat img = imread(std::string("C:\\Users\\Jacques\\Desktop\\Rhino Test Images\\Humans.jpg"));
    Mat img = imread(p_str+"8.jpg");
    //Mat img = imread(std::string("C:\\Users\\Jacques\\Desktop\\Rhino Test Images\\Aerial\\2.jpg"));
    initialize(img);
}

void MainWindow::AnalyseFeatures(){
   initialize_colour_processing(getimage(),getblobs());
}

void MainWindow::reset_blobs(){
    resetblobs();
}

void MainWindow::start_training(){
    nr = 8;
    stringstream ss;
    ss << nr;
    string num = ss.str();
    string p_str = "C:\\Users\\Jacques\\Desktop\\Rhino Test Images\\Original Images\\";
    //string p_str = "C:\\Users\\Jacques\\Desktop\\Rhino Test Images\\Bark\\T01_";
    //string p_str = "C:\\Users\\Jacques\\Desktop\\video\\output_";
    //Mat img = imread(std::string("C:\\Users\\Jacques\\Desktop\\Rhino Test Images\\Humans.jpg"));
    Mat img = imread(p_str+num+".jpg");
    //Mat img = imread(std::string("C:\\Users\\Jacques\\Desktop\\Rhino Test Images\\Aerial\\2.jpg"));
    //initialize(img);
    initialize_training(img);
}

void MainWindow::add_sample(){
    analyse_blobs_t();
}

void MainWindow::next(){
    reset_blobs();
    nr++;
    stringstream ss;
    ss << nr;
    string num = ss.str();
    string p_str = "C:\\Users\\Jacques\\Desktop\\Rhino Test Images\\Original Images\\";
    //string p_str = "C:\\Users\\Jacques\\Desktop\\video\\output_";
    //string p_str = "C:\\Users\\Jacques\\Desktop\\Rhino Test Images\\Bark\\T01_";
    Mat img = imread(p_str+num+".jpg");
    initialize(img);
}

void MainWindow::train_svm(){
    train();
}

void MainWindow::delay()
{
    QTime dieTime = QTime::currentTime().addSecs(3);
    while( QTime::currentTime() < dieTime )
    QCoreApplication::processEvents(QEventLoop::AllEvents, 100);
}


Mat MainWindow::norm_0_255(Mat src) {
    // Create and return normalized image:
    Mat dst;
    switch(src.channels()) {
    case 1:
        normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
    }
    return dst;
}

vector<double> MainWindow::getFeaturesSet(Rect boundbox){

}


double MainWindow::getAspectRatio(double width, double height){         //bv, feature 0
    return width/height;
}

MainWindow::~MainWindow()
{

}
