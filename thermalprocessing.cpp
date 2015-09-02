#include "thermalprocessing.h"

using namespace cv;
using namespace std;

RNG rng = 12345;
Mat OriginalImage;
Mat normImage;
Mat grayImage;
Mat uvimg;
Mat segmentedimg;
Mat grimage;
Mat binaryImage;
Mat contourImage;
vector<vector<Point> > pointsMat;
vector<Rect> blobs_1;
int threshold_value = 0;
int threshold_type = 1;











void initialize(Mat image){
    OriginalImage = image.clone();  //Stoor net die Original
    //blobimg = OriginalImage.clone();
    namedWindow("ROISelection");    //Display window vir original
    //setMouseCallback("ROISelection",onMouse);   //Set callback function vir mouse event
    //imshow("ROISelection",OriginalImage);   //Wys die original om mee te begin
    cvtColor(OriginalImage,grimage,CV_BGR2GRAY);
    //blobimg =grimage.clone();
    //imshow("ROISelection",grimage);
    //namedWindow("HSV_IMG",CV_WINDOW_AUTOSIZE);  //Display windows vir HSV beeld

    namedWindow("drawing",CV_WINDOW_AUTOSIZE);
    namedWindow("drawing2",CV_WINDOW_AUTOSIZE);
    //namedWindow("drawing4",CV_WINDOW_AUTOSIZE);

    segmentImage(); // segment die thermal image op grond van die heat signatures

    contourImage = Mat::zeros(OriginalImage.rows, OriginalImage.cols, CV_8UC3);
    //getContours(segmentedimg,0);
    findContours(segmentedimg,pointsMat,0,CHAIN_APPROX_SIMPLE,Point(0,0));
    for (size_t i = 0; i<pointsMat.size();i++){
        drawContours( contourImage, pointsMat, (int)i, Scalar (0,0,255), 1, 8, vector<Vec4i>(), 0, Point() );
    }
    drawBoundingBoxes(contourImage);
    imshow("drawing2",contourImage);
    initialize_colour_processing(OriginalImage,blobs_1);
//    drawBoundingBoxes(binaryImage);

}





void segmentImage(){
    vector<Mat> LUVChannels;
    cvtColor(OriginalImage, uvimg, CV_BGR2Luv);
    //split(uvimg,LUVChannels);

    inRange(uvimg,Scalar(0,96,134),Scalar(255,110,148),segmentedimg);

    //threshold(LUVChannels[1],LUVChannels[1],110,0,threshold_type);
    //threshold(LUVChannels[2],LUVChannels[2],160,0,threshold_type);

    //threshold(LUVChannels[1],LUVChannels[1],95,0,threshold_type);
    //threshold(LUVChannels[2],LUVChannels[2],160,0,threshold_type);

    //merge(LUVChannels,uvimg);
    //cvtColor(uvimg, segmentedimg, CV_Luv2BGR);
    imshow("drawing",segmentedimg);
    //imshow("drawing3",binaryImage);
    //imshow("drawing4",OriginalImage);
}

vector<vector<Point> > getContours(Mat image, int ret_mode){
    //normImage = norm_0_255(image);
    //cvtColor(image,grayImage,CV_BGR2GRAY);
    //threshold(grayImage,binaryImage,thresh,255,threshold_type);
    //Mat binaryimage2 = binaryImage.clone();

//    namedWindow("drawing2");
//    imshow("drawing2",binaryimage2);

    findContours(image,pointsMat,ret_mode,CHAIN_APPROX_SIMPLE,Point(0,0));
    return pointsMat;
}

vector<Rect> getBoundingBoxes(vector<vector<Point> > contourPoints){

    //vector<vector<Point> > pointsMatPoly(pointsMat.size());
    vector<Rect> boundRect;
    Rect recta;
    //qDebug()<< "voor: " <<boundRect.size();
    for( size_t i = 0; i < contourPoints.size(); i++ )
    {
        //approxPolyDP( pointsMat[i], pointsMatPoly[i], 3, true );
        //boundRect[i] = boundingRect( Mat(contourPoints[i]) );
        recta = boundingRect(Mat(contourPoints[i]));
        if( (recta.width*recta.height > 3000) && (recta.width>30) && (recta.height > 30) && (recta.width*recta.height < 24000) ){
            boundRect.push_back(recta);
        }

        //qDebug()<< "voor: " <<boundRect.size();

        //minEnclosingCircle( contours_poly[i], center[i], radius[i] );
    }
    qDebug()<< "na: " << boundRect.size();
    return boundRect;
}

//void MainWindow::drawContours(Scalar color, Mat image){
//    for (size_t i = 0; i<pointsMat.size();i++){
//        drawContours( image, pointsMat, (int)i, color, 1, 8, vector<Vec4i>(), 0, Point() );
//    }
//}


void drawBoundingBoxes(Mat image){
    vector<Rect> boundrecct = getBoundingBoxes(pointsMat);
    blobs_1 = boundrecct;
    //drawing = Mat::zeros(image.size(),CV_8UC3);
    //drawing2 = Mat::zeros(image.size(),CV_8UC3);
    for( size_t i = 0; i< boundrecct.size(); i++ )
    {

        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        rectangle( image, boundrecct[i].tl(), boundrecct[i].br(), color, 2, 8, 0 );
        //circle( drawing, center[i], (int)radius[i], color, 2, 8, 0 );

    }

    //was om nog 'n stel contours te kry met ander ret_mode
//    boundrecct = getBoundingBoxes(image,thresh,1);
//    qDebug()<< boundrecct.size();
//    for( size_t i = 0; i< boundrecct.size(); i++ )
//    {
//        if(boundrecct[i].width*boundrecct[i].height>500){
//            Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
//            drawContours( drawing2, pointsMat, (int)i, color, 1, 8, vector<Vec4i>(), 0, Point() );
//            rectangle( drawing2, boundrecct[i].tl(), boundrecct[i].br(), color, 2, 8, 0 );
//            //circle( drawing, center[i], (int)radius[i], color, 2, 8, 0 );
//        };

//    }
}
