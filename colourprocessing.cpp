#include "colourprocessing.h"

using namespace cv;
using namespace std;

CvSVM mySVM1;
vector<Rect> blobs;
vector<int> lookup;
Mat ColourImage;
Mat hsvImg;
Mat luvImg;
Mat gimage;

Scalar HSVmean;
Scalar BGRmean;

Mat UVfeatures;
Mat HSfeatures;
Mat TFeatures;
Mat features;
Mat results;
Mat lbpres;
Mat histogram;
Mat hist_canvas;
Mat atom_images = Mat::zeros( 800, 800, CV_8UC3 );

void initialize_colour_processing(Mat image, vector<Rect> new_blobs){
    mySVM1.load("C:\\Users\\Jacques\\Desktop\\svm_rhino.xml");
    ColourImage = image.clone();  //Stoor net die Original

    cvtColor(image,gimage,CV_BGR2GRAY);
    blobs = new_blobs;
    features = Mat::zeros(0,10,CV_32FC1);
    results = Mat(blobs.size(),1,CV_32FC1);

    convert_to_HSV(ColourImage);
    convert_to_CIELUV(ColourImage);

    analyse_blobs();
    //histogram = get_hist(lbpres,10);
    //namedWindow("Histogram of LBP's");
    //show_hist();
}


void convert_to_HSV(Mat image){
    cvtColor(image, hsvImg, CV_BGR2HSV);
    //imshow("HSV_IMG",hsvImg);
}

void convert_to_CIELUV(Mat image){
    cvtColor(image, luvImg, CV_BGR2Luv);
}

void analyse_blobs(){



    for(size_t b = 0;b<blobs.size();b++){

        Rect lbp_roi(blobs[b].x+blobs[b].width/2-10,blobs[b].y+blobs[b].height/2-10,20,20);
        LBP(gimage(lbp_roi),lbpres,1,8);
        //Kry die features

        //qDebug()<< hsvImg.type()<<luvImg.type();
        HSfeatures = HS_feature(hsvImg,lbp_roi);
        UVfeatures = UV_feature(luvImg,lbp_roi);

        double dst = sqrt(   pow(UVfeatures.at<float>(0)-97,2)+ pow(UVfeatures.at<float>(1)-146,2)    );
        qDebug() << dst;
        TFeatures = get_hist(lbpres,10);
        TFeatures.convertTo(TFeatures,CV_32FC1);

        qDebug()<< HSfeatures.at<float>(0) << HSfeatures.at<float>(1);
        qDebug()<< UVfeatures.at<float>(0) << UVfeatures.at<float>(1);

        Mat s_f = Mat(1,12,CV_32FC1);

        hconcat(TFeatures, HSfeatures, s_f);

        //Laai die features in feature vektor
        //features.push_back(s_f.row(0));
        features.push_back(TFeatures.row(0));

        //features.at<float>(b,10) = HSfeatures.at<float>(0);    //HUE
        //features.at<float>(b,11) = HSfeatures.at<float>(1);    //Saturation
        //qDebug()<<HSfeatures[0]<<HSfeatures[1]<<HSfeatures[2];

        //features.at<float>(b,1) = Tfeatures[1];
        //qDebug()<<features.at<float>(b,0)<<features.at<float>(b,1)<<features.at<float>(b,2)<<features.at<float>(b,3)<<features.at<float>(b,4);
        //qDebug()<<features.at<float>(b,5)<<features.at<float>(b,6)<<features.at<float>(b,7)<<features.at<float>(b,8)<<features.at<float>(b,9);
    }





    mySVM1.predict(features,results);
    for(size_t b = 0;b<blobs.size();b++){
        if(results.at<float>(b)==1){
            rectangle(ColourImage, blobs[b], Scalar(0,255,0), 1, 8, 0);

        }
        else{
            rectangle(ColourImage, blobs[b], Scalar(0,0,255), 1, 8, 0);
        }
    }
    imshow("ROISelection",ColourImage);

    qDebug()<<"Klaar gepredict";


//    //Teken support vectors
//    int c     = mySVM1.get_support_vector_count();
//    Mat sv = Mat(c,12,CV_32FC1);
//    for (int i = 0; i < c; ++i)
//    {
//        const float* v = mySVM1.get_support_vector(i);
//        for(int f=0;f<12;f++){
//            sv.at<float>(i,f) = v[f];
//        }
//        features.push_back(sv.row(i));
//    }


//PCA pca(features, Mat(), CV_PCA_DATA_AS_ROW,2);
//Mat res;
//pca.project(features,res);
//Mat resnorm;
//normalize(res, resnorm, 0,100,NORM_MINMAX,CV_8UC1);
//resnorm = resnorm/255;
//resnorm.convertTo(thh,CV_8UC1);


//namedWindow("avg");
//imshow("avg", pca.mean.reshape(1, 0));
//namedWindow("drawing");
//imshow("drawing",resnorm);

//for(size_t i = 0;i<5;i++){
//    MyFilledCircle(atom_images,Point(8*resnorm.at<uchar>(i,0),8*resnorm.at<uchar>(i,1)),Scalar (0,0,255),5);
//    //MyFilledCircle(atom_image,Point(10* feat.at<uchar>(i,0), 10*feat.at<uchar>(i,1)),Scalar (0,0,255));
//    //qDebug() << res.at<uchar>(i,2);
//}

//for(size_t s = 5;s<5+c;s++){
//    MyFilledCircle(atom_images,Point(8*resnorm.at<uchar>(s,0),8*resnorm.at<uchar>(s,1)),Scalar (255,0,0),5);
//    //MyFilledCircle(atom_image,Point(10* feat.at<uchar>(i,0), 10*feat.at<uchar>(i,1)),Scalar (0,0,255));
//    //qDebug() << res.at<uchar>(i,2);
//}
//imshow("drawing",atom_images);
}

void MyFilledCircle( Mat img, Point center, Scalar colour,int size )
{
 int thickness = -1;
 int lineType = 8;

 circle( img,
         center,
         size,
         colour,
         thickness,
         lineType );
}

int countSetBits(int code)
{
  int count=0;
  int v=code;
  for(count=0;v;count++)
  {
  v&=v-1; //clears the LSB
  }
  return count;
}

//circular
int rightshift(int num, int shift,int width)
{
    return (num >> shift) | ((num << (width - shift)&static_cast<int>(pow(2,width)-1)));
}


bool isUniform(int code)
{
    int b = rightshift(code,1,8);
  ///int d = code << 1;
  int c = code ^ b;
  //d= code ^d;
  int count=countSetBits(c);
  //int count1=countSetBits(d);
  if (count <=2 )
      return true;
  else
      return false;
}

bool isMinimal(int code){
    int new_code = rightshift(code,1,8);;
    for (int k=0;k<8;k++){
        if(new_code < code){
            return false;
        }
        new_code = rightshift(new_code,1,8);

    }
    return true;
}

int getMinimal(int code){
    int min = code;
    for (int k=0;k<8;k++){
        code = rightshift(code,1,8);
        if(code<min){
            min=code;
        }
    }
    return min;
}

void initLookup()
{
    lookup.resize(255);
    int index=0;
    for(int i=0;i<256;i++)
    {
        //check if minimal rotatian
        if (isMinimal(i) == false){
          lookup[i] = lookup[getMinimal(i)];
          //qDebug()<<isMinimal(i);
        }
        else{
            bool status=isUniform(i);
            if(status==true)
            {
                lookup[i]=index;
                index++;
            }
            else
            {
                lookup[i]=9;
            }
        }
        qDebug()<<i<<lookup[i];
    }

    //initHistogram();
}

Mat HS_feature(Mat src, Rect blob){

    Mat HSVmean = Mat(1,2,CV_32FC1);
    HSVmean.at<float>(0) = mean(src(blob))[0];
    HSVmean.at<float>(1) = mean(src(blob))[1];
    //vector<float> HS_features;
    //HS_features[0] = HSVmean[0];
    //HS_features[1] = HSVmean[1];

    return HSVmean;
}

Mat UV_feature(Mat src, Rect blob){

    Mat LUVmean = Mat(1,2,CV_32FC1);
    LUVmean.at<float>(0) = mean(src(blob))[1];
    LUVmean.at<float>(1) = mean(src(blob))[2];
    //vector<float> HS_features;
    //HS_features[0] = HSVmean[0];
    //HS_features[1] = HSVmean[1];

    return LUVmean;
}

void LBP(const Mat& src, Mat& dst, int radius, int neighbors) {
    neighbors = max(min(neighbors,31),1); // set bounds...
    // Note: alternatively you can switch to the new OpenCV Mat_
    // type system to define an unsigned int matrix... I am probably
    // mistaken here, but I didn't see an unsigned int representation
    // in OpenCV's classic typesystem...
    dst = Mat::zeros(src.rows-2*radius, src.cols-2*radius, CV_32SC1);
    for(int n=0; n<neighbors; n++) {
        // sample points
        float x = static_cast<float>(radius) * cos(2.0*M_PI*n/static_cast<float>(neighbors));
        float y = static_cast<float>(radius) * -sin(2.0*M_PI*n/static_cast<float>(neighbors));
        // relative indices
        int fx = static_cast<int>(floor(x));
        int fy = static_cast<int>(floor(y));
        int cx = static_cast<int>(ceil(x));
        int cy = static_cast<int>(ceil(y));
        // fractional part
        float ty = y - fy;
        float tx = x - fx;
        // set interpolation weights
        float w1 = (1 - tx) * (1 - ty);
        float w2 =      tx  * (1 - ty);
        float w3 = (1 - tx) *      ty;
        float w4 =      tx  *      ty;
        // iterate through your data
        for(int i=radius; i < src.rows-radius;i++) {
            for(int j=radius;j < src.cols-radius;j++) {
                float t = w1*src.at<uchar>(i+fy,j+fx) + w2*src.at<uchar>(i+fy,j+cx) + w3*src.at<uchar>(i+cy,j+fx) + w4*src.at<uchar>(i+cy,j+cx);
                // we are dealing with floating point precision, so add some little tolerance
                dst.at<unsigned int>(i-radius,j-radius) += ((t > src.at<uchar>(i,j)) && (abs(t-src.at<uchar>(i,j)) > std::numeric_limits<float>::epsilon())) << n;

                if(n==neighbors-1){
                    //qDebug()<<"LBP: " <<dst.at<unsigned int>(i-radius,j-radius);
                    dst.at<unsigned int>(i-radius,j-radius) = lookup[dst.at<unsigned int>(i-radius,j-radius)];
                }
            }
        }
    }

}

Mat get_hist(Mat lbp_mat, int size){
    int val;
    Mat hist = Mat::zeros(1,size,CV_32SC1);
    for (int i = 0; i < lbp_mat.rows; i++)
    {
        for (int j = 0; j < lbp_mat.cols; j++)
        {
            val = lbp_mat.at<unsigned int>(i,j);
            hist.at<int>(val) += 1;
        }
    }
    return hist;
}

void show_hist(){
    hist_canvas = Mat::ones(400, 500, CV_8UC3);
    for (int j = 0; j < 10; j++)
    {
        //qDebug()<< "histogram value:"<< j <<histogram.at<int>(j);
        line(
            hist_canvas,
            Point(5+j*50, 400),
            Point(5+j*50, 400 - histogram.at<int>(j)),
            Scalar(0,0,255),
            10, 8, 0
        );
    }
    imshow("Histogram of LBP's",hist_canvas);
}
\
void segment(Mat orgimg){

}
