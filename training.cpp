#include "training.h"

using namespace cv;
using namespace std;

vector<Rect> blobs_t;
vector<int> lookup_t;
Mat HSfeatures_t;
Mat UVfeatures_t;
Mat TFeatures_t;
Mat ColourImage_t;
Mat hsvImg_t;
Mat luvimg_t;
Mat gimage_t;
Mat features_t;
Mat labels;
Mat lbpres_t;
Mat blobimg;
Mat trainingimg;
Mat img;
vector<Rect> blobbies;
Rect cropRect(0,0,0,0);
Point P1(0,0);
Point P2(0,0);

bool clicked=false;

CvSVM svm;
CvSVMParams params;

void resetblobs(){
    blobimg = trainingimg.clone();
    imshow("ROISelection",trainingimg);
    blobbies.clear();
}
Mat getimage(){
    //return OriginalImage;
}

vector<Rect> getblobs(){
    //return blobbies;
}

void onMouse( int event, int x, int y, int f, void* ){
    switch(event){

        case  CV_EVENT_LBUTTONDOWN  :
                                        clicked=true;

                                        P1.x=x;
                                        P1.y=y;
                                        P2.x=x;
                                        P2.y=y;
                                        break;

        case  CV_EVENT_LBUTTONUP    :
                                        P2.x=x;
                                        P2.y=y;
                                        clicked=false;
                                        blobbies.push_back(cropRect);
                                        //qDebug()<<cropRect.width<<cropRect.height;
                                        blobimg = img.clone();
                                        break;

        case  CV_EVENT_MOUSEMOVE    :
                                        if(clicked){
                                        P2.x=x;
                                        P2.y=y;
                                        }
                                        break;

        default                     :   break;


    }
    if(clicked){
     if(P1.x>P2.x){ cropRect.x=P2.x;
                       cropRect.width=P1.x-P2.x; }
        else {         cropRect.x=P1.x;
                       cropRect.width=P2.x-P1.x; }

        if(P1.y>P2.y){ cropRect.y=P2.y;
                       cropRect.height=P1.y-P2.y; }
        else {         cropRect.y=P1.y;
                       cropRect.height=P2.y-P1.y; }


        img = blobimg.clone();
        rectangle(img, cropRect, Scalar(255,0,0), 1, 8, 0);
        imshow("ROISelection",img);
    }



}


void initialize_training(Mat image){
    /*TYPE*/

    //  (100) CvSVM::C_SVC
    //C-Support Vector Classification. n-class classification (n \geq 2), allows imperfect separation of classes with penalty multiplier C for outliers.

    //  (101) CvSVM::NU_SVC
    //\nu-Support Vector Classification. n-class classification with possible imperfect separation. Parameter \nu (in the range 0..1, the larger the value, the smoother the decision boundary) is used instead of C.

    //  (102) CvSVM::ONE_CLASS
    //Distribution Estimation (One-class SVM). All the training data are from the same class, SVM builds a boundary that separates the class from the rest of the feature space.

    /*KERNEL TYPE*/

    //  (0) CvSVM::LINEAR Linear kernel. No mapping is done, linear discrimination (or regression) is done in the original feature space. It is the fastest option. K(x_i, x_j) = x_i^T x_j.
    //  (1) CvSVM::POLY Polynomial kernel: K(x_i, x_j) = (\gamma x_i^T x_j + coef0)^{degree}, \gamma > 0.
    //  (2) CvSVM::RBF Radial basis function (RBF), a good choice in most cases. K(x_i, x_j) = e^{-\gamma ||x_i - x_j||^2}, \gamma > 0.
    //  (3) CvSVM::SIGMOID Sigmoid kernel: K(x_i, x_j) = \tanh(\gamma x_i^T x_j + coef0).

    /*OTHER*/

    //degree – Parameter degree of a kernel function (POLY).
    //gamma – Parameter \gamma of a kernel function (POLY / RBF / SIGMOID).
    //coef0 – Parameter coef0 of a kernel function (POLY / SIGMOID).

    //Cvalue – Parameter C of a SVM optimization problem (C_SVC / EPS_SVR / NU_SVR).
    //nu – Parameter \nu of a SVM optimization problem (NU_SVC / ONE_CLASS / NU_SVR).
    //p – Parameter \epsilon of a SVM optimization problem (EPS_SVR).
    //class_weights – Optional weights in the C_SVC problem , assigned to particular classes. They are multiplied by C so the parameter C of class #i becomes class\_weights_i * C. Thus these weights affect the misclassification penalty for different classes. The larger weight, the larger penalty on misclassification of data from the corresponding class.
    //term_crit – Termination criteria of the iterative SVM training procedure which solves a partial case of constrained quadratic optimization problem. You can specify tolerance and/or the maximum number of iterations.

    // void setParams(int svm_type, int kernel_type, double degree, double gamma, double coef0, double CValue, double nu, double p, CvTermCriteria term_crit);
    setParams(             102,             2,             1,           0.00001,          1,            0.1,        0.0001,      0,     TermCriteria(CV_TERMCRIT_ITER, (int)1e7, 1e-6));





    initLookup_t();
    features_t = Mat::zeros(0,10,CV_32FC1);
    //qDebug()<< features_t.rows<<features_t.depth()<<features_t.type();
    labels = Mat::ones(10,1,CV_32FC1);                                         //Create die class identification vector 100x1

    //analyse_blobs_t();

    trainingimg = image.clone();  //Stoor net die Original
    blobimg = trainingimg.clone();
    namedWindow("ROISelection");    //Display window vir original
    setMouseCallback("ROISelection",onMouse);   //Set callback function vir mouse event

}

void train(){
    svm.train(features_t,labels,Mat(),Mat(),params);
    svm.save("C:\\Users\\Jacques\\Desktop\\svm_rhino.xml");
}


void analyse_blobs_t(){
    ColourImage_t = trainingimg.clone();  //Stoor net die Original
    cvtColor(trainingimg,gimage_t,CV_BGR2GRAY);
    cvtColor(trainingimg, hsvImg_t, CV_BGR2HSV);
    cvtColor(trainingimg, luvimg_t, CV_BGR2Luv);

    blobs_t = blobbies;

    for(size_t b = 0;b<blobs_t.size();b++){
        Rect lbp_roi(blobs_t[b].x+blobs_t[b].width/2-10,blobs_t[b].y+blobs_t[b].height/2-10,20,20);
        LBP_t(gimage_t(lbp_roi),lbpres_t,1,8);
        //Kry die features
        HSfeatures_t = HS_feature(hsvImg_t,lbp_roi);
        UVfeatures_t = UV_feature(luvimg_t,lbp_roi);
        qDebug()<< UVfeatures_t.at<float>(0) << UVfeatures_t.at<float>(1);
        TFeatures_t = get_hist(lbpres_t,10);
        TFeatures_t.convertTo(TFeatures_t,CV_32FC1);
        //qDebug()<< TFeatures.rows << TFeatures.cols;

        Mat s_f = Mat(1,10,CV_32FC1);

        hconcat(TFeatures_t, HSfeatures_t, s_f);

        //qDebug()<<TFeatures_t.at<float>(0)<<TFeatures_t.at<float>(1)<<TFeatures_t.at<float>(2)<<TFeatures_t.at<float>(3)<<TFeatures_t.at<float>(4)<<TFeatures_t.at<float>(5)<<TFeatures_t.at<float>(6)<<TFeatures_t.at<float>(7)<<TFeatures_t.at<float>(8)<<TFeatures_t.at<float>(9)<<HSfeatures_t.at<float>(0)<<HSfeatures_t.at<float>(1);
        //qDebug()<<s_f.at<float>(0)<<s_f.at<float>(1)<<s_f.at<float>(2)<<s_f.at<float>(3)<<s_f.at<float>(4)<<s_f.at<float>(5)<<s_f.at<float>(6)<<s_f.at<float>(7)<<s_f.at<float>(8)<<s_f.at<float>(9)<<s_f.at<float>(10)<<s_f.at<float>(11);

        //qDebug()<<s_f.rows<<s_f.cols;
        //Laai die features in feature vektor
        //TFeatures_t.row(0).copyTo(features_t.row(0));


        //features_t.push_back(s_f.row(0));
        features_t.push_back(TFeatures_t.row(0));


        //qDebug()<<"Y1";
        //features_t.at<float>(b,10) = HSfeatures_t.at<float>(0);    //HUE
        //features_t.at<float>(b,11) = HSfeatures_t.at<float>(1);    //Saturation
        //qDebug()<<HSfeatures[0]<<HSfeatures[1]<<HSfeatures[2];

        qDebug()<< features_t.rows<<features_t.cols;
        //features.at<float>(b,1) = Tfeatures[1];
    }

}


void LBP_t(const Mat& src, Mat& dst, int radius, int neighbors) {
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
        for(int i=radius; i < src.rows-radius;i++){
            for(int j=radius;j < src.cols-radius;j++){
                float t = w1*src.at<uchar>(i+fy,j+fx) + w2*src.at<uchar>(i+fy,j+cx) + w3*src.at<uchar>(i+cy,j+fx) + w4*src.at<uchar>(i+cy,j+cx);
                // we are dealing with floating point precision, so add some little tolerance
                dst.at<unsigned int>(i-radius,j-radius) += ((t > src.at<uchar>(i,j)) && (abs(t-src.at<uchar>(i,j)) > std::numeric_limits<float>::epsilon())) << n;

                if(n==neighbors-1){
                    //qDebug()<<"LBP: " <<dst.at<unsigned int>(i-radius,j-radius);
                    dst.at<unsigned int>(i-radius,j-radius) = lookup_t[dst.at<unsigned int>(i-radius,j-radius)];
                }
            }
        }
    }
}


void initLookup_t()
{
    lookup_t.resize(255);
    int index=0;
    for(int i=0;i<256;i++)
    {
        //check if minimal rotatian
        if (isMinimal(i) == false){
          lookup_t[i] = lookup_t[getMinimal(i)];
          //qDebug()<<isMinimal(i);
        }
        else{
            bool status=isUniform(i);
            if(status==true)
            {
                lookup_t[i]=index;
                index++;
            }
            else
            {
                lookup_t[i]=9;
            }
        }
    }

    //initHistogram();
}

void setParams(int svm_type, int kernel_type, double degree, double gamma, double coef0, double CValue, double nu, double p, CvTermCriteria term_crit){
    params.svm_type = svm_type;
    params.kernel_type = kernel_type;
    params.degree = degree;
    params.gamma = gamma;
    params.coef0 = coef0;
    params.nu = nu;
    params.term_crit = term_crit;
}
