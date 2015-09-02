#include "videocapture.h"

using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
    capture_thermal_feed();
}

void capture_thermal_feed(){

    // Open the input video files or camera stream.
    VideoCapture capture1("p1.mp4");
    VideoCapture capture2("p1.mp4");

    Mat frame_1,frame_2;
    namedWindow("frame_1");
    namedWindow("frame_2");


    double rate = capture1.get(CV_CAP_PROP_FPS);
    int delay = 1000/rate;

    while(true)
    {
        if(!capture1.read(frame_1) || !capture2.read(frame_2)){
            break;
        }
        imshow("frame_1",frame_1);
        imshow("frame_2",frame_2);
        initialize(frame_1);
        if(waitKey(delay)>=0)
            break;
    }
    capture1.release();
    capture2.release();


}
