#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>

using namespace std;
using namespace cv;

/**
 */
static bool readDetectorParameters(string filename, Ptr<aruco::DetectorParameters> &params) {
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
        return false;
    fs["adaptiveThreshWinSizeMin"] >> params->adaptiveThreshWinSizeMin;
    fs["adaptiveThreshWinSizeMax"] >> params->adaptiveThreshWinSizeMax;
    fs["adaptiveThreshWinSizeStep"] >> params->adaptiveThreshWinSizeStep;
    fs["adaptiveThreshConstant"] >> params->adaptiveThreshConstant;
    fs["minMarkerPerimeterRate"] >> params->minMarkerPerimeterRate;
    fs["maxMarkerPerimeterRate"] >> params->maxMarkerPerimeterRate;
    fs["polygonalApproxAccuracyRate"] >> params->polygonalApproxAccuracyRate;
    fs["minCornerDistanceRate"] >> params->minCornerDistanceRate;
    fs["minDistanceToBorder"] >> params->minDistanceToBorder;
    fs["minMarkerDistanceRate"] >> params->minMarkerDistanceRate;
    fs["cornerRefinementMethod"] >> params->cornerRefinementMethod;
    fs["cornerRefinementWinSize"] >> params->cornerRefinementWinSize;
    fs["cornerRefinementMaxIterations"] >> params->cornerRefinementMaxIterations;
    fs["cornerRefinementMinAccuracy"] >> params->cornerRefinementMinAccuracy;
    fs["markerBorderBits"] >> params->markerBorderBits;
    fs["perspectiveRemovePixelPerCell"] >> params->perspectiveRemovePixelPerCell;
    fs["perspectiveRemoveIgnoredMarginPerCell"] >> params->perspectiveRemoveIgnoredMarginPerCell;
    fs["maxErroneousBitsInBorderRate"] >> params->maxErroneousBitsInBorderRate;
    fs["minOtsuStdDev"] >> params->minOtsuStdDev;
    fs["errorCorrectionRate"] >> params->errorCorrectionRate;
    return true;
}

int main(int argc, char *argv[]) {

    int dictionaryId = 16;
    bool showRejected = false;
    bool estimatePose = true;
    float markerLength = 200;//////////////////////////////////////////////////////////

    string detector_params = "/home/luohao/aruco/detect_aruco_marker/build/detector_params.yml";
    Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();
    bool readOk = readDetectorParameters(detector_params, detectorParams);
    if(!readOk) {
        cerr << "Invalid detector parameters file" << endl;
        return 0;
    }

    //override cornerRefinementMethod read from config file
    detectorParams->cornerRefinementMethod = 0;
    std::cout << "Corner refinement method (0: None, 1: Subpixel, 2:contour, 3: AprilTag 2): " << detectorParams->cornerRefinementMethod << std::endl;
    
    int camId = 1;// my rgb_cam
    
    Ptr<aruco::Dictionary> dictionary =
        aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

    Mat camMatrix = (Mat_<double>(3,3) << 761.1770, 0, 319.9148, 0, 760.7583,239.9870, 0, 0, 1);
    Mat distCoeffs = (Mat_<double>(1,5) << 0.0208, -0.0336, 0.0, 0.0, 0.0);
    // cout << "camMatrix = " << endl << " " << camMatrix << endl << endl;
    // cout << "distCoeffs = " << endl << " " << distCoeffs << endl << endl;

    VideoCapture inputVideo;
    int waitTime;

    inputVideo.open(camId);
    waitTime = 10;
    
    double totalTime = 0;
    int totalIterations = 0;

    while(inputVideo.grab()) {
        Mat image, imageCopy;
        inputVideo.retrieve(image);

        double tick = (double)getTickCount();

        vector< int > ids;
        vector< vector< Point2f > > corners, rejected;
        vector< Vec3d > rvecs, tvecs;

        // detect markers and estimate pose
        aruco::detectMarkers(image, dictionary, corners, ids, detectorParams, rejected);
        if(estimatePose && ids.size() > 0)
        {
            aruco::estimatePoseSingleMarkers(corners, markerLength, camMatrix, distCoeffs, rvecs,
                                             tvecs);
            double currentTime = ((double)getTickCount() - tick) / getTickFrequency();
            totalTime += currentTime;
            totalIterations++;
            if(totalIterations % 30 == 0) {
                cout << "Detection Time = " << currentTime * 1000 << " ms "
                    << "(Mean = " << 1000 * totalTime / double(totalIterations) << " ms)" << endl;
                cout <<  tvecs[0]  << endl;
            }
            
        }
      
        // draw results
        image.copyTo(imageCopy);
        if(ids.size() > 0) {
            aruco::drawDetectedMarkers(imageCopy, corners, ids);

            if(estimatePose) {
                for(unsigned int i = 0; i < ids.size(); i++)
                    aruco::drawAxis(imageCopy, camMatrix, distCoeffs, rvecs[i], tvecs[i],
                                    markerLength * 0.5f);
            }
        }

        if(abs(tvecs[0][0]) < 300)
        {
            cout <<"save..."<< endl;
             imwrite("/home/luohao/aruco/detect_aruco_marker/build/1.jpg",image);
        }
       
        if(showRejected && rejected.size() > 0)
            aruco::drawDetectedMarkers(imageCopy, rejected, noArray(), Scalar(100, 0, 255));

        namedWindow( "out", WINDOW_AUTOSIZE  ); // Create a window for display.
        imshow("out", imageCopy);
        char key = (char)waitKey(waitTime);
        if(key == 27) break;
    }

    return 0;
}
      
