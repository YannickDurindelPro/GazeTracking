#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/face/facemark.hpp>

#include <iostream>
#include <queue>
#include <stdio.h>
#include <math.h>

#include "constants.h"
#include "findEyeCenter.h"
#include "findEyeCorner.h"

/* Attempt at supporting openCV version 4.0.1 or higher */
#if CV_MAJOR_VERSION >= 4
#define CV_WINDOW_NORMAL                WINDOW_NORMAL
#define CV_BGR2YCrCb                    COLOR_BGR2YCrCb
#define CV_HAAR_SCALE_IMAGE             CASCADE_SCALE_IMAGE
#define CV_HAAR_FIND_BIGGEST_OBJECT     CASCADE_FIND_BIGGEST_OBJECT
#endif

using namespace cv;
using namespace face;
using namespace std;

/** Constants **/


/** Function Headers */
void detectAndDisplay( Mat frame, Ptr<Facemark> facemark );
void landmarks( Mat frame, vector<Rect> faces, Ptr<Facemark> facemark );

/** Global variables */
//-- Note, either copy these two files from opencv/data/haarscascades to your current folder, or change these locations
String face_cascade_name = "../res/haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;

// Load pre-trained model
string modelFile = "/home/eyelights/Documents/face_detection/detect_godot/src/camera/res10_300x300_ssd_iter_140000.caffemodel";
string configFile = "/home/eyelights/Documents/face_detection/detect_godot/src/camera/deploy.prototxt";
dnn::Net net = dnn::readNetFromCaffe(configFile, modelFile);
string main_window_name = "Capture - Face detection";
string face_window_name = "Capture - Face";
RNG rng(12345);
Mat debugImage;
Mat skinCrCbHist = Mat::zeros(Size(256, 256), CV_8UC1);

/**
 * @function main
 */
int main( int argc, const char** argv ) {
  Mat frame;

  // Load the cascades
  if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade, please change face_cascade_name in source code.\n"); return -1; };
  namedWindow(main_window_name,CV_WINDOW_NORMAL);
  moveWindow(main_window_name, 400, 100);
  namedWindow(face_window_name,CV_WINDOW_NORMAL);
  moveWindow(face_window_name, 10, 100);
  namedWindow("Right Eye",CV_WINDOW_NORMAL);
  moveWindow("Right Eye", 10, 600);
  namedWindow("Left Eye",CV_WINDOW_NORMAL);
  moveWindow("Left Eye", 10, 800);

  String model_filename = "../res/face_landmark_model.dat"; // Replace with actual path
  Ptr<Facemark> facemark = createFacemarkKazemi();
  facemark->loadModel(model_filename);
  //cout << "Loaded model" << endl;

  /* As the matrix dichotomy will not be applied, these windows are useless.
  namedWindow("aa",CV_WINDOW_NORMAL);
  moveWindow("aa", 10, 800);
  namedWindow("aaa",CV_WINDOW_NORMAL);
  moveWindow("aaa", 10, 800);*/

  createCornerKernels();
  ellipse(skinCrCbHist, Point(113, 155), Size(23, 15),
          43.0, 0.0, 360.0, Scalar(255, 255, 255), -1);

  // I make an attempt at supporting both 2.x and 3.x OpenCV
#if CV_MAJOR_VERSION < 3
  CvCapture* capture = cvCaptureFromCAM( 0 );
  if( capture ) {
    while( true ) {
      frame = cvQueryFrame( capture );
#else
  VideoCapture capture(0);
  if( capture.isOpened() ) {
    while( true ) {
      capture.read(frame);
#endif
      // mirror it
      flip(frame, frame, 1);
      
      // Apply the classifier to the frame
      if( !frame.empty() ) {
        detectAndDisplay( frame, facemark );
      }
      else {
        printf(" --(!) No captured frame -- Break!");
        break;
      }

      frame.copyTo(debugImage);
      imshow(main_window_name,debugImage);

      int c = waitKey(10);
      if( (char)c == 'c' ) { break; }
      if( (char)c == 'f' ) {
        imwrite("frame.png",frame);
      }

    }
  }

  releaseCornerKernels();

  return 0;
}

void findEyes(Mat frame_gray, Rect face, vector<vector<Point2f>> shapes) {
  Mat faceROI = frame_gray(face);
  Mat debugFace = faceROI;

  if (kSmoothFaceImage) {
    double sigma = kSmoothFaceFactor * face.width;
    GaussianBlur( faceROI, faceROI, Size( 0, 0 ), sigma);
  }
  //-- Find eye regions and draw them
  int eye_region_width = face.width * (kEyePercentWidth/100.0);
  int eye_region_height = face.width * (kEyePercentHeight/100.0);
  int eye_region_top = face.height * (kEyePercentTop/100.0);
  Rect leftEyeRegion(face.width*(kEyePercentSide/100.0),
                         eye_region_top,eye_region_width,eye_region_height);
  Rect rightEyeRegion(face.width - eye_region_width - face.width*(kEyePercentSide/100.0),
                          eye_region_top,eye_region_width,eye_region_height);

  //-- Find Eye Centers
  Point leftPupil = findEyeCenter(faceROI,leftEyeRegion,"Left Eye");
  Point rightPupil = findEyeCenter(faceROI,rightEyeRegion,"Right Eye");
  // get corner regions
  Rect leftRightCornerRegion(leftEyeRegion);
  leftRightCornerRegion.width -= leftPupil.x;
  leftRightCornerRegion.x += leftPupil.x;
  leftRightCornerRegion.height /= 2;
  leftRightCornerRegion.y += leftRightCornerRegion.height / 2;
  Rect leftLeftCornerRegion(leftEyeRegion);
  leftLeftCornerRegion.width = leftPupil.x;
  leftLeftCornerRegion.height /= 2;
  leftLeftCornerRegion.y += leftLeftCornerRegion.height / 2;
  Rect rightLeftCornerRegion(rightEyeRegion);
  rightLeftCornerRegion.width = rightPupil.x;
  rightLeftCornerRegion.height /= 2;
  rightLeftCornerRegion.y += rightLeftCornerRegion.height / 2;
  Rect rightRightCornerRegion(rightEyeRegion);
  rightRightCornerRegion.width -= rightPupil.x;
  rightRightCornerRegion.x += rightPupil.x;
  rightRightCornerRegion.height /= 2;
  rightRightCornerRegion.y += rightRightCornerRegion.height / 2;
  rectangle(debugFace,leftRightCornerRegion,200);
  rectangle(debugFace,leftLeftCornerRegion,200);
  rectangle(debugFace,rightLeftCornerRegion,200);
  rectangle(debugFace,rightRightCornerRegion,200);

  // change eye centers to face coordinates
  rightPupil.x += rightEyeRegion.x;
  rightPupil.y += rightEyeRegion.y;
  leftPupil.x += leftEyeRegion.x;
  leftPupil.y += leftEyeRegion.y;
  // draw eye centers
  circle(debugFace, rightPupil, 3, 1234);
  circle(debugFace, leftPupil, 3, 1234);

  vector<float> left_limit = {(shapes[0][38].x + shapes[0][40].x)/2 , (shapes[0][44].x + shapes[0][46].x)/2};
  vector<float> right_limit = {(shapes[0][37].x + shapes[0][41].x)/2 , (shapes[0][43].x + shapes[0][47].x)/2};
  vector<float> up_limit = {(shapes[0][36].y + shapes[0][39].y + shapes[0][37].y + shapes[0][38].y)/4 , (shapes[0][42].y + shapes[0][45].y + shapes[0][43].y + shapes[0][44].y)/4};
  vector<float> down_limit = {(shapes[0][36].y + shapes[0][39].y)/2 , (shapes[0][42].y + shapes[0][45].y)/2};

  /*float lefteyex = (shapes[0][36].x + shapes[0][37].x + shapes[0][38].x + shapes[0][39].x + shapes[0][40].x + shapes[0][41].x)/6 - face.x; 
  float lefteyey = (shapes[0][36].y + shapes[0][37].y + shapes[0][38].y + shapes[0][39].y + shapes[0][40].y + shapes[0][41].y)/6 - face.y;
  cv::Point lefteye = Point(int(lefteyex) , int(lefteyey));
  arrowedLine(faceROI, leftPupil, lefteye, Scalar(255,255,255), 1);
  circle(faceROI, lefteye, 3, 200);

  float righteyex = (shapes[0][42].x + shapes[0][43].x + shapes[0][44].x + shapes[0][45].x + shapes[0][46].x + shapes[0][47].x)/6 - face.x; 
  float righteyey = (shapes[0][42].y + shapes[0][43].y + shapes[0][44].y + shapes[0][45].y + shapes[0][46].y + shapes[0][47].y)/6 - face.y;
  cv::Point righteye = Point(int(righteyex) , int(righteyey));
  arrowedLine(faceROI, rightPupil, righteye, Scalar(255, 255, 255), 1);
  circle(faceROI, righteye, 3, 200);*/

  // change eye centers to frame coordinates
  rightPupil.x += face.x;
  rightPupil.y += face.y;
  leftPupil.x += face.x;
  leftPupil.y += face.y;

  string direction = "Looking CENTER";
  
  if (leftPupil.x < left_limit[0] -20 and rightPupil.x < left_limit[1] -20) {
    direction = "Looking LEFT ";
  }
  if (leftPupil.x > right_limit[0] + 20 and rightPupil.x > right_limit[1] + 20) {
    direction = "Looking RIGHT ";
  }
  if (leftPupil.y < up_limit[0] and rightPupil.y < up_limit[1]) {
    direction += "UP";
  }
  if (leftPupil.y > down_limit[0] and rightPupil.y < down_limit[1]) {
    direction += "DOWN";
  }

  putText(faceROI, direction, Point(0,30), FONT_HERSHEY_SIMPLEX, 0.9, Scalar(255, 255, 255), 2);



  //-- Find Eye Corners
  if (kEnableEyeCorner) {
    Point2f leftRightCorner = findEyeCorner(faceROI(leftRightCornerRegion), true, false);
    leftRightCorner.x += leftRightCornerRegion.x;
    leftRightCorner.y += leftRightCornerRegion.y;
    Point2f leftLeftCorner = findEyeCorner(faceROI(leftLeftCornerRegion), true, true);
    leftLeftCorner.x += leftLeftCornerRegion.x;
    leftLeftCorner.y += leftLeftCornerRegion.y;
    Point2f rightLeftCorner = findEyeCorner(faceROI(rightLeftCornerRegion), false, true);
    rightLeftCorner.x += rightLeftCornerRegion.x;
    rightLeftCorner.y += rightLeftCornerRegion.y;
    Point2f rightRightCorner = findEyeCorner(faceROI(rightRightCornerRegion), false, false);
    rightRightCorner.x += rightRightCornerRegion.x;
    rightRightCorner.y += rightRightCornerRegion.y;
    circle(faceROI, leftRightCorner, 3, 200);
    circle(faceROI, leftLeftCorner, 3, 200);
    circle(faceROI, rightLeftCorner, 3, 200);
    circle(faceROI, rightRightCorner, 3, 200);
  }

  imshow(face_window_name, faceROI);
//  Rect roi( Point( 0, 0 ), faceROI.size());
//  Mat destinationROI = debugImage( roi );
//  faceROI.copyTo( destinationROI );
}


Mat findSkin (Mat &frame) {
  Mat input;
  Mat output = Mat(frame.rows,frame.cols, CV_8U);

  cvtColor(frame, input, CV_BGR2YCrCb);

  for (int y = 0; y < input.rows; ++y) {
    const Vec3b *Mr = input.ptr<Vec3b>(y);
//    uchar *Or = output.ptr<uchar>(y);
    Vec3b *Or = frame.ptr<Vec3b>(y);
    for (int x = 0; x < input.cols; ++x) {
      Vec3b ycrcb = Mr[x];
//      Or[x] = (skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) > 0) ? 255 : 0;
      if(skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) == 0) {
        Or[x] = Vec3b(0,0,0);
      }
    }
  }
  return output;
}

/**
 * @function detectAndDisplay
 */
void detectAndDisplay( Mat frame, Ptr<Facemark> facemark ) {
  vector<Rect> faces;
  vector<vector<Point2f>> shapes;
  //Mat frame_gray;
  int x1; int x2; int y1; int y2;

  vector<Mat> rgbChannels(3);
  split(frame, rgbChannels);
  Mat frame_gray = rgbChannels[2];

  //-- Detect faces
  face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, Size(150, 150) );

  // Perform face detection
  net.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
  net.setPreferableTarget(dnn::DNN_TARGET_CUDA_FP16);

  Mat blob = dnn::blobFromImage(frame, 1.0, Size(300, 300), Scalar(104, 177, 123));
  net.setInput(blob);
  Mat detections = net.forward();

  // Process the detections and draw bounding boxes around faces
  Mat detectionsMat(detections.size[2], detections.size[3], CV_32F, detections.ptr<float>());
  for (int i = 0; i < detectionsMat.rows; ++i) {
    float confidence = detectionsMat.at<float>(i, 2);
    if (confidence > 0.3) {  // You can adjust this threshold as needed
      x1 = static_cast<int>(detectionsMat.at<float>(i, 3) * frame.cols);
      y1 = static_cast<int>(detectionsMat.at<float>(i, 4) * frame.rows);
      x2 = static_cast<int>(detectionsMat.at<float>(i, 5) * frame.cols);
      y2 = static_cast<int>(detectionsMat.at<float>(i, 6) * frame.rows);
      rectangle(frame, Point(x1, y1), Point(x2, y2), Scalar(0, 255, 0), 2);
    }
  }
  
  if (!faces.empty()) {
    if (facemark->fit(frame, faces, shapes))  {
      for (size_t i = 0; i < faces.size(); i++) {
        for (unsigned long k = 0; k < shapes[i].size() ; k++)
          circle(frame, shapes[i][k], 2, Scalar(0, 255, 0), FILLED);
      }
    }

    cv::Point center = Point((x1+x2)/2, (y1+y2)/2);
    circle(frame, center, 2, Scalar(255, 0, 0), FILLED);

    float eyesx = (shapes[0][36].x + shapes[0][37].x + shapes[0][38].x + shapes[0][39].x + shapes[0][40].x + shapes[0][41].x + shapes[0][42].x + shapes[0][43].x + shapes[0][44].x + shapes[0][45].x + shapes[0][46].x + shapes[0][47].x)/12 ; 
    float eyesy = (shapes[0][36].y + shapes[0][37].y + shapes[0][38].y + shapes[0][39].y + shapes[0][40].y + shapes[0][41].y + shapes[0][42].y + shapes[0][43].y + shapes[0][44].y + shapes[0][45].y + shapes[0][46].y + shapes[0][47].y)/12 ;
    cv::Point eyes = Point(int(eyesx) , int(eyesy*1.1));
    //circle(frame, eyes, 3, Scalar(255, 0, 0), FILLED);
    arrowedLine(frame, center, Point(eyes.x,center.y), Scalar(0,0,255), 3);

    float eyebrowsx = (shapes[0][18].x + shapes[0][19].x + shapes[0][20].x + shapes[0][21].x + shapes[0][22].x + shapes[0][23].x + shapes[0][24].x + shapes[0][25].x + shapes[0][26].x + shapes[0][27].x)/10; 
    float eyebrowsy = (shapes[0][18].y + shapes[0][19].y + shapes[0][20].y + shapes[0][21].y + shapes[0][22].y + shapes[0][23].y + shapes[0][24].y + shapes[0][25].y + shapes[0][26].y + shapes[0][27].y)/10;
    cv::Point eyebrows = Point(int(eyebrowsx) , int(eyebrowsy*1.26));
    //circle(frame, eyebrows, 3, Scalar(0, 0, 0), FILLED);
    //cout << (eyebrowsy*1.25-(y1+y2)/2)/(y1-y2) << endl;
    arrowedLine(frame, center, Point(center.x,eyebrows.y), Scalar(255,0,0), 3, 2);
  }

//  findSkin(debugImage);

  for( int i = 0; i < faces.size(); i++ )
  {
    rectangle(debugImage, faces[i], 1234);
  }
  //-- Show what you got
  if (faces.size() > 0) {
    findEyes(frame_gray, faces[0], shapes);
  }
}