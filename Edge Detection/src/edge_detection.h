#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cmath>
#include <queue>
#include <vector>
#define EPS 0.00000001

using namespace cv;
using namespace std;

void Sobel_Detection(Mat src, Mat &dst, Mat &Gx, Mat &Gy, int GaussianBlur_ksize = 3, int thresh = 200);
void Prewitt_Detection(Mat src, Mat &dst, Mat &Gx, Mat &Gy, int GaussianBlur_ksize = 3, int thresh = 200);
void Canny_Detection(Mat src, Mat &dst, int threshold_low, int threshold_high, int GaussianBlur_ksize = 3);
void Sobel_Detection_Adaptive(Mat src, Mat &dst, Mat &Gx, Mat &Gy, int GaussianBlur_ksize);
void Prewitt_Detection_Adaptive(Mat src, Mat &dst, Mat &Gx, Mat &Gy, int GaussianBlur_ksize);
void Canny_Detection_Adaptive(Mat src, Mat &dst, int GaussianBlur_ksize);
