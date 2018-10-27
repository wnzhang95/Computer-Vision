#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cmath>
#include <queue>
#include <vector>
#define EPS 0.00000001

using namespace std;
using namespace cv;


void Atan_Mat(Mat src1, Mat src2, Mat &dst);
void ConcateImages(vector<Mat> imgs, Mat &newImage, int cols, int rows, int Margin);
double generateGaussianNoise(double mean, double variance);
void Add_GaussianNoise(Mat src, Mat &dst, double mean, double variance, double factor);
void Add_PepperSaltNoise(Mat src, Mat &dst, int n1, int n2);