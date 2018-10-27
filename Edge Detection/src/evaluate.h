#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cmath>
#include <queue>
#include <vector>
#include <algorithm>
#define EPS 0.00000001

using namespace cv;
using namespace std;

int Find(uchar x, uchar* parent);
void Union(uchar big, uchar small, uchar* parent);
Mat Label(Mat I);