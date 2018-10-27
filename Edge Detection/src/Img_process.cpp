#include "Img_Process.h"

void Atan_Mat(Mat src1, Mat src2, Mat &dst)
{
	double* data = dst.ptr<double>(0);
	for (int i = 0; i < dst.rows; i++)
	{
		data = dst.ptr <double>(i);
		for (int j = 0; j < dst.cols; j++)
		{
			data[j] = atan(src2.at<short>(i, j) / (src1.at<short>(i, j) + EPS));
		}
	}
}

void ConcateImages(vector<Mat> imgs, Mat &newImage, int cols, int rows, int Margin)
{
	int ImgAmount = imgs.size();
	int w = 0, h = 0;
	int Width, Height;
	for (vector<Mat>::iterator i = imgs.begin(); i != imgs.end(); i++)
	{
		if ((*i).rows > h)
			h = (*i).rows;
		if ((*i).cols > w)
			w = (*i).cols;
	}
	Width = w * cols + (cols + 1)*Margin;
	Height = h * rows + (rows + 1)*Margin;
	newImage = Mat::zeros(Height, Width, CV_8U);

	int r = 0, c = 0, imgcnt = 0;
	while (imgcnt < ImgAmount)
	{
		imgs[imgcnt].copyTo(newImage(Rect(c*w + (c + 1)*Margin, r*h + (r + 1)*Margin, imgs[imgcnt].cols, imgs[imgcnt].rows)));
		imgcnt++;
		if (c == (cols - 1))
		{
			c = 0;
			r++;
		}
		else
			c++;
	}
}

double generateGaussianNoise(double mean, double variance)
{
	//Box-Muller transform
	const double epsilon = numeric_limits<double>::min();
	static double z0, z1;
	static bool flag = false;
	flag = !flag;
	if (!flag)
		return z1 * variance + mean;
	double u1, u2;
	do
	{
		u1 = rand() * (1.0 / RAND_MAX);
		u2 = rand() * (1.0 / RAND_MAX);
	} while (u1 <= epsilon);
	z0 = sqrt(-2.0*log(u1))*cos(2 * CV_PI*u2);
	z1 = sqrt(-2.0*log(u1))*sin(2 * CV_PI*u2);
	return z0 * variance + mean;
}

void Add_GaussianNoise(Mat src, Mat &dst,double mean,double variance,double factor)
{
	for (int i = 0; i<src.rows; i++)
		for (int j = 0; j < src.cols; j++)
		{
			int val = src.at<uchar>(i, j) + (int)(generateGaussianNoise(mean, variance)*factor);
			if (val < 0)
				val = 0;
			if (val>255)
				val = 255;
			dst.at<uchar>(i, j) = val;
		}
}

void Add_PepperSaltNoise(Mat src, Mat &dst, int n1, int n2)
{
	srand((unsigned)time(NULL));
	for (int k = 0; k < n1; k++)
	{
		int i = rand() % src.rows;
		int j = rand() % src.cols;
		src.at<uchar>(i, j) = 0;
	}
	for (int k = 0; k < n2; k++)
	{
		int i = rand() % src.rows;
		int j = rand() % src.cols;
		src.at<uchar>(i, j) = 255;
	}
}