#include "edge_detection.h"


void Sobel_Detection(Mat src, Mat &dst, Mat &Gx, Mat &Gy, int GaussianBlur_ksize , int thresh )
{
	Mat Kernal_x, Kernal_y;
	Mat grad_x, grad_y, grad;
	Kernal_x = (Mat_<double>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
	Kernal_y = (Mat_<double>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);

	GaussianBlur(src, src, Size(GaussianBlur_ksize, GaussianBlur_ksize), 0, 0, BORDER_DEFAULT);
	//ddepth=CV_16S prevent overflow
	filter2D(src, grad_x, CV_16S, Kernal_x, Point(-1, -1), 0, BORDER_DEFAULT);
	filter2D(src, grad_y, CV_16S, Kernal_y, Point(-1, -1), 0, BORDER_DEFAULT);

	//convert CV_16S to CV_8U
	convertScaleAbs(grad_x, Gx);
	convertScaleAbs(grad_y, Gy);

	//imshow("grad_x", grad_x_abs);
	//imshow("grad_y", grad_y_abs);

	grad = abs(grad_x) + abs(grad_y);
	convertScaleAbs(grad, dst);
	threshold(dst, dst, thresh, 255, THRESH_BINARY);
}

void Prewitt_Detection(Mat src, Mat &dst, Mat &Gx, Mat &Gy, int GaussianBlur_ksize , int thresh )
{
	Mat Kernal_x, Kernal_y;
	Mat grad_x, grad_y, grad;
	Kernal_x = (Mat_<double>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
	Kernal_y = (Mat_<double>(3, 3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);

	GaussianBlur(src, src, Size(GaussianBlur_ksize, GaussianBlur_ksize), 0, 0, BORDER_DEFAULT);

	filter2D(src, grad_x, CV_16S, Kernal_x, Point(-1, -1), 0, BORDER_DEFAULT);
	filter2D(src, grad_y, CV_16S, Kernal_y, Point(-1, -1), 0, BORDER_DEFAULT);
	convertScaleAbs(grad_x, Gx);
	convertScaleAbs(grad_y, Gy);

	grad = abs(grad_x) + abs(grad_y);
	convertScaleAbs(grad, dst);
	threshold(dst, dst, thresh, 255, THRESH_BINARY);
}

void Canny_Detection(Mat src, Mat &dst, int threshold_low, int threshold_high, int GaussianBlur_ksize )
{
	Mat grad_x, grad_y, grad;// theta = Mat_<double>(src.rows, src.cols);
	Mat Kernal_x, Kernal_y;

	Mat grad_temp(src.rows + 2, src.cols + 2, CV_16S, Scalar(0));
	Mat mask(src.rows, src.cols, CV_8U, Scalar(1));
	Mat dst_temp(src.rows + 2, src.cols + 2, CV_8U, Scalar(0));
	double g1, g2, t, theta;
	if (threshold_high < threshold_low)
	{
		int temp = threshold_low;
		threshold_low = threshold_high;
		threshold_high = temp;
	}

	Kernal_x = (Mat_<double>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
	Kernal_y = (Mat_<double>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);

	GaussianBlur(src, src, Size(GaussianBlur_ksize, GaussianBlur_ksize), 0, 0, BORDER_DEFAULT);


	filter2D(src, grad_x, CV_16S, Kernal_x, Point(-1, -1), 0, BORDER_DEFAULT);
	filter2D(src, grad_y, CV_16S, Kernal_y, Point(-1, -1), 0, BORDER_DEFAULT);

	grad = abs(grad_x) + abs(grad_y);
	grad.copyTo(grad_temp(Rect(1, 1, grad.cols, grad.rows)), mask);

	// NMS
	queue<CvPoint> Q_weak;

	for (int i = 1; i <= grad.rows; i++)
		for (int j = 1; j <= grad.cols; j++)
		{
			theta = atan(grad_y.at<short>(i - 1, j - 1) / (grad_x.at<short>(i - 1, j - 1) + EPS));
			if (theta >= -90 * CV_PI / 180 && theta < -45 * CV_PI / 180)
			{
				t = abs(grad_x.at<short>(i - 1, j - 1) / (grad_y.at<short>(i - 1, j - 1) + EPS));
				g1 = grad_temp.at<short>(i - 1, j) * (1 - t) + grad_temp.at<short>(i - 1, j - 1) * t;
				g2 = grad_temp.at<short>(i + 1, j) * (1 - t) + grad_temp.at<short>(i + 1, j + 1) * t;
			}
			else if (theta >= -45 * CV_PI / 180 && theta < 0)
			{
				t = abs(grad_y.at<short>(i - 1, j - 1) / (grad_x.at<short>(i - 1, j - 1) + EPS));
				g1 = grad_temp.at<short>(i, j - 1) * (1 - t) + grad_temp.at<short>(i - 1, j - 1) * t;
				g2 = grad_temp.at<short>(i, j + 1) * (1 - t) + grad_temp.at<short>(i + 1, j + 1) * t;
			}
			else if (theta >= 0 && theta < 45 * CV_PI / 180)
			{
				t = abs(grad_y.at<short>(i - 1, j - 1) / (grad_x.at<short>(i - 1, j - 1) + EPS));
				g1 = grad_temp.at<short>(i, j + 1) * (1 - t) + grad_temp.at<short>(i - 1, j + 1) * t;
				g2 = grad_temp.at<short>(i, j - 1) * (1 - t) + grad_temp.at<short>(i + 1, j - 1) * t;
			}
			else if (theta >= 45 * CV_PI / 180 && theta < 90 * CV_PI / 180)
			{
				t = abs(grad_x.at<short>(i - 1, j - 1) / (grad_y.at<short>(i - 1, j - 1) + EPS));
				g1 = grad_temp.at<short>(i - 1, j) * (1 - t) + grad_temp.at<short>(i - 1, j + 1) * t;
				g2 = grad_temp.at<short>(i + 1, j) * (1 - t) + grad_temp.at<short>(i + 1, j - 1) * t;
			}
			if (grad_temp.at<short>(i, j) >= g1 && grad_temp.at<short>(i, j) >= g2)
			{
				if (grad_temp.at<short>(i, j) > threshold_high)
				{
					dst_temp.at<uchar>(i, j) = 255;
				}
				else if (grad_temp.at<short>(i, j) > threshold_low)
				{
					dst_temp.at<uchar>(i, j) = 1;
					CvPoint p(i, j);
					Q_weak.push(p);
				}
			}
		}

	while (!Q_weak.empty())
	{
		int queuesize = Q_weak.size();
		for (int i = 0; i < queuesize; i++)
		{
			CvPoint p = Q_weak.front();
			int x, y;
			x = p.x;
			y = p.y;
			if (dst_temp.at<uchar>(x - 1, y - 1) == 255 || dst_temp.at<uchar>(x - 1, y) == 255 || dst_temp.at<uchar>(x - 1, y + 1) == 255 ||
				dst_temp.at<uchar>(x + 1, y - 1) == 255 || dst_temp.at<uchar>(x + 1, y) == 255 || dst_temp.at<uchar>(x + 1, y + 1) == 255 ||
				dst_temp.at<uchar>(x, y - 1) == 255 || dst_temp.at<uchar>(x, y + 1) == 255)
			{
				dst_temp.at<uchar>(x, y) = 255;
			}
			else if (dst_temp.at<uchar>(x - 1, y - 1) == 0 && dst_temp.at<uchar>(x - 1, y) == 0 && dst_temp.at<uchar>(x - 1, y + 1) == 0 &&
				dst_temp.at<uchar>(x + 1, y - 1) == 0 && dst_temp.at<uchar>(x + 1, y) == 0 && dst_temp.at<uchar>(x + 1, y + 1) == 0 &&
				dst_temp.at<uchar>(x, y - 1) == 0 && dst_temp.at<uchar>(x, y + 1) == 0)
			{
				dst_temp.at<uchar>(x, y) = 0;
			}
			else
			{
				Q_weak.push(Q_weak.front());
			}
			Q_weak.pop();
		}
		if (Q_weak.size() >= queuesize)
		{
			while (!Q_weak.empty())
			{
				CvPoint q = Q_weak.front();
				dst_temp.at<uchar>(q.x, q.y) = 0;
				Q_weak.pop();
			}
		}
	}
	dst_temp(Rect(1, 1, src.cols, src.rows)).copyTo(dst);
}


void Sobel_Detection_Adaptive(Mat src, Mat &dst, Mat &Gx, Mat &Gy, int GaussianBlur_ksize)
{
	Mat Kernal_x, Kernal_y;
	Mat grad_x, grad_y, grad;
	Kernal_x = (Mat_<double>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
	Kernal_y = (Mat_<double>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);

	GaussianBlur(src, src, Size(GaussianBlur_ksize, GaussianBlur_ksize), 0, 0, BORDER_DEFAULT);
	//ddepth=CV_16S prevent overflow
	filter2D(src, grad_x, CV_16S, Kernal_x, Point(-1, -1), 0, BORDER_DEFAULT);
	filter2D(src, grad_y, CV_16S, Kernal_y, Point(-1, -1), 0, BORDER_DEFAULT);

	//convert CV_16S to CV_8U
	convertScaleAbs(grad_x, Gx);
	convertScaleAbs(grad_y, Gy);

	grad = abs(grad_x) + abs(grad_y);
	convertScaleAbs(grad, dst);

	//imshow("grad_x", grad_x_abs);
	//imshow("grad_y", grad_y_abs);


	//MatND dstHist;
	//int histsize = 256;
	//float Range[] = { 0,256 };
	//const float* ranges[] = { Range };
	//calcHist(&dst, 1, 0, Mat(), dstHist, 1, &histsize, ranges,true,false);
	//int cnt = (int)(src.rows*src.cols*(1-edge_ratio)), sum = 0,idx=0;
	//for (idx = 0; idx < histsize; idx++)
	//{
	//	sum += dstHist.at<float>(idx);
	//	if (sum >= cnt)
	//		break;
	//}
	threshold(dst, dst, 0, 255, THRESH_OTSU | THRESH_BINARY);
}

void Prewitt_Detection_Adaptive(Mat src, Mat &dst, Mat &Gx, Mat &Gy, int GaussianBlur_ksize)
{
	Mat Kernal_x, Kernal_y;
	Mat grad_x, grad_y, grad;
	Kernal_x = (Mat_<double>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
	Kernal_y = (Mat_<double>(3, 3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);

	GaussianBlur(src, src, Size(GaussianBlur_ksize, GaussianBlur_ksize), 0, 0, BORDER_DEFAULT);

	filter2D(src, grad_x, CV_16S, Kernal_x, Point(-1, -1), 0, BORDER_DEFAULT);
	filter2D(src, grad_y, CV_16S, Kernal_y, Point(-1, -1), 0, BORDER_DEFAULT);
	convertScaleAbs(grad_x, Gx);
	convertScaleAbs(grad_y, Gy);

	grad = abs(grad_x) + abs(grad_y);
	convertScaleAbs(grad, dst);

	//MatND dstHist;
	//int histsize = 256;
	//float Range[] = { 0,256 };
	//const float* ranges[] = { Range };
	//calcHist(&dst, 1, 0, Mat(), dstHist, 1, &histsize, ranges, true, false);
	//int cnt = (int)(src.rows*src.cols*(1 - edge_ratio)), sum = 0, idx = 0;
	//for (idx = 0; idx < histsize; idx++)
	//{
	//	sum += dstHist.at<float>(idx);
	//	if (sum >= cnt)
	//		break;
	//}
	threshold(dst, dst, 0, 255, THRESH_OTSU | THRESH_BINARY);

}

void Canny_Detection_Adaptive(Mat src, Mat &dst, int GaussianBlur_ksize)
{
	Mat grad_x, grad_y, grad;// theta = Mat_<double>(src.rows, src.cols);
	Mat Kernal_x, Kernal_y;
	double threshold_low,threshold_high;

	Mat grad_temp(src.rows + 2, src.cols + 2, CV_16S, Scalar(0));
	Mat mask(src.rows, src.cols, CV_8U, Scalar(1));
	Mat dst_temp(src.rows + 2, src.cols + 2, CV_8U, Scalar(0));
	double g1, g2, t, theta;

	Kernal_x = (Mat_<double>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
	Kernal_y = (Mat_<double>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);

	GaussianBlur(src, src, Size(GaussianBlur_ksize, GaussianBlur_ksize), 0, 0, BORDER_DEFAULT);


	filter2D(src, grad_x, CV_16S, Kernal_x, Point(-1, -1), 0, BORDER_DEFAULT);
	filter2D(src, grad_y, CV_16S, Kernal_y, Point(-1, -1), 0, BORDER_DEFAULT);

	grad = abs(grad_x) + abs(grad_y);
	grad.copyTo(grad_temp(Rect(1, 1, grad.cols, grad.rows)), mask);
	//Mat grad_16u;
	//grad.convertTo(grad_16u, CV_16U);
	convertScaleAbs(grad, grad);
	Mat high_mat;
	threshold_high=threshold(grad,high_mat, 0, 255, THRESH_BINARY | THRESH_OTSU);
	threshold_low = threshold_high / 2.5;

	//double MaxVal;
	//minMaxIdx(grad, 0, &MaxVal, 0, 0);
	//MatND dstHist;
	//int histsize=MaxVal;
	//float Range[] = { 0,MaxVal };
	//const float* ranges[] = { Range };
	//calcHist(&grad_16u, 1, 0, Mat(), dstHist, 1, &histsize, ranges, true, false);
	//int cnt = (int)(src.rows*src.cols*(1 - edge_ratio)), sum = 0, idx = 0;
	//for (idx = 0; idx < histsize; idx++)
	//{
	//	sum += dstHist.at<float>(idx);
	//	if (sum >= cnt)
	//		break;
	//}
	//threshold_high=idx*MaxVal/histsize;
	//threshold_low= threshold_high/2.5;

	// NMS
	queue<CvPoint> Q_weak;

	for (int i = 1; i <= grad.rows; i++)
		for (int j = 1; j <= grad.cols; j++)
		{
			theta = atan(grad_y.at<short>(i - 1, j - 1) / (grad_x.at<short>(i - 1, j - 1) + EPS));
			if (theta >= -90 * CV_PI / 180 && theta < -45 * CV_PI / 180)
			{
				t = abs(grad_x.at<short>(i - 1, j - 1) / (grad_y.at<short>(i - 1, j - 1) + EPS));
				g1 = grad_temp.at<short>(i - 1, j) * (1 - t) + grad_temp.at<short>(i - 1, j - 1) * t;
				g2 = grad_temp.at<short>(i + 1, j) * (1 - t) + grad_temp.at<short>(i + 1, j + 1) * t;
			}
			else if (theta >= -45 * CV_PI / 180 && theta < 0)
			{
				t = abs(grad_y.at<short>(i - 1, j - 1) / (grad_x.at<short>(i - 1, j - 1) + EPS));
				g1 = grad_temp.at<short>(i, j - 1) * (1 - t) + grad_temp.at<short>(i - 1, j - 1) * t;
				g2 = grad_temp.at<short>(i, j + 1) * (1 - t) + grad_temp.at<short>(i + 1, j + 1) * t;
			}
			else if (theta >= 0 && theta < 45 * CV_PI / 180)
			{
				t = abs(grad_y.at<short>(i - 1, j - 1) / (grad_x.at<short>(i - 1, j - 1) + EPS));
				g1 = grad_temp.at<short>(i, j + 1) * (1 - t) + grad_temp.at<short>(i - 1, j + 1) * t;
				g2 = grad_temp.at<short>(i, j - 1) * (1 - t) + grad_temp.at<short>(i + 1, j - 1) * t;
			}
			else if (theta >= 45 * CV_PI / 180 && theta < 90 * CV_PI / 180)
			{
				t = abs(grad_x.at<short>(i - 1, j - 1) / (grad_y.at<short>(i - 1, j - 1) + EPS));
				g1 = grad_temp.at<short>(i - 1, j) * (1 - t) + grad_temp.at<short>(i - 1, j + 1) * t;
				g2 = grad_temp.at<short>(i + 1, j) * (1 - t) + grad_temp.at<short>(i + 1, j - 1) * t;
			}
			if (grad_temp.at<short>(i, j) >= g1 && grad_temp.at<short>(i, j) >= g2)
			{
				if (grad_temp.at<short>(i, j) > threshold_high)
				{
					dst_temp.at<uchar>(i, j) = 255;
				}
				else if (grad_temp.at<short>(i, j) > threshold_low)
				{
					dst_temp.at<uchar>(i, j) = 1;
					CvPoint p(i, j);
					Q_weak.push(p);
				}
			}
		}

	while (!Q_weak.empty())
	{
		int queuesize = Q_weak.size();
		for (int i = 0; i < queuesize; i++)
		{
			CvPoint p = Q_weak.front();
			int x, y;
			x = p.x;
			y = p.y;
			if (dst_temp.at<uchar>(x - 1, y - 1) == 255 || dst_temp.at<uchar>(x - 1, y) == 255 || dst_temp.at<uchar>(x - 1, y + 1) == 255 ||
				dst_temp.at<uchar>(x + 1, y - 1) == 255 || dst_temp.at<uchar>(x + 1, y) == 255 || dst_temp.at<uchar>(x + 1, y + 1) == 255 ||
				dst_temp.at<uchar>(x, y - 1) == 255 || dst_temp.at<uchar>(x, y + 1) == 255)
			{
				dst_temp.at<uchar>(x, y) = 255;
			}
			else if (dst_temp.at<uchar>(x - 1, y - 1) == 0 && dst_temp.at<uchar>(x - 1, y) == 0 && dst_temp.at<uchar>(x - 1, y + 1) == 0 &&
				dst_temp.at<uchar>(x + 1, y - 1) == 0 && dst_temp.at<uchar>(x + 1, y) == 0 && dst_temp.at<uchar>(x + 1, y + 1) == 0 &&
				dst_temp.at<uchar>(x, y - 1) == 0 && dst_temp.at<uchar>(x, y + 1) == 0)
			{
				dst_temp.at<uchar>(x, y) = 0;
			}
			else
			{
				Q_weak.push(Q_weak.front());
			}
			Q_weak.pop();
		}
		if (Q_weak.size() >= queuesize)
		{
			while (!Q_weak.empty())
			{
				CvPoint q = Q_weak.front();
				dst_temp.at<uchar>(q.x, q.y) = 0;
				Q_weak.pop();
			}
		}
	}
	dst_temp(Rect(1, 1, src.cols, src.rows)).copyTo(dst);
}

