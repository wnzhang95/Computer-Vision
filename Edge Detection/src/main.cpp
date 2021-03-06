#include "edge_detection.h"
#include "Img_Process.h"

using namespace cv;
using namespace std;


const String WindowName1 = "Edge detection",WindowName2="Prewitt detection",WindowName3="Sobel detection" ,WindowName4="Canny detection";
const String WinRes = "result",WinSrc="source";
char trackBar_Low[10] = "Low", trackBar_High[10] = "High";
char trackBar_Pthresh[10] = "Pthresh",trackBar_Sthresh[10]="Sthresh";
char trackBar_Ksize[10] = "Ksize", trackBar_Noise[10] = "Noise";
char trackBar_Ratio[10] = "Ratio";


const int Low_threshold_min = 0, Low_idx_max = 1020, Low_step=1,High_threshold_min = 0,  High_idx_max=1020, High_step=1;
const int P_threshold_min =0 ,P_idx_max=255,P_step=1;
const int S_threshold_min = 0, S_idx_max = 255, S_step = 1;
const int ksize_min = 3, ksize_idx_max = 5, ksize_step = 2;
const int Ratio_idx_max = 100;
int Low_idx = 0, High_idx = 0, P_idx, S_idx, Gaussian_ksize_idx = 0, Noise_Type=0, Ratio_idx=0;

double Gaussian_Noise_mean=0, Gaussian_Noise_variance=0.8, Gaussian_Noise_factor = 64;
int Pepper_cnt = 2000, Salt_cnt = 2000;
Mat res;
char srcimg[100]="1.jpg";


void on_Trackbar(int , void*)
{
	Mat src, dst_prewitt, dst_sobel, dst_canny, temp;
	Mat result;
	vector<Mat> Imgs;

	int Low_threshold = Low_idx * Low_step + Low_threshold_min;
	int High_threshold = High_idx * High_step + High_threshold_min;
	int P_threshold = P_idx * P_step+P_threshold_min;
	int S_threshold = S_idx * S_step + S_threshold_min;
	int Gaussian_ksize = Gaussian_ksize_idx * ksize_step + ksize_min;
	
	src = imread(srcimg);
	imshow(WinSrc, src);
	cvtColor(src, src, CV_RGB2GRAY);

	switch (Noise_Type)
	{
	case 0: break;
	case 1:Add_PepperSaltNoise(src, src, Pepper_cnt,Salt_cnt); break;
	case 2:Add_GaussianNoise(src, src,Gaussian_Noise_mean,Gaussian_Noise_variance,Gaussian_Noise_factor); break;
	}
	Imgs.clear();
	Imgs.push_back(src);

	
	Prewitt_Detection(src, dst_prewitt,temp,temp, Gaussian_ksize,P_threshold);
	Sobel_Detection(src, dst_sobel,temp,temp, Gaussian_ksize,S_threshold);
	Canny_Detection(src, dst_canny, Low_threshold, High_threshold, Gaussian_ksize);

	Imgs.push_back(dst_prewitt);
	Imgs.push_back(dst_sobel);
	Imgs.push_back(dst_canny);
	ConcateImages(Imgs, result, 4, 1, 20);
	imshow(WinRes, result);
	res = result;
}

void on_Trackbar_Prewitt(int, void*)
{
	Mat src, dst_prewitt, dst_prewitt_x, dst_prewitt_y;
	Mat result;
	vector<Mat> Imgs;

	int P_threshold = P_idx * P_step + P_threshold_min;
	int Gaussian_ksize = Gaussian_ksize_idx * ksize_step + ksize_min;

	src = imread(srcimg);
	imshow(WinSrc, src);
	cvtColor(src, src, CV_RGB2GRAY);

	Imgs.clear();
	Imgs.push_back(src);

	Prewitt_Detection(src, dst_prewitt, dst_prewitt_x,dst_prewitt_y, Gaussian_ksize,P_threshold);
	
	Imgs.push_back(dst_prewitt_x);
	Imgs.push_back(dst_prewitt_y);
	Imgs.push_back(dst_prewitt);
	ConcateImages(Imgs, result, 4, 1, 20);
	//ConcateImages(Imgs, result, 2, 1, 20);
	imshow(WinRes, result);
	res = result;

}

void on_Trackbar_Sobel(int, void*)
{
	Mat src, dst_sobel, dst_sobel_x, dst_sobel_y;
	Mat result;
	vector<Mat> Imgs;

	int S_threshold = S_idx * S_step + S_threshold_min;
	int Gaussian_ksize = Gaussian_ksize_idx * ksize_step + ksize_min;


	src = imread(srcimg);
	imshow(WinSrc, src);
	cvtColor(src, src, CV_RGB2GRAY);

	Imgs.clear();
	Imgs.push_back(src);

	Sobel_Detection(src, dst_sobel, dst_sobel_x, dst_sobel_y, Gaussian_ksize, S_threshold);
	Imgs.push_back(dst_sobel_x);
	Imgs.push_back(dst_sobel_y);
	Imgs.push_back(dst_sobel);
	ConcateImages(Imgs, result, 4, 1, 20);
	//ConcateImages(Imgs, result, 2, 1, 20);
	imshow(WinRes, result);
	res = result;

}

void on_Trackbar_Canny(int, void*)
{
	Mat src, dst_canny;
	Mat result;
	vector<Mat> Imgs;

	int Low_threshold = Low_idx * Low_step + Low_threshold_min;
	int High_threshold = High_idx * High_step + High_threshold_min;
	int Gaussian_ksize = Gaussian_ksize_idx * ksize_step + ksize_min;

	src = imread(srcimg);
	imshow(WinSrc, src);
	cvtColor(src, src, CV_RGB2GRAY);

	Imgs.clear();
	Imgs.push_back(src);

	Canny_Detection(src, dst_canny, Low_threshold, High_threshold, Gaussian_ksize);

	Imgs.push_back(dst_canny);
	ConcateImages(Imgs, result, 2, 1, 20);
	imshow(WinRes, result);
	res = result;
}

void on_Trackbar_Adaptive(int, void*)
{
	Mat src, dst_prewitt, dst_sobel, dst_canny, temp;
	Mat result;
	vector<Mat> Imgs;

	int Gaussian_ksize = Gaussian_ksize_idx * ksize_step + ksize_min;

	src = imread(srcimg);
	imshow(WinSrc, src);
	cvtColor(src, src, CV_RGB2GRAY);

	switch (Noise_Type)
	{
	case 0: break;
	case 1:Add_PepperSaltNoise(src, src, Pepper_cnt, Salt_cnt); break;
	case 2:Add_GaussianNoise(src, src, Gaussian_Noise_mean, Gaussian_Noise_variance, Gaussian_Noise_factor); break;
	}
	Imgs.clear();
	Imgs.push_back(src);

	double start;
	double time_p, time_s, time_c;

	start = static_cast<double>(getTickCount());
	Prewitt_Detection_Adaptive(src, dst_prewitt, temp, temp, Gaussian_ksize);
	time_p = ((double)getTickCount() - start) / getTickFrequency();

	start = static_cast<double>(getTickCount());
	Sobel_Detection_Adaptive(src, dst_sobel, temp, temp, Gaussian_ksize);
	time_s = ((double)getTickCount() - start) / getTickFrequency();

	start = static_cast<double>(getTickCount());
	Canny_Detection_Adaptive(src, dst_canny, Gaussian_ksize);
	time_c = ((double)getTickCount() - start) / getTickFrequency();

	cout << "Prewitt: " << time_p << " s" << endl
		<< "Sobel: " << time_s << " s" << endl
		<< "Canny: " << time_c << " s" << endl;

	Imgs.push_back(dst_prewitt);
	Imgs.push_back(dst_sobel);
	Imgs.push_back(dst_canny);
	ConcateImages(Imgs, result, 4, 1, 20);
	//ConcateImages(Imgs, result, 2, 1, 20);
	imshow(WinRes, result);

	res = result;
}

int main()
{
	int func;
	char c;
	char imgname[100];

	while (1) {
		cout << "Input image :" << endl;
		cin >> srcimg;
		while (!(imread(srcimg).data))
		{
			cout << "Fail to read image! " << endl;
			cin >> srcimg;
		}
		cout << "Choose function :" << endl
			<< "1:All" << endl
			<< "2:Prewitt" << endl
			<< "3:Sobel:" << endl
			<< "4:Canny" << endl
			<< "5:Adaptive" << endl;
		cout << "Press s to save image and Press q to destroy all windows" << endl;
		cin >> func;
		switch (func)
		{
		case 1:
			namedWindow(WindowName1, WINDOW_FREERATIO);
			namedWindow(WinRes, WINDOW_AUTOSIZE);
			createTrackbar(trackBar_Low, WindowName1, &Low_idx, Low_idx_max, on_Trackbar);
			createTrackbar(trackBar_High, WindowName1, &High_idx, High_idx_max, on_Trackbar);
			createTrackbar(trackBar_Pthresh, WindowName1, &P_idx, P_idx_max, on_Trackbar);
			createTrackbar(trackBar_Sthresh, WindowName1, &S_idx, S_idx_max, on_Trackbar);
			createTrackbar(trackBar_Ksize, WindowName1, &Gaussian_ksize_idx, ksize_idx_max, on_Trackbar);
			createTrackbar(trackBar_Noise, WindowName1, &Noise_Type, 2, on_Trackbar);

			while (1)
			{
				on_Trackbar(Low_idx, 0);
				on_Trackbar(High_idx, 0);
				on_Trackbar(P_idx, 0);
				on_Trackbar(S_idx, 0);
				on_Trackbar(Gaussian_ksize_idx, 0);
				on_Trackbar(Noise_Type, 0);
				c = waitKey(0);
				if ( c== 's')
				{
					sprintf_s(imgname, "All_%d_%d_%d_%d_%d_%d.jpg", Gaussian_ksize_idx * 2 + 3, P_idx,S_idx,Low_idx, High_idx, Noise_Type);
					imwrite(imgname, res);
				}
				if ( c == 'q') break;
			}
			destroyAllWindows();
			break;

		case 2:
			namedWindow(WindowName2, WINDOW_AUTOSIZE);
			namedWindow(WinRes, WINDOW_AUTOSIZE);
			createTrackbar(trackBar_Ksize, WindowName2, &Gaussian_ksize_idx, ksize_idx_max, on_Trackbar_Prewitt);
			createTrackbar(trackBar_Low, WindowName2, &P_idx, P_idx_max, on_Trackbar_Prewitt);
			while (1)
			{
				on_Trackbar_Prewitt(Gaussian_ksize_idx, 0);
				on_Trackbar_Prewitt(P_idx, 0);
				c = waitKey(0);
				if ( c== 's')
				{
					sprintf_s(imgname, "Prewitt_%d_%d.jpg", Gaussian_ksize_idx * 2 + 3, P_idx);
					imwrite(imgname, res);
				}
				if ( c == 'q') break;
			}
			destroyAllWindows();
			break;

		case 3:
			namedWindow(WindowName3, WINDOW_AUTOSIZE);
			namedWindow(WinRes, WINDOW_AUTOSIZE);
			createTrackbar(trackBar_Ksize, WindowName3, &Gaussian_ksize_idx, ksize_idx_max, on_Trackbar_Sobel);
			createTrackbar(trackBar_Low, WindowName3, &S_idx, S_idx_max, on_Trackbar_Sobel);
			
			while (1)
			{
				on_Trackbar_Sobel(Gaussian_ksize_idx, 0);
				on_Trackbar_Sobel(S_idx, 0);
				c = waitKey(0);
				if ( c== 's')
				{
					sprintf_s(imgname, "Sobel_%d_%d.jpg", Gaussian_ksize_idx * 2 + 3, Low_idx);
					imwrite(imgname, res);
				}
				if (c == 'q') break;
			}
			destroyAllWindows();
			break;

		case 4:
			namedWindow(WindowName4, WINDOW_AUTOSIZE);
			namedWindow(WinRes, WINDOW_AUTOSIZE);
			createTrackbar(trackBar_Low, WindowName4, &Low_idx, Low_idx_max, on_Trackbar_Canny);
			createTrackbar(trackBar_High, WindowName4, &High_idx, High_idx_max, on_Trackbar_Canny);
			createTrackbar(trackBar_Ksize, WindowName4, &Gaussian_ksize_idx, ksize_idx_max, on_Trackbar_Canny);
			while (1)
			{
				on_Trackbar_Canny(Low_idx, 0);
				on_Trackbar_Canny(High_idx, 0);
				on_Trackbar_Canny(Gaussian_ksize_idx, 0);
				c = waitKey(0);
				if (c == 's')
				{
					sprintf_s(imgname, "Canny_%d_%d_%d.jpg", Gaussian_ksize_idx * 2 + 3, Low_idx, High_idx);
					imwrite(imgname, res);
				}
				if (c == 'q') break;
			}
			destroyAllWindows();
			break;
		case 5:
			namedWindow(WindowName1, WINDOW_AUTOSIZE);
			namedWindow(WinRes, WINDOW_AUTOSIZE);
			//createTrackbar(trackBar_Ratio, WindowName1, &Ratio_idx, Ratio_idx_max, on_Trackbar_Adaptive);
			createTrackbar(trackBar_Ksize, WindowName1, &Gaussian_ksize_idx, ksize_idx_max, on_Trackbar_Adaptive);
			createTrackbar(trackBar_Noise, WindowName1, &Noise_Type, 2, on_Trackbar_Adaptive);
			while (1)
			{
				//on_Trackbar_Adaptive(Ratio_idx, 0);
				on_Trackbar_Adaptive(Gaussian_ksize_idx, 0);
				on_Trackbar_Adaptive(Noise_Type, 0);
				c = waitKey(0);
				if (c == 's')
				{
					sprintf_s(imgname, "Adaptive_%d_%d.jpg", Gaussian_ksize_idx * 2 + 3, Noise_Type);
					imwrite(imgname, res);
				}
				if (c == 'q') break;
			}
			destroyAllWindows();
			break;
		}
	}
	return 0;
}

