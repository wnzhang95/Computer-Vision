#include "evaluate.h"

#define MAXLABEL 500
uchar parent[MAXLABEL] = { 0 };

int Find(uchar x, uchar* parent)
{
	int i = x;
	while (0 != parent[i])
		i = parent[i];
	return i;
}

void Union(uchar big, uchar small, uchar* parent)
{
	uchar i = big;
	uchar j = small;
	while (0 != parent[i])
		i = parent[i];
	while (0 != parent[j])
		j = parent[j];
	if (i != j)
		parent[i] = j;
}

Mat Connected_Component_8(Mat I)
{
	/// first pass
	int label = 0;

	Mat dst = Mat::zeros(I.size(), I.type());
	for (int nY = 0; nY < I.rows; nY++)
	{
		for (int nX = 0; nX < I.cols; nX++)
		{
			if (I.at<uchar>(nY, nX) != 0)
			{
				uchar left = nX - 1<0 ? 0 : dst.at<uchar>(nY, nX - 1);
				uchar up = nY - 1<0 ? 0 : dst.at<uchar>(nY - 1, nX);
				uchar leftup = (nY - 1 < 0 || nX - 1 < 0) ? 0 : dst.at<uchar>(nY - 1, nX - 1);
				vector<uchar> V;
				V.push_back(left);
				V.push_back(up);
				V.push_back(leftup);
				sort(V.begin(), V.end());

				if( V.back()==0 ) 
					dst.at<uchar>(nY, nX) = ++label;
				else
				{
					int flag = 0,minVal=0;
					for (int i = 0; i < V.size(); i++)
					{
						if (V[i] > 0 && flag == 0)
						{
							flag = 1;
							minVal = V[i];
							dst.at<uchar>(nY, nX) = minVal;
						}
						else if (V[i] > 0)
						{
							Union(V[i], minVal, parent);
						}
					}
				}

				//if (left != 0 || up != 0)
				//{
				//	if (left != 0 && up != 0)
				//	{
				//		dst.at<uchar>(nY, nX) = min(left, up);
				//		if (left < up)
				//			Union(up, left, parent);
				//		else if (up<left)
				//			Union(left, up, parent);
				//	}
				//	else
				//		dst.at<uchar>(nY, nX) = max(left, up);
				//}
				//else
				//{
				//	dst.at<uchar>(nY, nX) = ++label;
				//}
			}
		}
	}
	/// second pass 
	for (int nY = 0; nY < I.rows; nY++)
	{
		for (int nX = 0; nX < I.cols; nX++)
		{
			if (I.at<uchar>(nY, nX) == 1)
				dst.at<uchar>(nY, nX) = Find(dst.at<uchar>(nY, nX), parent);
		}
	}

	return dst;

}
