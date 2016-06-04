#include <opencv2\opencv.hpp>
using namespace cv;

void bgr_to_hsv(Mat & _src, Mat & _dst)
{
	float h, s, v;
	float b, g, r;

	int rowNum = _src.rows;
	int colNum = _src.cols;

	for (int i = 0; i < rowNum; i++)
	{
		uchar* datasrc = _src.ptr<uchar>(i);
		uchar* datadst = _dst.ptr<uchar>(i);
		for (int j = 0; j < colNum; j++)
		{
			b = datasrc[3 * j];
			g = datasrc[3 * j + 1];
			r = datasrc[3 * j + 2];
			v = MAX(MAX(b, g), r);
			if (v != 0)
			{
				s = (v - MIN(MIN(b, g), r)) / v;
			}
			else
			{
				s = 0;
			}

			if (v == r)
				h = 60 * (g - b) / (v - MIN(MIN(b, g), r));
			else if (v == g)
				h = 120.f + 60 * (b - r) / (v - MIN(MIN(b, g), r));
			else if (v == b)
				h = 240.f + 60 * (r - g) / (v - MIN(MIN(b, g), r));
			if (h < 0)
				h += 360.f;

			datadst[3 * j] = h / 2;
			datadst[3 * j + 1] = s * 255;
			datadst[3 * j + 2] = v * 255;
		}
	}
}