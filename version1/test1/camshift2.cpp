#include <iostream>
#include <ctype.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;
bool selectObject = false;
int trackObject = 0;
Rect selection;
Point origin;
Mat image;
void meanshift(Mat & _prob_img, Rect& _box);
void bgr_to_hsv(Mat & _src, Mat & _dst);
void calcBackProj(Mat &hsv, Mat &hist, Mat &backProj);
void calcHistOfHsv(Mat &hsv, Mat &hist);
static void onMouse(int event, int x, int y, int, void*)
{
	if (selectObject)
	{
		selection.x = MIN(x, origin.x);
		selection.y = MIN(y, origin.y);
		selection.width = std::abs(x - origin.x);
		selection.height = std::abs(y - origin.y);
		selection &= Rect(0, 0, image.cols, image.rows);
	}

	switch (event)
	{
	case CV_EVENT_LBUTTONDOWN:
		origin = Point(x, y);
		selection = Rect(x, y, 0, 0);
		selectObject = true;
		break;
	case CV_EVENT_LBUTTONUP:
		selectObject = false;
		if (selection.width > 0 && selection.height > 0)
			trackObject = -1;
		break;
	}
}

void drawBox(Mat& image, CvRect box, Scalar color, int thick)
{
	rectangle(image, cvPoint(box.x, box.y), cvPoint(box.x + box.width, box.y + box.height), color, thick);
}
int main()
{
	Mat frame;
	VideoCapture cap(0);
	cap.read(frame);
	imshow("CamShift Demo", frame);
	setMouseCallback("CamShift Demo", onMouse, 0);
	int count = 0;
	Mat  hsv = Mat::zeros(frame.size(), CV_8UC3), hist = Mat::zeros(1, 20, CV_8UC1), backproj;
	while (cap.read(frame))
	{
		image = frame.clone();
		if (selectObject)

		{
			rectangle(frame, selection, Scalar(255, 0, 0), 3, 8);
		}
		if (trackObject)
		{
			bgr_to_hsv(image, hsv);
			if (trackObject < 0)
			{
				//cvtColor(image, hsv, COLOR_BGR2HSV);
				calcHistOfHsv(hsv(selection), hist);
				trackObject = 1;
			}

			calcBackProj(hsv, hist, backproj);
			meanshift(backproj, selection);
			imshow("backproj", backproj);
			rectangle(frame, selection, Scalar(255, 0, 0), 3, 8);
			count++;
		}
		imshow("CamShift Demo", frame);
		waitKey(30);
		cout << "frame=" << count << endl;
	}
	return 0;
}


void meanshift(Mat & _prob_img, Rect& _box)
{
	Rect box = selection;
	Mat img = _prob_img.clone();
	Mat roi;
	double m00;
	double m10, m01;
	float temp = 0;
	int detx, dety;
	int width = 0, height = 0;
	int count = 0;
	while (count++<20)
	{
		m00 = 0;
		m01 = 0;
		m10 = 0;
		detx = 0;
		dety = 0;
		/*if (_box.x + _box.width > _prob_img.cols || _box.y + _box.height > _prob_img.rows)
		break;
		else*/
		roi = img(_box);
		for (int x = 0; x < roi.rows; x++)
		{
			uchar* data = roi.ptr<uchar>(x);
			for (int y = 0; y < roi.cols; y++)
			{
				m00 += data[y];
				m10 += y*data[y];
				m01 += x*data[y];
			}
		}
		detx = m10 / m00;
		dety = m01 / m00;
		temp = pow(detx - roi.cols*0.5, 2) + pow(dety - roi.rows*0.5, 2);
		if (temp < 15)
			break;

		width =  1.8*sqrt(m00 / 256);

		if (width > 400)
			width = 400;
		if (width < 30)
			width = 30;
		height = width;
		_box.x += detx;
		_box.y += dety;
		_box.x = _box.x - 0.5*width;
		_box.y = _box.y - 0.5*height;
		/*_box.x = _box.x - 0.5*_box.width;
		_box.y = _box.y - 0.5*_box.height;*/
		_box.width = width;
		_box.height = height;

		_box = _box & Rect(0, 0, _prob_img.cols, _prob_img.rows);

		if (_box.width <= 0 || _box.height <= 0)
			break;
		/*_box.x = _box.x > 1?_box.x:1;
		_box.x = (_box.x + width) < _prob_img.cols-1 ? _box.x : _prob_img.cols-1;
		_box.y = _box.y >1?_box.y:1;
		_box.y = (_box.y + height) < _prob_img.rows-1 ? _box.y : _prob_img.rows-1;*/

		imshow("roi", roi);
	}

}
void calcHistOfHsv(Mat &hsv, Mat &hist)
{
	Mat histTmp = Mat::zeros(hist.size(), CV_32SC1);
	int *dataHist = histTmp.ptr<int>(0);
	int rowNumber = hsv.rows;
	int colNumber = hsv.cols;//只遍历H channel
	for (int i = 0; i < rowNumber; i++)
	{
		uchar *dataHsv = hsv.ptr<uchar>(i);
		for (int j = 0; j < colNumber; j++)
			dataHist[dataHsv[3 * j] / 18]++;
	}
	normalize(histTmp, histTmp, 0, 255, NORM_MINMAX);
	hist = histTmp.clone();
}

void calcBackProj(Mat &hsv, Mat &hist, Mat &backProj)
{
	Mat backProjTmp = Mat::zeros(hsv.size(), CV_8UC1);
	int numberChanel;
	numberChanel = hsv.channels();

	int rowNumber = hsv.rows;
	int colNumber = hsv.cols;
	int *dataHist = hist.ptr<int>(0);

	for (int i = 0; i < rowNumber; i++)
	{
		uchar *dataBackProjTmp = backProjTmp.ptr<uchar>(i);
		uchar *dataHsv = hsv.ptr<uchar>(i);

		for (int j = 0; j < colNumber; j++)
			dataBackProjTmp[j] = dataHist[dataHsv[numberChanel*j] / 18];
	}
	backProj = backProjTmp.clone();
}

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

//#include<opencv2/opencv.hpp>
//#include<opencv2/highgui.hpp>
//#include <emmintrin.h>
//using namespace cv;
//#include<time.h>
//void mat_mul_pl(Mat&_mat1, Mat & _mat2, Mat & _mat3);
//int main()
//{
//	Mat mat1 = imread("1.jpg",0);
//	Mat mat2 = imread("1.jpg",0);
//	mat1.convertTo(mat1, CV_32FC1);
//	mat2.convertTo(mat2,CV_32FC1);
//	Mat mat3 = Mat::zeros(mat1.size(),CV_32FC1);
//
//	clock_t t1 = clock();
//	mat_mul_pl(mat1, mat2, mat3);
//	clock_t t2 = clock();
//	imshow("mat1", mat1);
//	imshow("mat2", mat2);
//	imshow("mat3", mat3);
//	std::cout << t2 - t1 << std::endl;
//	waitKey(0);
//	
//	return 0;
//}
//
////void mat_mul_pl(Mat&_mat1, Mat & _mat2, Mat & _mat3)
////{
////	
////	for (int x = 0; x < _mat1.rows; x++)
////	{
////		uchar* data1= _mat1.ptr<uchar>(x);
////		uchar* data2 = _mat2.ptr<uchar>(x);
////		float* data3 = _mat3.ptr<float>(x);
////
////			for (int y = 0; y < _mat1.cols; y++)
////			{
////				
////				data3[3*y] = data1[3*y] * data2[3*y]+0.5f;
////				data3[3 * y + 1] = data1[3 * y + 1] * data2[3 * y + 1]+0.5f;
////				data3[3 * y + 2] = data1[3 * y + 2] + data2[3 * y + 2]+0.5f;
////			}
////	}
////}
//
//void mat_mul_pl(Mat&_mat1, Mat & _mat2, Mat & _mat3)
//{
//	__m128i xidSum = _mm_setzero_si128();
//	__m128i xidLoad1;    // 加载
//	__m128i xidLoad2;
//	__m128i xidLoad3;
//
//	for (int x = 0; x < _mat1.rows; x++)
//	{
//		float* data1 = _mat1.ptr<float>(x);
//		float* data2 = _mat2.ptr<float>(x);
//		float* data3 = _mat3.ptr<float>(x);
//
//
//	for (int y = 0; y < _mat1.cols; ++y)
//	{
//
//		__m128i* p1 = (__m128i*)data1;
//		__m128i* p2 = (__m128i*)data2;
//		__m128i* p3 = (__m128i*)data3;
//
//		xidLoad1 = _mm_load_si128(p1[y]);    // [SSE2] 加载
//		xidLoad2= _mm_load_si128(p2[y]);
//		xidLoad3 = _mm_load_si128(p3);
//		xidSum = _mm_mul_epi32(xidLoad1, xidLoad2);    // [SSE2] 带符号32位紧缩加法
//		xidLoad3 = _mm_mul_epi32(xidLoad1, xidLoad2);
//		++p1;
//		++p2; 
//		++p3;
//	}
//}