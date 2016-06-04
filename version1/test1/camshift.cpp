#include <iostream>
#include <ctype.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/videoio.hpp>
using namespace cv;
using namespace std;
bool selectObject = false;
bool trackObject = false;
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
			trackObject = true;
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
	Mat  hsv= Mat::zeros(frame.size(), CV_8UC3), hist = Mat::zeros(1, 20, CV_8UC1), backproj;
	while (cap.read(frame))
	{
		image = frame.clone();

		if (selectObject)
		{
			rectangle(image, selection, Scalar(255, 0, 0), 3, 8);
			bgr_to_hsv(image, hsv);
			calcHistOfHsv(hsv(selection), hist);
			
		}
		if (trackObject)
		{
			cvtColor(image, hsv, COLOR_BGR2HSV);
			bgr_to_hsv(image, hsv);
//			calcHistOfHsv(hsv(selection), hist);
			calcBackProj(hsv, hist, backproj);
			imshow("backproj", backproj);
		//	meanshift(backproj, selection);
			Rect trackWindow = selection;
			
			RotatedRect trackBox = CamShift(backproj, trackWindow,
				TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 1));
			if (trackWindow.area() <= 1)
			{
				int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5) / 6;
				trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
					trackWindow.x + r, trackWindow.y + r) &
					Rect(0, 0, cols, rows);
			}
			
			rectangle(image, trackWindow, Scalar(255, 0, 0), 3, 8);
		}
		imshow("CamShift Demo", image);
		waitKey(30);
	}
	return 0;
}


void meanshift(Mat & _prob_img, Rect& _box)
{
	Mat roi;
	double m00;
	double m10, m01;
	float temp;
	int detx, dety;
	while (1)
	{
		m00 = 0;
		m01 = 0;
		m10 = 0;
		detx = 0;
		dety = 0;
		roi = Mat(_prob_img, _box);
		for (int x = 0; x < roi.cols; x++)
		{
			uchar* data = roi.ptr<uchar>(x);
			for (int y = 0; y < roi.rows; y++)
			{
				m00 += data[y];
				m10 += x*data[y];
				m01 += y*data[y];
			}
		}
		detx = m10 / m00;
		dety = m01 / m00;
		temp = pow(detx - roi.cols*0.5, 2) + pow(dety - roi.rows*0.5, 2);
		if (temp < 9)
			break;

		_box.width =2*sqrt(m00/256);
		if (_box.width>100)
			_box.width = 100;
		if (_box.width < 10)
			_box.width = 10;
		_box.height = 1.2*_box.width;

		_box.x += detx;
		_box.y += dety;
		_box.x = _box.x - 0.5*_box.width;
		_box.y = _box.y - 0.5*_box.height;
		/*_box.x = _box.x - 0.5*_box.width;
		if (_box.x < 0)
			_box.x = 0;
		if (_box.x+_box.width>_prob_img.cols)
			_box.x=
			(_box.x - 0.5*_box.width,_prob_img.cols 

		_box.y = _box.y - 0.5*_box.height;*/
		
	}
}
/*void calcHistOfHsv(Mat &hsv, Mat &hist)
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
	//normalize(histTmp, histTmp, 0, 255, NORM_MINMAX);
	hist = histTmp.clone();
}

void calcBackProj(Mat &hsv, Mat &hist, Mat &backProj)
{
	Mat backProjTmp=Mat::zeros(hsv.size(),CV_8UC1);
	int numberChanel;
	numberChanel = hsv.channels();

	int rowNumber = hsv.rows;
	int colNumber = hsv.cols;
	int *dataHist = hist.ptr<int>(0);
	int nLoop = hist.cols*hist.channels();

	int maxValue = dataHist[0];
	for (int i = 0; i < nLoop; i++)
	{
		if (dataHist[i]>maxValue)
			maxValue = dataHist[i];
	}

	for (int i = 0; i < rowNumber; i++)
	{
		uchar *dataBackProjTmp = backProjTmp.ptr<uchar>(i);
		uchar *dataHsv = hsv.ptr<uchar>(i);

		for (int j = 0; j < colNumber; j++)
		{
			int value = dataHist[dataHsv[numberChanel*j] / 18];
			dataBackProjTmp[j] = value / maxValue * 255;
		}
	}
	backProj = backProjTmp.clone();
}
*/
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