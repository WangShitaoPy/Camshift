/*#include <opencv2/core/utility.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>
#include <ctype.h>

using namespace cv;
using namespace std;
Mat image;

bool selectObject = false;
int trackObject = 0;
Point origin;
Rect selection;

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
	case EVENT_LBUTTONDOWN:
		origin = Point(x, y);
		selection = Rect(x, y, 0, 0);
		selectObject = true;
		break;
	case EVENT_LBUTTONUP:
		selectObject = false;
		if (selection.width > 0 && selection.height > 0)
			trackObject = -1;
		break;
	}
}

Mat mImage;
int main(int argc, const char** argv)
{
	VideoCapture cap;
	Mat frame;

	namedWindow("CamShift Demo", 0);
	setMouseCallback("CamShift Demo", onMouse, 0);

	cap.open(0);
	if (!cap.isOpened())
	{
		cout << "***Could not initialize capturing...***\n";
		cout << "Current parameter's value: \n";
		return -1;
	}
	for (;;)
	{
		cap >> frame;
		if (frame.empty())
		{
			break;
		}
		frame.copyTo(image);
		imshow("CamShift Demo", image);
		waitKey(50);
	}
	return 0;
}*/

#include <opencv2/core/utility.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>
#include <ctype.h>

using namespace cv;
using namespace std;

void calcHistOfHsv(Mat &hsv, Mat &hist);
void bgr_to_hsv(Mat & _src, Mat & _dst);
void calcBackProj(Mat &hsv, Mat &hist, Mat &backProj);

Mat image;
Mat selectMat;

bool backprojMode = false;
bool selectObject = false;
int trackObject = 0;
bool showHist = true;
Point origin;
Rect selection(100,100,25,25);
int vmin = 10, vmax = 256, smin = 30;

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
	case EVENT_LBUTTONDOWN:
		origin = Point(x, y);
		selection = Rect(x, y, 0, 0);
		selectObject = true;
		break;
	case EVENT_LBUTTONUP:
		selectObject = false;
		if (selection.width > 0 && selection.height > 0)
			trackObject = -1;
		break;
	}
}

string hot_keys =
"\n\nHot keys: \n"
"\tESC - quit the program\n"
"\tc - stop the tracking\n"
"\tb - switch to/from backprojection view\n"
"\th - show/hide object histogram\n"
"\tp - pause video\n"
"To initialize tracking, select the object with mouse\n";

static void help()
{
	cout << "\nThis is a demo that shows mean-shift based tracking\n"
		"You select a color objects such as your face and it tracks it.\n"
		"This reads from video camera (0 by default, or the camera number the user enters\n"
		"Usage: \n"
		"   ./camshiftdemo [camera number]\n";
	cout << hot_keys;
}

const char* keys =
{
	"{help h | | show help message}{@camera_number| 0 | camera number}"
};

int main(int argc, const char** argv)
{
	VideoCapture cap;
	Rect trackWindow;
	int hsize = 16;
	float hranges[] = { 0, 180 };
	const float* phranges = hranges;
	CommandLineParser parser(argc, argv, keys);
	if (parser.has("help"))
	{
		help();
		return 0;
	}
	int camNum = parser.get<int>(0);
	cap.open(camNum);

	if (!cap.isOpened())
	{
		help();
		cout << "***Could not initialize capturing...***\n";
		cout << "Current parameter's value: \n";
		parser.printMessage();
		return -1;
	}
	cout << hot_keys;
	namedWindow("Histogram", 0);
	namedWindow("CamShift Demo", 0);
	setMouseCallback("CamShift Demo", onMouse, 0);
	createTrackbar("Vmin", "CamShift Demo", &vmin, 256, 0);
	createTrackbar("Vmax", "CamShift Demo", &vmax, 256, 0);
	createTrackbar("Smin", "CamShift Demo", &smin, 256, 0);

	Mat frame, hsv, hue, mask, hist, histimg = Mat::zeros(200, 320, CV_8UC3), backproj;
	bool paused = false;

	for (;;)
	{
		if (!paused)
		{
			cap >> frame;
			if (frame.empty())
				break;
		}

		frame.copyTo(image);

		if (!paused)
		{
			cvtColor(image, hsv, COLOR_BGR2HSV);
			bgr_to_hsv(image, hsv);

			hue.create(hsv.size(), hsv.depth());
			int ch[] = { 0, 0 };
			mixChannels(&hsv, 1, &hue, 1, ch, 1);
		
			calcHistOfHsv(hsv(selection), hist);
			normalize(hist, hist, 0, 255, NORM_MINMAX);

			calcBackProj(hsv, hist, backproj);
		}
	}

	return 0;
}

/*hsv 为要计算直方图的图像矩阵*/
void calcHistOfHsv(Mat &hsv, Mat &hist)
{
	Mat histTmp = Mat::zeros(1,20,CV_8UC1);
	uchar *dataHist = histTmp.ptr<uchar>(0);
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
	Mat backProjTmp = Mat::zeros(hsv.size(),CV_8UC1);
	//Mat backProjTmp;
	//Mat backProjTmp = Mat::ones(hsv.size(), CV_8UC1);
	int numberChanel;
	numberChanel = hsv.channels();
	backProjTmp.create(hsv.size(), 1);

	int rowNumber = hsv.rows;
	int colNumber = hsv.cols;
	uchar *dataHist = hist.ptr<uchar>(0);

	for (int i = 0; i < rowNumber; i++)
	{
		uchar *dataBackProjTmp = backProjTmp.ptr<uchar>(i);
		uchar *dataHsv = hsv.ptr<uchar>(i);
		
		for (int j = 0; j < colNumber; j++)
		{
			uchar histNum = dataHsv[numberChanel*j] / 18;
			dataBackProjTmp[j] = dataHist[histNum];
		}
	}

	backProj = backProjTmp.clone();
}

void bgr_to_hsv(Mat & _src, Mat & _dst)
{
	float h, s, v;
	float b, g, r;

	int rowNum = _src.rows;
	int colNum = _src.cols;

	_dst.create(rowNum, colNum, CV_8UC3);

	for (int i = 0; i < rowNum; i++)
	for (int j = 0; j < colNum; j++)
	{
		b = _src.at<Vec3b>(i, j)[0];
		g = _src.at<Vec3b>(i, j)[1];
		r = _src.at<Vec3b>(i, j)[2];
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
		h /= 2;
		_dst.at<Vec3b>(i, j)[0] = h;
		_dst.at<Vec3b>(i, j)[1] = s * 255;
		_dst.at<Vec3b>(i, j)[2] = v * 255;
	}
}

