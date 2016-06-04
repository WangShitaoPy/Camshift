工程名：Camshift
工程构架：
	Camshift
		|-->README.txt
		|-->bgr_to_hsv.cpp
		|-->

函数功能简介：
	bgr_to_hsv.cpp
		输入：Mat & _src:当前帧图像引用
	  	      Mat & _dst:输出HSV图像
		注：需要满足单通道和多通道

	void calcHistOfHsv(Mat &hsv, Mat &hist)
		输入：hsv:目标hsv矩阵，0通道为H值
	      	      hist:直方图输出	
		注：需要满足单通道和多通道

	void calcBackProj(Mat &hsv, Mat &hist, Mat &backProj)
		输入：hsv:hsv矩阵，0通道H值
    		      hist:直方图
		backProj:反向投影
		注：需要满足单通道和多通道

	void meanshift(Mat & _img，Rect & _box)
		输入：_img当前视频帧的hsv图像,_box更新窗口
		注：需要满足单通道和多通道

