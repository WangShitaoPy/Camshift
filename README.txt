��������Camshift
���̹��ܣ�
	Camshift
		|-->README.txt
		|-->bgr_to_hsv.cpp
		|-->

�������ܼ�飺
	bgr_to_hsv.cpp
		���룺Mat & _src:��ǰ֡ͼ������
	  	      Mat & _dst:���HSVͼ��
		ע����Ҫ���㵥ͨ���Ͷ�ͨ��

	void calcHistOfHsv(Mat &hsv, Mat &hist)
		���룺hsv:Ŀ��hsv����0ͨ��ΪHֵ
	      	      hist:ֱ��ͼ���	
		ע����Ҫ���㵥ͨ���Ͷ�ͨ��

	void calcBackProj(Mat &hsv, Mat &hist, Mat &backProj)
		���룺hsv:hsv����0ͨ��Hֵ
    		      hist:ֱ��ͼ
		backProj:����ͶӰ
		ע����Ҫ���㵥ͨ���Ͷ�ͨ��

	void meanshift(Mat & _img��Rect & _box)
		���룺_img��ǰ��Ƶ֡��hsvͼ��,_box���´���
		ע����Ҫ���㵥ͨ���Ͷ�ͨ��

