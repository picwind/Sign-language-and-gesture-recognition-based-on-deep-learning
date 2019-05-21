#include<iostream>
#include<opencv2/opencv.hpp>
#include <iomanip>
#include <sstream>

using namespace std;
using namespace cv;

int main()
{
	int i;
	Mat src = imread("001.jpg");
	CvVideoWriter *writer = 0;
	int isColor = 1;
	int fps = 10;
	int frameWidth = src.cols;
	int frameHeight = src.rows;
	for (i = 1; i <= 2940; i++)
	{
		stringstream videopath;
		videopath << "D://大创数据集/T&T/" << std::setfill('0') << std::setw(5) << i << ".avi";
		string trans;
		videopath >> trans;
		writer = cvCreateVideoWriter(trans.c_str(), CV_FOURCC('D', 'I', 'V', 'X'),
			fps, cvSize(frameWidth, frameHeight), isColor);
		int n;
		IplImage* img = 0;
		for (n = 0; n<33; n++) {

			char tmpName[300];
			sprintf_s(tmpName, "D://大创数据集/mydada/%05d/%03d.jpg", i, n);
			cout << tmpName << endl;
			img = cvLoadImage(tmpName);

			if (img == NULL)continue;

			cvWriteFrame(writer, img);      // add the frame to the file

			cvReleaseImage(&img);
		}
		cvReleaseVideoWriter(&writer);
	}
	return 0;
}
