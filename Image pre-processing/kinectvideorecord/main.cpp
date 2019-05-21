#include <iostream>
#include <vector>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <Kinect.h>

using namespace std;
using namespace cv;
template<class Interface>
inline void SafeRelease(Interface *& pInterfaceToReleasr) {
	if (pInterfaceToReleasr != NULL) {
		pInterfaceToReleasr->Release();
		pInterfaceToReleasr = NULL;
	}
}

// 转换depth图像到cv::Mat  
/*cv::Mat ConvertMat(const UINT16* pBuffer, int nWidth, int nHeight, USHORT nMinDepth, USHORT nMaxDepth)
{
cv::Mat img(nHeight, nWidth, CV_8UC3);
uchar* p_mat = img.data;

const UINT16* pBufferEnd = pBuffer + (nWidth * nHeight);

while (pBuffer < pBufferEnd)
{
USHORT depth = *pBuffer;

BYTE intensity = static_cast<BYTE>((depth >= nMinDepth) && (depth <= nMaxDepth) ? (depth % 256) : 0);

*p_mat = intensity;
p_mat++;
*p_mat = intensity;
p_mat++;
*p_mat = intensity;
p_mat++;

++pBuffer;
}
return img;
}*/

cv::Mat ConvertMat(const UINT16* pBuffer, int nWidth, int nHeight, USHORT nMinDepth, USHORT nMaxDepth)
{
	cv::Mat img(nHeight, nWidth, CV_8UC3);
	uchar* p_mat = img.data;

	const UINT16* pBufferEnd = pBuffer + (nWidth * nHeight);

	while (pBuffer < pBufferEnd)
	{
		USHORT depth = *pBuffer;

		double intensity = static_cast<double>((depth >= nMinDepth) && (depth <= nMaxDepth) ? depth : 0);

		double min_intensity = double(nMinDepth);
		double max_intensity = double(nMaxDepth);
		uchar intensity_255 = (uchar)(255 * (intensity - min_intensity) / (max_intensity - min_intensity + 0.000000001));

		*p_mat = intensity_255;
		p_mat++;
		*p_mat = intensity_255;
		p_mat++;
		*p_mat = intensity_255;
		p_mat++;

		++pBuffer;
	}
	return img;
}

// 转换color图像到cv::Mat  
cv::Mat ConvertMat(const RGBQUAD* pBuffer, int nWidth, int nHeight)
{
	cv::Mat img(nHeight, nWidth, CV_8UC3);
	uchar* p_mat = img.data;

	const RGBQUAD* pBufferEnd = pBuffer + (nWidth * nHeight);

	while (pBuffer < pBufferEnd)
	{
		*p_mat = pBuffer->rgbBlue;
		p_mat++;
		*p_mat = pBuffer->rgbGreen;
		p_mat++;
		*p_mat = pBuffer->rgbRed;
		p_mat++;

		++pBuffer;
	}
	return img;
}

//16位深度图和红外图
cv::Mat ConvertMat(const UINT16* pBuffer, int nWidth, int nHeight)
{
	cv::Mat img(nHeight, nWidth, CV_16UC1);
	UINT16* p_mat = (UINT16*)img.data;

	const UINT16* pBufferEnd = pBuffer + (nWidth * nHeight);

	while (pBuffer < pBufferEnd)
	{
		*p_mat = *pBuffer;
		p_mat++;
		++pBuffer;
	}
	return img;
}

//8位红外图
cv::Mat ConvertMat(const UINT16* pBuffer, int nWidth, int nHeight, int flag)
{
	cv::Mat img(nHeight, nWidth, CV_8UC1);
	uchar* p_mat = img.data;

	const UINT16* pBufferEnd = pBuffer + (nWidth * nHeight);

	while (pBuffer < pBufferEnd)
	{
		USHORT infrared = *pBuffer;
		double intensity = static_cast<double>(infrared / 255);

		uchar intensity_255 = (uchar)(intensity);

		*p_mat = intensity_255;
		p_mat++;
		++pBuffer;
	}
	return img;
}


int main()
{
	////////////////////////////////////////////////////////////////  
	int depth_width = 512; //depth图像、  
	int depth_height = 424;
	int color_width = 1920; //color图像
	int color_height = 1080;
	int infrared_width = 512; //infrared图像
	int infrared_height = 424;

	cv::Mat depthImg_show = cv::Mat::zeros(depth_height, depth_width, CV_8U);//  
	cv::Mat depthImg = cv::Mat::zeros(depth_height, depth_width, CV_16UC1);//the depth image  
	cv::Mat colorImg = cv::Mat::zeros(color_height, color_width, CV_8UC3);//the color image
	cv::Mat colorImg_resize = cv::Mat::zeros(360, 640, CV_8UC3);//the color image
	cv::Mat infraredImg = cv::Mat::zeros(infrared_height, infrared_width, CV_16UC1);//the infrared image
	cv::Mat infraredImg_show = cv::Mat::zeros(infrared_height, infrared_width, CV_8UC1);//the infrared image for show
																						// Current Kinect  
	IKinectSensor* m_pKinectSensor = NULL;
	// Depth reader  
	IDepthFrameReader*  m_pDepthFrameReader = NULL;
	// Color reader  
	IColorFrameReader*  m_pColorFrameReader = NULL;
	RGBQUAD* m_pColorRGBX = new RGBQUAD[color_width * color_height];
	//Infrared reader
	IInfraredFrameReader  * m_pInfraredFrameReader = NULL;

	HRESULT hr;
	hr = GetDefaultKinectSensor(&m_pKinectSensor);
	if (FAILED(hr))
	{
		cout << "Can not find the Kinect!" << endl;
		cv::waitKey(0);
		exit(0);
	}

	if (m_pKinectSensor)
	{
		// Initialize the Kinect   
		hr = m_pKinectSensor->Open();

		//get the depth reader
		IDepthFrameSource* pDepthFrameSource = NULL;
		if (SUCCEEDED(hr))
		{
			hr = m_pKinectSensor->get_DepthFrameSource(&pDepthFrameSource);
		}
		if (SUCCEEDED(hr))
		{
			hr = pDepthFrameSource->OpenReader(&m_pDepthFrameReader);
		}
		SafeRelease(pDepthFrameSource);

		// for color  
		IColorFrameSource* pColorFrameSource = NULL;
		if (SUCCEEDED(hr))
		{
			hr = m_pKinectSensor->get_ColorFrameSource(&pColorFrameSource);
		}
		if (SUCCEEDED(hr))
		{
			hr = pColorFrameSource->OpenReader(&m_pColorFrameReader);
		}
		SafeRelease(pColorFrameSource);

		//infrared
		IInfraredFrameSource* pInfraredFrameSource = NULL;

		hr = m_pKinectSensor->Open();

		if (SUCCEEDED(hr))
		{
			hr = m_pKinectSensor->get_InfraredFrameSource(&pInfraredFrameSource);
		}

		if (SUCCEEDED(hr))
		{
			hr = pInfraredFrameSource->OpenReader(&m_pInfraredFrameReader);
		}
		SafeRelease(pInfraredFrameSource);
	}

	//verify the depth reader  
	if (!m_pDepthFrameReader)
	{
		cout << "Can not find the m_pDepthFrameReader!" << endl;
		cv::waitKey(0);
		exit(0);
	}
	//verify the color reader  
	if (!m_pDepthFrameReader)
	{
		cout << "Can not find the m_pColorFrameReader!" << endl;
		cv::waitKey(0);
		exit(0);
	}
	//infrared
	if (!m_pInfraredFrameReader)
	{
		cout << "Can not find the m_pInfraredFrameReader!" << endl;
		cv::waitKey(0);
		exit(0);
	}

	// get the data!  
	UINT nBufferSize_depth = 0;
	UINT16 *pBuffer_depth = NULL;
	UINT nBufferSize_color = 0;
	RGBQUAD *pBuffer_color = NULL;
	UINT nBufferSize_infrared = 0;
	UINT16 *pBuffer_infrared = NULL;

	char key = 0;

	int num = 0;

	VideoWriter writer_depth;
	VideoWriter writer_color;
	VideoWriter writer_infrared;
	string outputfile_depth = "D:/data/depth/1.avi";
	string outputfile_color = "D:/data/color/1.avi";
	string outputfile_infrared = "D:/data/infrared/1.avi";

	int fourcc = 2;
	double rate = 10;  //帧率
	writer_depth.open(outputfile_depth, fourcc, rate, Size(depth_width, depth_height));
	writer_color.open(outputfile_color, fourcc, rate, Size(640, 360));
	writer_infrared.open(outputfile_infrared, fourcc, rate, Size(infrared_width, infrared_height));

	while (true) //  
	{
		//depth
		IDepthFrame* pDepthFrame = NULL;
		HRESULT hr = m_pDepthFrameReader->AcquireLatestFrame(&pDepthFrame);
		if (SUCCEEDED(hr))
		{
			USHORT nDepthMinReliableDistance = 0;
			USHORT nDepthMaxReliableDistance = 0;
			if (SUCCEEDED(hr))
			{
				hr = pDepthFrame->get_DepthMinReliableDistance(&nDepthMinReliableDistance);
			}

			if (SUCCEEDED(hr))
			{
				hr = pDepthFrame->get_DepthMaxReliableDistance(&nDepthMaxReliableDistance);
			}
			if (SUCCEEDED(hr))
			{
				hr = pDepthFrame->AccessUnderlyingBuffer(&nBufferSize_depth, &pBuffer_depth);
				//depthImg=ConvertMat(pBuffer_depth, depth_width, depth_height);
				depthImg_show = ConvertMat(pBuffer_depth, depth_width, depth_height, nDepthMinReliableDistance, nDepthMaxReliableDistance);
			}
		}
		SafeRelease(pDepthFrame);

		//color  
		IColorFrame* pColorFrame = NULL;
		hr = m_pColorFrameReader->AcquireLatestFrame(&pColorFrame);
		ColorImageFormat imageFormat = ColorImageFormat_None;
		if (SUCCEEDED(hr))
		{
			ColorImageFormat imageFormat = ColorImageFormat_None;
			if (SUCCEEDED(hr))
			{
				hr = pColorFrame->get_RawColorImageFormat(&imageFormat);
			}
			if (SUCCEEDED(hr))
			{
				hr = pColorFrame->get_RawColorImageFormat(&imageFormat);
			}
			if (SUCCEEDED(hr))
			{
				if (imageFormat == ColorImageFormat_Bgra)//
				{
					hr = pColorFrame->AccessRawUnderlyingBuffer(&nBufferSize_color, reinterpret_cast<BYTE**>(&pBuffer_color));
				}
				else if (m_pColorRGBX)
				{
					pBuffer_color = m_pColorRGBX;
					nBufferSize_color = color_width * color_height * sizeof(RGBQUAD);
					hr = pColorFrame->CopyConvertedFrameDataToArray(nBufferSize_color, reinterpret_cast<BYTE*>(pBuffer_color), ColorImageFormat_Bgra);
				}
				else
				{
					hr = E_FAIL;
				}
				colorImg = ConvertMat(pBuffer_color, color_width, color_height);
				cv::resize(colorImg, colorImg_resize, cv::Size(640, 360), 0, 0, INTER_CUBIC);
			}

			SafeRelease(pColorFrame);
		}

		//infrared
		IInfraredFrame* pInfraredFrame = NULL;
		hr = m_pInfraredFrameReader->AcquireLatestFrame(&pInfraredFrame);
		if (SUCCEEDED(hr))
		{
			if (SUCCEEDED(hr))
			{
				hr = pInfraredFrame->AccessUnderlyingBuffer(&nBufferSize_infrared, &pBuffer_infrared);
				//infraredImg=ConvertMat(pBuffer_infrared, infrared_width, infrared_height);
				infraredImg_show = ConvertMat(pBuffer_infrared, infrared_width, infrared_height, 1);
			}
			SafeRelease(pInfraredFrame);
		}

		cv::imshow("depth", depthImg_show);
		cv::imshow("color", colorImg_resize);
		cv::imshow("infrared", infraredImg_show);

		num++;

		if (num % 50 == 0)
			cout << num << endl;

		writer_depth.write(depthImg_show);
		writer_color.write(colorImg_resize);
		writer_infrared.write(infraredImg_show);

		/*char path_depth[300];
		char path_color[300];
		char path_infrared[300];
		sprintf(path_depth,"D://data/frames/depth/%d.jpg",num);
		sprintf(path_color,"D://data/frames/color/%d.jpg",num);
		sprintf(path_infrared,"D://data/frames/infrared/%d.jpg",num);
		cv::imwrite(path_depth,depthImg_show);
		cv::imwrite(path_color,colorImg);
		cv::imwrite(path_infrared,infraredImg_show);*/

		/*stringstream str_depth;
		stringstream str_color;
		stringstream str_infrared;
		str_depth << "D://data/frames/depth/" << std::setw(3) << std::setfill('0') << num << ".jpg";
		str_color << "D://data/frames/color/" << std::setw(3) << std::setfill('0') << num << ".jpg";
		str_infrared << "D://data/frames/infrared/" << std::setw(3) << std::setfill('0') << num << ".jpg";
		cv::imwrite(str_depth.str(),depthImg_show);
		cv::imwrite(str_color.str(),colorImg_resize);
		cv::imwrite(str_infrared.str(),infraredImg_show);*/

		key = cv::waitKey(1);
		if (key == 27)
			break;
	}

	if (m_pColorRGBX)
	{
		delete[] m_pColorRGBX;
		m_pColorRGBX = NULL;
	}
	// close the Kinect Sensor  
	if (m_pKinectSensor)
	{
		m_pKinectSensor->Close();
	}
	SafeRelease(m_pKinectSensor);

	return 0;

}
