#include <opencv2/opencv.hpp>
#include<iostream>
#include <iomanip>  //主要是对cin,cout之类的一些操纵运算子，比如setfill,setw,setbase,setprecision等等
#include<string>
#include<fstream>  //文件流

using namespace std;
using namespace cv;

//读取txt文件中的整型数据begin，end
//输入字符串s
//输出整型pdata
void stringTOnum1(string s, int* pdata)
{
	bool temp = false;		//读取一个数据标志位
	int data = 0;			    //分离的一个数据
	int m = 0;				//数组索引值
	for (int i = 0; i<s.length(); i++)
	{
		while ((s[i] >= '0') && (s[i] <= '9'))		//当前字符是数据，并一直读后面的数据，只要遇到不是数字为止
		{
			temp = true;		//读数据标志位置位
			data *= 10;
			data += (s[i] - '0');		//字符在系统以ASCII码存储，要得到其实际值必须减去‘0’的ASCII值
			i++;
		}
		//刚读取了数据
		if (temp)		//判断是否完全读取一个数据
		{
			pdata[m] = data;		//赋值
			m++;
			data = 0;
			temp = false;		//标志位复位
		}
	}
}

//对两幅图片进行加权融合
void blending()
{
	Mat src1, src2, dst;
	double alpha = 0.5;
	double beta = 1 - alpha;
	double gama = 0;

	src1 = imread("010.jpg");
	src2 = imread("011.jpg");
	//判断两幅图片是否相同  
	CV_Assert(src1.depth() == CV_8U);
	CV_Assert(src1.depth() == src2.depth());
	CV_Assert(src1.size() == src2.size());
	//为dst申请内存  
	dst.create(src1.size(), src1.type());

	const int nChannels = src1.channels();//颜色通道数为1

	if (!src1.data) cout << "error loading src1" << endl;
	if (!src2.data) cout << "Error loading src2" << endl;

	for (int i = 0; i < src1.rows; i++)
	{
		const uchar* src1_ptr = src1.ptr<uchar>(i);//取第i行所有像素的值
		const uchar* src2_ptr = src2.ptr<uchar>(i);
		uchar* dst_ptr = dst.ptr<uchar>(i);
		
		//对第i行每个像素的灰度值做加权运算
		for (int j = 0; j < src1.cols*nChannels; j++)
		{
			dst_ptr[j] = src1_ptr[j] * alpha + src2_ptr[j] * beta + gama;
		}
	}
}

	void main(void)
	{

		int* pdata = new int[2];
		string s;

		//读取起始帧txt文件
		ifstream infile1;
		infile1.open("search.txt");//起始帧索引表路径路径
		
		int num = 0;
		while (getline(infile1, s))
		{
			stringTOnum1(s, pdata);
		    num++;
			printf("\n%d\n",num);
			//begin,end,length分别表示起始帧，终止帧和总帧数
			//n表示32帧中第n帧
			int begin = pdata[0], end = pdata[1], length, n = 1;
			length = end - begin + 1;

			printf("%d %d\n", begin, end);

			//线性融合参数alpha,beta分别表示左右两张图片的权值
			//例如：当(length / 32.0)*n=3.8时，处理后的第n帧由原图片第3,4张融合得到
			//      3.8离4较近,于是第3张权值为1-0.8=0.2，第4张权值为0.8
			double alpha, beta;
			for (n = 1; n <= 32; n++)
			{
				//left,right表示对应帧在left和right之间
				int left, right;

				alpha = 1.0 - (length / 32.0)*n + int((length / 32.0)*n);
				beta = 1.0 - alpha;
				left = begin + int((length / 32.0)*n) - 1;
				right = left + 1;
				if (left < begin)
					left = begin;
				if (right > end)
					right = end;

				Mat src1, src2, dst;
				//添加原图片路径
				stringstream l, r;                                     //如 cout<<setfill(‘@‘)<<setw(5)<<255<<endl;结果是:@@255
				l << "E://Imagedata/" << std::setw(5) << std::setfill('0') <<num<<"/"<< std::setw(3) << std::setfill('0') << left << ".jpg";//左边图片路径
				r << "E://Imagedata/" << std::setw(5) << std::setfill('0')<< num<<"/"<< std::setw(3) << std::setfill('0') << right<< ".jpg";//右边图片路径
				src1 = imread(l.str(), CV_LOAD_IMAGE_COLOR);
				src2 = imread(r.str(), CV_LOAD_IMAGE_COLOR);
				//加权运算合成新图像
				addWeighted(src1, alpha, src2, beta, 0.0, dst);

				//生成jpg
				stringstream str;
				str << "E://Newimages/" << std::setw(5) << std::setfill('0') << num<<"/"<< std::setw(3) << std::setfill('0') << n << ".jpg";//合成图片路径
				imwrite(str.str(), dst);

				waitKey(0);
			}
		}
		infile1.close();
	}
