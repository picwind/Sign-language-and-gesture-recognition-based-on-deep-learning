#include<iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
                 int videonum;
                 int imagesnum=1;
                 for (videonum = 1; videonum <= 2940; videonum++)
                {
                                 stringstream videopath; //创建自由的stringstream对象
                                videopath << "F://data/depth/" << std::setw(4) << std::setfill('0' ) << videonum << ".avi";//将"F://data/depth/videonum.avi" 复制给videopath

                                 //Open the video file
                                 VideoCapture capture(videopath.str());//返回videopath中存储的string对象

                                 // check if video successfully opened
                                 if (!capture.isOpened())
                                                 return 1;



                                 // 视频相关信息
                                 double rate = capture.get(CV_CAP_PROP_FPS ); //帧率，即每秒多少帧
                                cout << rate << endl;
                                 long n = static_cast <long>(capture.get( CV_CAP_PROP_FRAME_COUNT));  //帧数
                                cout << n << endl;
                                 long width = static_cast <long>(capture.get( CV_CAP_PROP_FRAME_WIDTH)); //每帧宽度
                                cout << width << endl;
                                 long height = static_cast <long>(capture.get( CV_CAP_PROP_FRAME_HEIGHT)); //每帧高度
                                cout << height << endl;
                                 double time = 1000 * n / rate;  //视频时长，单位毫秒
                                cout << time << endl;

                                 //视频逐帧保存成图像，从position_f这一帧开始,保存其后的N帧
                                 double position_f = 0;
                                capture.set( CV_CAP_PROP_POS_FRAMES, position_f);
                                 int N = 200;
                                 Mat frame; // current video frame
                                                                   //char path[400];
                                 int num = 1;
                                 while (num <= N)
                                {
                                                 if (!capture.read(frame))
                                                                 break;

                                                 //每一帧图片保存路径
                                                 stringstream str;
                                                str << "E://Imagedata/" << std::setw(5) << std::setfill('0') <<imagesnum <<"/" << std::setw(3) << std::setfill('0') << num << ".jpg" ;
                                                
                                                num++;

                                                 //imwrite(path,frame);
                                                imwrite(str.str(), frame);

                                                 if (cvWaitKey(1) == 27)
                                                                 break;
                                }
                                capture.release(); //释放打开的视频空间
                                imagesnum=imagesnum+60;

                                 //当imagesnum加了49次60后初始值加1，即imagesnum = imagesnum - 60*49+1
                                 if (videonum % 49 == 0)
                                                imagesnum = imagesnum - 2048;
                }
                 return 0;
}
