//制作标签(这是一个.cpp)
#include <stdio.h>
int main(void)
{
	FILE *fp;
	int i, j, k=0 ,n,f=0,u= 0; int a[2450], b[2450] = { 0 };
	fp = fopen("D:\\workspace\\matlab\\rawFR_train.txt", "w");//打开文件以便写入数据D:\workspace\matlab\FR_RF\rag_train.txt
	for (j = 0; j <= 48; j++)
	{ //准备要写入文件的数组

		for (i = 1; i <= 50; i++)
		{
			a[k] = j;
			k++;
		}

	}
	k = 0;
	for (n = 0; n < 49; n++)
	{
		
		for (i = 0; i < 50; i++)
		{
			
			fprintf(fp, "%d.jpg ", (f + 1));
			//将a数组中的整数写入fp指向的c:\a.txt文件
			fprintf(fp, "%d\n", a[k]); 
			f++;
			k++;
		}
		f += 10;
		
	}
		fclose(fp); //写入完毕，关闭文件

					//读取txt文件，不是必要的
		fp = fopen("D:\\workspace\\matlab\\rawFR_train.txt", "r");

	for (i = 0; i < 10; i++)
	{ //从fp指向的文件中读取10个整数到b数组
		fscanf(fp, "%d", &b[i]);
	}
	fclose(fp); //读取完毕，关闭文件

	for (i = 0; i < 10; i++)
	{ //输出从fp文件读取的10个整数。
		printf("%d\n", b[i]);
	}
	return 0;
}//media/liuyu/_data2/LSTM/fr  51  60     111    120
