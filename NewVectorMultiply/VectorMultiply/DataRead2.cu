#include <fstream>
#include<iostream>
#include<assert.h>
#include<vector>
using namespace std;

float** DataRead2(int SizeRow, int SizeCol, char*name, int blocknum, int blockdim)
{
	//获取要打开文件的文件名
	char matrixname[40];
	memcpy(matrixname, name, 40);
	FILE *outfile;
	//printf("请输入文件名：");
	//gets(name);
	//采用流的方式读取二进制文件
	float **a = new float*[blocknum];
	for (int i = 0; i < blocknum; i++){
		a[i] = new float[blockdim];
	}
	assert(a != NULL);

	ifstream infile(matrixname, ios::binary | ios::in);
	if (!infile)
	{
		cerr << "open error!" << endl;
		exit(1);
	}
	for (int j = 0; j < blocknum; j++){
		infile.read((char *)a[j], sizeof(float)*blockdim);
	}  //从磁盘文件读入数据,顺序存放在a数组中,按行读取,注意matlab数据与此处数据存储方法的不同

	//关闭流
	infile.close();
	//for (int i = 0; i < blocknum; i++){
	//	for (int j = 0; j < blockdim; j++){
	//		cout << a[i][j] ;

	//	}
	//}
	return a;
}