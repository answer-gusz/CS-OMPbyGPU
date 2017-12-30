#include <fstream>
#include<iostream>
#include<assert.h>
#include<vector>
using namespace std;

float** DataRead2(int SizeRow, int SizeCol, char*name, int blocknum, int blockdim)
{
	//��ȡҪ���ļ����ļ���
	char matrixname[40];
	memcpy(matrixname, name, 40);
	FILE *outfile;
	//printf("�������ļ�����");
	//gets(name);
	//�������ķ�ʽ��ȡ�������ļ�
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
	}  //�Ӵ����ļ���������,˳������a������,���ж�ȡ,ע��matlab������˴����ݴ洢�����Ĳ�ͬ

	//�ر���
	infile.close();
	//for (int i = 0; i < blocknum; i++){
	//	for (int j = 0; j < blockdim; j++){
	//		cout << a[i][j] ;

	//	}
	//}
	return a;
}