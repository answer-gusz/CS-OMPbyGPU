#include <cublas_v2.h> //cuda自带库函数  
#include <cuda.h>  
//#include <stdio.h>  
#include<cuda_runtime.h>
#include "HANDLERROR_H.cuh"
#include<cublas_api.h>
#include<math.h>
#include<vector>
#include"DataRead2.cuh"
#include"MaxVectorComp.cuh"
#include<iostream>
#include<complex>
#include<vector>


//定义循环次数
#define K 3
//定义结构体存储内积绝对值最大的变量
struct Max{
	int colnum;
	float maxvector[3];
};
//按照转置之后矩阵的行列数编辑
#define SSizeRow 1
#define SSizeCol 15309
#define ASizeRow 26691
#define ASizeCol 15309
#define blockinclu 1024.0
int main(void)
{ 
	//float blockinclu = 1000; //每块含有（行数）
	int blocknuma = ceil(ASizeRow / blockinclu);//总共分成的块数
	int blockdima = ASizeCol*(int)blockinclu;//每块中的数量
	int blocknums = SSizeRow ;//总共分成的块数
	int blockdims = SSizeCol;//每块中的数量
	char sImagName[40] = "SImagLoopCrossCUDA.dat";
	char sRealName[40] = "SRealLoopCrossCUDA.dat";
	char aImagName[40] = "AImagLoopCrossCUDA.dat";
	char aRealName[40] = "ARealLoopCrossCUDA.dat";
	float **pointreals;
	float **pointimags;
	float **pointreala;
	float **pointimaga;
	pointreals = DataRead2(SSizeRow, SSizeCol, sRealName, blocknums, blockdims);
	pointimags = DataRead2(SSizeRow, SSizeCol, sImagName, blocknums, blockdims);
	cout << "File SMatrix Read Done!" << endl;

	pointreala = DataRead2(ASizeRow, ASizeCol, aRealName, blocknuma, blockdima);
	pointimaga = DataRead2(ASizeRow, ASizeCol, aImagName, blocknuma, blockdima);
	cout << "File AMatrix Read Done!" << endl;
	//关于S矩阵的处理
	//use shared memory
	    int i = 0;
		vector <complex <float> > vecs(blockdims);
		for (int j = 0; j < blockdims; j++){
			vecs[j].real(pointreals[i][j]);
			vecs[j].imag(pointimags[i][j]);
		}
		cout << "SComplexMatrix Done!" << endl;
		cuComplex *d_s;
		HandleError(cudaMalloc((void**)&d_s, blockdims* sizeof(complex<float>)));
		//将数组从CPU拷贝到GPU上
		cout << "SCMatrix GPU Malloc Done!" << endl;
	
		HandleError(cudaMemcpy(d_s, vecs.data(), blockdims* sizeof(complex<float>), cudaMemcpyHostToDevice));
		cout << "SCMatrix GPU Memcpy Done!" << endl;
    //关于A矩阵的处理
	i = 0;
	while (i != blocknuma){

		vector <complex <float> > veca(blockdima);
		for (int j = 0; j < blockdima; j++){
			veca[j].real(pointreala[i][j]);
			veca[j].imag(pointimaga[i][j]);
		}
		cout << "A" << "[" << i << "]" << "ComplexMatrix Done!" << endl;
		i++;
		cuComplex *d_a, *d_r;
		//gpu上为数组分配内存
		HandleError(cudaMalloc((void**)&d_a, blockdima * sizeof(complex<float>)));
		HandleError(cudaMalloc((void**)&d_r, (int)blockinclu * sizeof(complex<float>)));
		cout << "ACMatrix GPU Malloc Done!" << endl;
		HandleError(cudaMemcpy(d_a, veca.data(), blockdima * sizeof(complex<float>), cudaMemcpyHostToDevice));
		HandleError(cudaMemset(d_r, 0, (int)blockinclu * sizeof(complex<float>)));
		cout << "ACMatrix GPU Memcpy Done!" << endl;

		//调用cublasSgemm函数，先创建句柄
		//cublassgemm参数定义 c=alpha*op(a)*op(b)+beta*c
		cublasHandle_t handle;
		cublasCreate(&handle);
		
		cuComplex alpha ;
		cuComplex beta;
		alpha.x = 1.0;
		alpha.y = 1.0;
		beta.x = 0.0;
		beta.y = 0.0;
		cout << "referenc set done!" << endl;
		//第一个参数是句柄，第二三个参数意思是输入是原始矩阵，第四五参数是C转置矩阵的行列，第六个参数是AB共有的参数
		//第八个参数是参与运算的右侧矩阵，第九个参数是该矩阵转置后的行数
		//第十个参数是参与运算的左侧矩阵，第十一个参数是该矩阵转置后的行数
		//第十三个参数是运算后得到的矩阵，第十四个参数是该结果矩阵转置后的行数
		cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, SSizeRow, ASizeRow,SSizeCol , &alpha, d_s, SSizeCol, d_a, ASizeRow, &beta, d_r, SSizeCol);
		for (int i = 0; i < 10; i++){
			cout << d_r[i] << " ";
			cout << endl;
		}
		//将计算好的结果从GPU拷贝回CPU
		cout << "cublasCgemm done!" << endl;
		complex<float> h_C[SSizeCol];
		for (int i = 0; i < 10; i++){
			cout << h_C[i] << " ";
			cout << endl;
		}
		HandleError(cudaMemcpy(h_C, d_r, SSizeCol * sizeof(complex<float>), cudaMemcpyDeviceToHost));
		
		for (int i = 0; i < 10; i++){
			cout << h_C[i] << " ";
			cout << endl;
		}
		cout << "Copy to CPU!" << endl;
		cout << "here" << endl;
		//求最大内积绝对值，并返回位置和这个向量
	}

	Max MaxVector;
		//MaxVector.colnum = MaxVectorComp(h_C, sizeof(h_C));
	
	//for (int i = 0; i < BSize; i++){
	//	MaxVector.maxvector[i] = p_A[MaxVector.colnum][i];//A矩阵的列向量参与运算，在输入前已经将A转置，
		//所以实际是取得行向量标号
	//}

	//释放内存
	//delete[]points;
	//getchar();
	delete[]pointimaga;
	delete[]pointreala;
	delete[]pointimags;
	delete[]pointreals;
	getchar();
	//打印结果
	//	for (int i = 0; i<CSize; i++)
	//{
	//		printf("C[%d] = %f\n", i, h_C[i]);
	//	
	//}
	//	printf("MaxVector.colnum = %d\n", MaxVector.colnum);
	//	printf("MaxVector.maxvector[0] = %f\n", MaxVector.maxvector[0]);
	//	printf("MaxVector.maxvector[1] = %f\n", MaxVector.maxvector[1]);
	//getchar();
	////释放在GPU上分配的内存
	//cudaFree(d_a);
	//cudaFree(d_b);
	//cudaFree(d_c);
	getchar();
	return 0;
}