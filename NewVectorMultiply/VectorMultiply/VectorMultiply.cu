#include <cublas_v2.h> //cuda�Դ��⺯��  
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


//����ѭ������
#define K 3
//����ṹ��洢�ڻ�����ֵ���ı���
struct Max{
	int colnum;
	float maxvector[3];
};
//����ת��֮�������������༭
#define SSizeRow 1
#define SSizeCol 15309
#define ASizeRow 26691
#define ASizeCol 15309
#define blockinclu 1024.0
int main(void)
{ 
	//float blockinclu = 1000; //ÿ�麬�У�������
	int blocknuma = ceil(ASizeRow / blockinclu);//�ܹ��ֳɵĿ���
	int blockdima = ASizeCol*(int)blockinclu;//ÿ���е�����
	int blocknums = SSizeRow ;//�ܹ��ֳɵĿ���
	int blockdims = SSizeCol;//ÿ���е�����
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
	//����S����Ĵ���
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
		//�������CPU������GPU��
		cout << "SCMatrix GPU Malloc Done!" << endl;
	
		HandleError(cudaMemcpy(d_s, vecs.data(), blockdims* sizeof(complex<float>), cudaMemcpyHostToDevice));
		cout << "SCMatrix GPU Memcpy Done!" << endl;
    //����A����Ĵ���
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
		//gpu��Ϊ��������ڴ�
		HandleError(cudaMalloc((void**)&d_a, blockdima * sizeof(complex<float>)));
		HandleError(cudaMalloc((void**)&d_r, (int)blockinclu * sizeof(complex<float>)));
		cout << "ACMatrix GPU Malloc Done!" << endl;
		HandleError(cudaMemcpy(d_a, veca.data(), blockdima * sizeof(complex<float>), cudaMemcpyHostToDevice));
		HandleError(cudaMemset(d_r, 0, (int)blockinclu * sizeof(complex<float>)));
		cout << "ACMatrix GPU Memcpy Done!" << endl;

		//����cublasSgemm�������ȴ������
		//cublassgemm�������� c=alpha*op(a)*op(b)+beta*c
		cublasHandle_t handle;
		cublasCreate(&handle);
		
		cuComplex alpha ;
		cuComplex beta;
		alpha.x = 1.0;
		alpha.y = 1.0;
		beta.x = 0.0;
		beta.y = 0.0;
		cout << "referenc set done!" << endl;
		//��һ�������Ǿ�����ڶ�����������˼��������ԭʼ���󣬵����������Cת�þ�������У�������������AB���еĲ���
		//�ڰ˸������ǲ���������Ҳ���󣬵ھŸ������Ǹþ���ת�ú������
		//��ʮ�������ǲ�������������󣬵�ʮһ�������Ǹþ���ת�ú������
		//��ʮ���������������õ��ľ��󣬵�ʮ�ĸ������Ǹý������ת�ú������
		cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, SSizeRow, ASizeRow,SSizeCol , &alpha, d_s, SSizeCol, d_a, ASizeRow, &beta, d_r, SSizeCol);
		for (int i = 0; i < 10; i++){
			cout << d_r[i] << " ";
			cout << endl;
		}
		//������õĽ����GPU������CPU
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
		//������ڻ�����ֵ��������λ�ú��������
	}

	Max MaxVector;
		//MaxVector.colnum = MaxVectorComp(h_C, sizeof(h_C));
	
	//for (int i = 0; i < BSize; i++){
	//	MaxVector.maxvector[i] = p_A[MaxVector.colnum][i];//A������������������㣬������ǰ�Ѿ���Aת�ã�
		//����ʵ����ȡ�����������
	//}

	//�ͷ��ڴ�
	//delete[]points;
	//getchar();
	delete[]pointimaga;
	delete[]pointreala;
	delete[]pointimags;
	delete[]pointreals;
	getchar();
	//��ӡ���
	//	for (int i = 0; i<CSize; i++)
	//{
	//		printf("C[%d] = %f\n", i, h_C[i]);
	//	
	//}
	//	printf("MaxVector.colnum = %d\n", MaxVector.colnum);
	//	printf("MaxVector.maxvector[0] = %f\n", MaxVector.maxvector[0]);
	//	printf("MaxVector.maxvector[1] = %f\n", MaxVector.maxvector[1]);
	//getchar();
	////�ͷ���GPU�Ϸ�����ڴ�
	//cudaFree(d_a);
	//cudaFree(d_b);
	//cudaFree(d_c);
	getchar();
	return 0;
}