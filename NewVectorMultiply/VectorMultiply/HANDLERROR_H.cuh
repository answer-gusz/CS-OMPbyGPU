#include <cuda_runtime.h>
#include<iostream>

using namespace std;
static void HandleError(cudaError_t err)

{

	if (err != cudaSuccess)

	{

		cout << "ERROR HERE��" << endl;

		cout << "�������Ϊ��" << cudaGetErrorString(err) << endl;

	}

}