int MaxVectorComp(float *result, int size)
{

	float max = *result;
	int j = 0;
	for (int i = 1; i < size; i++){
		if (max < result[i]){
			max = result[i];
			j = i;
		}
	}
	return j;
}