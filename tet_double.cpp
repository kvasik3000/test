#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <iomanip>
// using namespace ::std;

double pi = acos(-1);

void full_array(double *arr, long long siz)
{
	#pragma data copyin(sum)
#pragma acc kernels present(arr [0:siz])
	for (int i = 0; i < siz; i++)
	{
		arr[i] = sin(2 * pi * i / siz);
	}
}

double sum_elem(double *arr, long long siz)
{
	double sum = 0;
#pragma data copy(sum)
#pragma acc parallel loop present(arr [0:siz]) reduction(+: sum)
	for (int i = 0; i < siz; i++)
	{
		sum += arr[i];
	}
	return sum;
}

void out_elem(double *arr, long long siz)
{
		
#pragma acc exit data copyout(arr [0:siz])
	for (int i = 0; i < siz; i += 10000)
	{
		std::cout << std::setprecision(20) << arr[i] << std::endl;
	}
}

int main()
{
	auto start = std::chrono::high_resolution_clock::now();
	double *arr = new double[10000000];
	double sum = 0;
	long long size = 10000000;
	
#pragma acc enter data create(arr [0:size])
	

		full_array(arr, size);
		sum = sum_elem(arr, size);

	std::cout << "sum = "<< std::setprecision(20) << sum << std::endl;
#pragma acc exit data delete(arr [0:size])
	delete[] arr;
	auto elapsed = std::chrono::high_resolution_clock::now() - start;
	long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << microseconds << std::endl;
}