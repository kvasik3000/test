#include <iostream>
#include <cstdlib>
#include<cmath>
#define _USE_MATH_DEFINES
using namespace ::std;

int main() {
	float* arr = new float[10000000];
	double sum = 0;
	const double pi = 3.14159265358979323846;
	for (int i = 0; i < 9999999; i++)
	{
		arr[i] = sin(2* pi * i/9999999) ;
		sum += arr[i];
	}
	cout << sum;

	delete[] arr;
}





