#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <ctime>
#include <string>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

// функция, обновляющая значения сетки (__global__ используется для обозначения функции, которая будет выполняться на устройстве GPU и вызываться из хоста)
__global__ void calculate(double *A, double *Anew, int size)
{
    //blockIdx.x(номер нити в блоке) * blockDim.x(номер блока в котором находится нить) + threadIdx.x(размер блока по оси х);
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (j < size - 1 && j > 0 && i > 0 && i < size - 1)
    {
        double left = A[i * size + j - 1];
        double right = A[i * size + j + 1];
        double top = A[(i - 1) * size + j];
        double mid = A[(i + 1) * size + j];
        Anew[i * size + j] = 0.25 * (left + right + top + mid);
    }
}

// функция нахождения разности двух массивов
__global__ void getDifference(double *A, double *Anew, double *res, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= 0 && i < size && j >= 0 && j < size)
        res[i * size + j] = Anew[i * size + j] - A[i * size + j];
}

int main(int argc, char *argv[])
{
    int n = 0;                   // размер
    int MAX_ITERATION = 0;       // максимальное число итераций
    double ACCURACY = 0;         // точность
    double error = 1.0;          // ошибка
    int cntIteration = 0;        // количество итераций
    size_t tempStorageBytes = 0; // размер выделенной памяти для d_temp_storage
    double *tempStorage = NULL;  // доступное для устройства выделение временного хранилища

    // считываем с командной строки
    for (int arg = 1; arg < argc; arg++)
    {
        if (arg == 2)
            ACCURACY = std::stod(argv[arg]);
        if (arg == 4)
            n = std::stoi(argv[arg]);
        if (arg == 6)
            MAX_ITERATION = std::stoi(argv[arg]);
    }

    // определяем наши шаги по рамкам
    double dt = 10 / ((double)n - 1);

    double *arr = new double[n * n];
    double *arrNew = new double[n * n];

    // инициализация масива и копии массива
    for (int i = 1; i < n * n; i++)
    {
        arr[i] = 0;
        arrNew[i] = 0;
    }

    // заполнение краев рамок нашей матрицы и матрицы клона
    arr[0] = 10.0;
    arr[n - 1] = 20.0;
    arr[n * (n - 1)] = 20.0;
    arr[n * n - 1] = 30.0;
    arrNew[0] = 10.0;
    arrNew[n - 1] = 20.0;
    arrNew[n * (n - 1)] = 20.0;
    arrNew[n * n - 1] = 30.0;

    // заполнение рамок нашей матрицы и матрицы клона
    for (int i = 1; i < n - 1; i++)
    {
        arr[i] = arr[i - 1] + dt;
        arr[i * n + n - 1] = arr[(i - 1) * n + n - 1] + dt;
        arr[i * n] = arr[(i - 1) * n] + dt;
        arrNew[i] = arr[i - 1] + dt;
        arrNew[i * n + n - 1] = arrNew[(i - 1) * n + n - 1] + dt;
        arrNew[i * n] = arrNew[(i - 1) * n] + dt;
    }
    for (int i = 0; i < n - 2; i++)
    {
        arr[n * (n - 1) + i + 1] = arr[n * (n - 1) + i] + dt;
        arrNew[n * (n - 1) + i + 1] = arrNew[n * (n - 1) + i] + dt;
    }

    // выделяем память на gpu через cuda для 3 сеток( cudaMalloc позволяет выделить блок памяти заданного размера на gpu)
    // указатели на которе будут указаны указатели, в который будет записан адрес начала выделенной памяти на устройстве.
    double *CudaArr, *CudaNewArr, *CudaDifArr;
    cudaMalloc((void **)&CudaArr, sizeof(double) * n * n);
    cudaMalloc((void **)&CudaNewArr, sizeof(double) * n * n);
    cudaMalloc((void **)&CudaDifArr, sizeof(double) * n * n);

    // копирование информации с CPU на GPU
    // куда, откуда, размер данных в байтах, тип копирования
    cudaMemcpy(CudaArr, arr, sizeof(double) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(CudaNewArr, arrNew, sizeof(double) * n * n, cudaMemcpyHostToDevice);

    // выделяем память на gpu. Хранение ошибки на device
    double *maxError = 0;
    cudaMalloc((void **)&maxError, sizeof(double));

    // получаем размер временного буфера для редукции
    // tempStorage - указатель на буфер памяти, используемый для временного хранения промежуточных данных и выделения памяти.
    // tempStorageBytes - размер требуемого буфера памяти tempStorage в байтах. Этот размер будет автоматически вычислен и возвращен функцией.
    // CudaDifArr входной массив
    // n * n кол-во элементов в массиве
    cub::DeviceReduce::Max(tempStorage, tempStorageBytes, CudaDifArr, maxError, n * n);

    // выделяем память для буфера
    cudaMalloc((void **)&tempStorage, tempStorageBytes);

    // размерность блоков и грида
    // 1024 потока
    dim3 blockDim = dim3(32, 32);
    dim3 gridDim = dim3((n + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);

    // создаем поток
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // bool graphCreated = false;
    // Граф CUDA позволяет оптимизировать выполнение операций и управлять потоками и зависимостями между ними.
    cudaGraph_t graph;
    // исполняемый экземпляр графа
    cudaGraphExec_t instance;

    // Начинаем захват операций, выполняемых в потоке stream и они добавляются в граф, Это позволяет создать граф, который представляет последовательность операций в потоке.
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    for (int i = 0; i < 200; i++)
    {
        calculate<<<gridDim, blockDim, 0, stream>>>(CudaNewArr, CudaArr, n);
        calculate<<<gridDim, blockDim, 0, stream>>>(CudaArr, CudaNewArr, n);
    }
    
    // тут мы завершаем захват и возвращаем граф CUDA, представляющий операции в потоке.
    cudaStreamEndCapture(stream, &graph);
  //используется для создания исполняемого экземпляра графа CUDA на основе предварительно созданного графа.
  //(указатель на переменную в которую будет сохранен дескриптор созданного экземпляра графа, сам граф , массив зависимостей, количество зависимостей)
    cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
    // ---------------------

    while (cntIteration < MAX_ITERATION / 200 && error > ACCURACY)
    {
        cntIteration++;

        // запускаем граф
        cudaGraphLaunch(instance, stream);

        // вычитаем один массив из другого
        getDifference<<<gridDim, blockDim, 0, stream>>>(CudaArr, CudaNewArr, CudaDifArr, n);

        // находим новое значение ошибки
        cub::DeviceReduce::Max(tempStorage, tempStorageBytes, CudaDifArr, maxError, n * n, stream);
        cudaMemcpy(&error, maxError, sizeof(double), cudaMemcpyDeviceToHost);
        error = std::abs(error);
    }

    // вывод резуьтата
    std::cout << "iteration: " << cntIteration << " \n"
              << "error: " << error << "\n";

    // чистка памяти
    cudaFree(CudaArr);
    cudaFree(CudaNewArr);
    cudaFree(CudaDifArr);
    delete[] arr;
    delete[] arrNew;
    return 0;
}
