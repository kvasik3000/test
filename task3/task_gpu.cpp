#include <iostream>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <ctype.h>
#include <string>
#include <cstring>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// определим макрос с помощью которого проведем индексацию, так как библиотека cublas совместима с Fortran использует хранение по столбцам и индексацию на основе 1
// и по этому не может использовать родную индексицию массивов
// i - line, j - column, ld - matrix size 
#define IDX2C(i,j,ld) (((j)*(ld))+(i)) 

void init_array(double *matrix, int tol)
{
    #pragma acc parallel present(matrix)
    {
        //происходит заполнение углов матрицы
        matrix[IDX2C(0, 0, tol)] = 10.0;
        matrix[IDX2C(0, tol-1, tol)] = 20.0;
        matrix[IDX2C(tol-1,0, tol)] = 20.0;
        matrix[IDX2C(tol-1, tol-1, tol)]= 30.0;

        //заполняем матрицу по периметру
        double step = (10.0 /(tol-1));

        #pragma acc loop independent
        for (int i = 1; i < tol-1; ++i)
        {
            double step_i = step * i;
            matrix[IDX2C(0, i, tol)]=matrix[IDX2C(0, 0, tol)] + step_i;
            matrix[IDX2C(i, 0, tol)]=matrix[IDX2C(0, 0, tol)] + step_i;
            matrix[IDX2C(tol-1,i, tol)]=matrix[IDX2C(tol-1,0, tol)] + step_i;
            matrix[IDX2C(i, tol-1, tol)]=matrix[IDX2C(0, tol-1, tol)] + step_i;
        }
        //Массив инициализирую значениями 20 , так как 20 это среднее значение по углам сетки, это значит,
        //что до основного кода заполнения матрицы, ответы будут уже приближены к правильным,
        //это ускоряет процесс вычисления на сетках большого размера.
        // #pragma acc loop independent collapse(2)
        // for (int i = 1;i < tol-1;  ++i)
        // {
        //     for(int j = 1; j < tol-1; ++j)
        //     {
        //         matrix[i][j] = 20;
        //     }
        // }
    }
}





int main(int argc, char* argv[]){
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasStatus_t status;

    auto start = std::chrono::high_resolution_clock::now();
    double err = std::stod(argv[2]);
    double error = 1.0;
    int iter = 0;
    int iter_max = std::stoi(argv[6]);
    int tol = std::stoi(argv[4]);
    int size = tol*tol;
    // скаляр для вычитания
    double alpha = -1;
    int step_inc = 1;//шаг инкремента
    int max_ind = 0;

    //создаем два вектора и доп вектор

    double *matrix = new double [size];
    double *new_matrix = new double [size];
    double *step_matrix = new double [size];

    // for (int i = 0;i < tol;  ++i)
    // {   
    //     matrix[i] = new double[tol];
    //     new_matrix[i] = new double[tol];        
    // }
    //использую эту директиву мы определяем область кода, в которой матрица остается на GPU и разделяется между всеми ядрами в этой области 
    // enter data определяет начало времени жизни
    // create выделяет память на GPU но не копирует
    #pragma acc enter data create(matrix[:size],new_matrix[:size],step_matrix[:size])
    //инициализируем
    init_array(matrix,tol);
    init_array(new_matrix,tol);

    // создаем флаг, который будет обновлять ошибку на хосте
    bool flag = true;
    //выделяем память на GPU для error и копируем данные с хоста на GPU
    #pragma acc data copy(error)
    {

        while(iter < iter_max)
        {
            // if(iter%tol==0){
                
            // }
            flag = !(iter%tol);

            if(flag){
                // зануляем ошибкy на GPU
                #pragma acc kernels
                error = 0;
            }
            //тут мы параллелим , директива (loop) является директивой параллельного исполнения(Описываем какой тип параллелизма будем использовать)
            //с помощью условия collapse мы превращаем наши два вложенных цикла в один, тем самым выигрываем время
            // выполняем редукцию для error
            // c помощью present мы говорим, что наши матрицы лежат на GPU
            // async - условие , где мы запускаем ядро асинхронно в 1 очереди, а когда закончим секцию мы принудительно не синхронизируем

            #pragma acc kernels loop independent collapse(2) reduction(max : error) present(matrix, new_matrix) async(1)
            for (int i = 1; i < tol - 1; ++i)
            {
                for (int j = 1; j < tol-1; ++j)
                {
                    new_matrix[IDX2C(i, j, tol)] = 0.25 *( matrix[IDX2C(i, j+1, tol)] + matrix[IDX2C(i, j-1, tol)]+ matrix[IDX2C(i-1,j, tol)] + matrix[IDX2C(i+1,j, tol)]);
                    // тут мы пересчитываем ошибку
                    // if(flag)
                    //     error = fmax(error,std::abs(new_matrix[i][j] - matrix[i][j]));
                }
                
            }
            // тут мы типа меняем матрицы местами 
            double *step = matrix;
            matrix = new_matrix;
            new_matrix = step;

            //проверяем ошибку 

            if(flag)
            {
                #pragma acc data present(matrix, new_matrix, step_matrix) wait(1)
                {
                    #pragma acc host_data use_device(matrix, new_matrix, step_matrix)
                    {
                        //Эта функция копирует вектор new_matrix в вектор step_matrix.
                        status = cublasDcopy(handle, size,new_matrix, step_inc, step_matrix, step_inc);
                        if(status != CUBLAS_STATUS_SUCCESS) {
                            std::cerr << "copy error" << std::endl; 
                            exit(1);
                        }
                        //Эта функция умножает вектор  matrix на скаляр alpha
                        //и добавляет его к вектору, step_matrix перезаписывая последний вектор с результатом.
                        //умножаем вектор на скаляр и + вектор
                        status = cublasDaxpy(handle, size, &alpha, matrix, step_inc, step_matrix, step_inc);
                        if(status != CUBLAS_STATUS_SUCCESS) {
                            std::cerr << "sum error" << std::endl;
                            exit(1);
                        }
                        
                        //Эта функция находит (наименьший) индекс элемента максимальной величины. 
                        status = cublasIdamax(handle, size, step_matrix, step_inc, &max_ind);
                        if(status != CUBLAS_STATUS_SUCCESS) {
                            std::cerr << "abs max error" << std::endl; 
                            exit(1);
                        }
                    
                        // std::cout<<error<<std::endl;

                        #pragma acc kernels present(error)
                        error = fabs(step_matrix[max_ind - 1]);	
                        
                    }   
                }

                // тут мы обновляем нашу ошибочку на CPU значениями с GPU, ну и синхронизируем с помощью wait
                #pragma acc update host(error) wait(1)
                // тут если ошибка не превышает поданную ошибку err(которая наша точность), то мы заканчиваем цикл
                if(error < err)
                    break;
            }
            
            iter++;
        }
        // синхронизируем
        #pragma acc wait(1)
    }
    
    std::cout<<"iter: "<<iter<<"|"<<"err: "<< err<<std::endl;

    // тут производится наш вывод
    // for (int i = 0; i < tol; ++i)
    // {
    //     for (int j = 0; j < tol; ++j)
    //     {
    //         //используем  значения с GPU
    //         #pragma acc kernels present(matrix)
    //         printf("%.2f\t", matrix[IDX2C(i, j, tol)]);
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;

    // тут мы типа выходим из неструктурированной  секции дата и очищаем память
    #pragma acc exit data delete(matrix[:size], new_matrix[:size], step_matrix[:size])
    // for (int i = 0; i < tol; ++i){
    //     delete[] matrix[i];
    //     delete[] new_matrix[i];
    // }
    cublasDestroy(handle);
    delete[] step_matrix;
    delete[] matrix;
    delete[] new_matrix;

    
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
	long long seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
	std::cout << "Время работы: "<< seconds << " c"<< std::endl;
    return 0;
}
