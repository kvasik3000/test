#include <iostream>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <ctype.h>
#include <string>
#include <cstring>
#include <algorithm>

void init_array(double **matrix, int tol)
{
    #pragma acc parallel present(matrix)
    {
        //происходит заполнение углов матрицы
        matrix[0][0] = 10;
        matrix[0][tol-1] = 20;
        matrix[tol-1][0] = 20;
        matrix[tol-1][tol-1] = 30;

        //заполняем матрицу по периметру
        double step = (10.0 /(tol-1));

        #pragma acc loop independent
        for (int i = 1; i < tol-1; ++i)
        {
            double step_i = step * i;
            matrix[0][i]=matrix[0][0] + step_i;
            matrix[i][0]=matrix[0][0] + step_i;
            matrix[tol-1][i]=matrix[tol-1][0] + step_i;
            matrix[i][tol-1]=matrix[0][tol-1] + step_i;
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
    auto start = std::chrono::high_resolution_clock::now();
    double err = std::stod(argv[2]);
    double error = 0.0;
    int iter = 0;
    int iter_max = std::stoi(argv[6]);
    int tol = std::stoi(argv[4]);
    //создаем две дмумерные матрицы

    double **matrix = new double*[tol];
    double **new_matrix = new double*[tol];

    for (int i = 0;i < tol;  ++i)
    {   
        matrix[i] = new double[tol];
        new_matrix[i] = new double[tol];        
    }
    //использую эту директиву мы определяем область кода, в которой матрица остается на GPU и разделяется между всеми ядрами в этой области 
    // enter data определяет начало времени жизни
    // create выделяет память на GPU но не копирует
    #pragma acc enter data create(matrix[:tol][:tol],new_matrix[:tol][:tol])
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
            if(iter%tol==0){
                
            }
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
                    new_matrix[i][j] = 0.25 *( matrix[i][j+1] + matrix[i][j-1]+ matrix[i-1][j] + matrix[i+1][j]);
                    // тут мы пересчитываем ошибку
                    if(flag)
                        error = fmax(error,std::abs(new_matrix[i][j] - matrix[i][j]));
                }
                
            }
            // тут мы типа меняем матрицы местами 
            double **step = matrix;
            matrix = new_matrix;
            new_matrix = step;

            //проверяем ошибку 

            if(flag)
            {
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
    //         printf("%.2f\t", matrix[i][j]);
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;

    // тут мы типа выходим из неструктурированной  секции дата и очищаем память
    #pragma acc exit data delete(matrix[:tol][:tol], new_matrix[:tol][:tol])
    for (int i = 0; i < tol; ++i){
        delete[] matrix[i];
        delete[] new_matrix[i];
    }
    delete[] matrix;
    delete[] new_matrix;

    
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
	long long seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
	std::cout << "Время работы: "<< seconds << " c"<< std::endl;
    return 0;
}