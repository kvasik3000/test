#include <iostream>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <ctype.h>
#include <string>
#include <cstring>
#include <algorithm>

int main(int argc, char* argv[]){
    auto start = std::chrono::high_resolution_clock::now();
    double error = std::stod(argv[2]);
    double err = 1.0;
    int iter = 0;
    int iter_max = std::stoi(argv[6]);
    int tol = std::stoi(argv[4]);

    double **matrix = new double*[tol];
    double **new_matrix = new double*[tol];

    for (int i = 0;i < tol;  ++i)
    {
        matrix[i] = new double[tol];
        new_matrix[i] = new double[tol];
        for(int j = 0; j < tol; ++j){
            matrix[i][j] = 20;
            new_matrix[i][j] = 20;
        }
    }
    #pragma acc enter data create(matrix[0:tol][0:tol],new_matrix[0:tol][0:tol],iter) copyin(tol,err)

    matrix[0][0] = 10;
    matrix[0][tol-1] = 20;
    matrix[tol-1][0] = 20;
    matrix[tol-1][tol-1] = 30;

    new_matrix[0][0] = 10;
    new_matrix[0][tol-1] = 20;
    new_matrix[tol-1][0] = 20;
    new_matrix[tol-1][tol-1] = 30;

    #pragma acc update device(matrix[:tol][:tol],new_matrix[:tol][:tol])
    

    #pragma acc parallel loop present(matrix[:tol][:tol],new_matrix[:tol][:tol  ]) 

    for (int i = 1; i < tol-1; ++i)
    {
        double step = (10.0 /(tol-1))*i;
        matrix[0][i]=matrix[0][0] + step;
        matrix[i][0]=matrix[0][0] + step;
        matrix[tol-1][i]=matrix[tol-1][0] + step;
        matrix[i][tol-1]=matrix[0][tol-1] + step;
        new_matrix[0][i]=new_matrix[0][0] + step;
        new_matrix[i][0]=new_matrix[0][0] + step;
        new_matrix[tol-1][i]=new_matrix[tol-1][0] + step;
        new_matrix[i][tol-1]=new_matrix[0][tol-1] + step;
        
    }
    
    // int it = 0;
    while(err > error && iter<iter_max){
        err = 0.0;
        #pragma acc update device(err)

        #pragma acc parallel loop collapse(2) present(matrix[0:tol][0:tol],new_matrix[0:tol][0:tol],tol,err) independent reduction(max:err)
        for (int i = 1; i < tol - 1; ++i)
        {
            for (int j = 1; j < tol-1; ++j)
            {
                new_matrix[i][j] = 0.25 *( matrix[i][j+1] + matrix[i][j-1]+ matrix[i-1][j] + matrix[i+1][j]);
                err = fmax(err,fabs(new_matrix[i][j] - matrix[i][j]));
            }
            
        }
        // std::copy_n(new_matrix[it], tol, matrix[it]);
        // swap(matrix,new_matrix);
        double **step = matrix;
        matrix = new_matrix;
        new_matrix = step;
       
        // it++;
        
        
        iter++;
        #pragma acc update self(err)
    }
    
    std::cout<<"iter: "<<iter<<"|"<<"err: "<< err<<std::endl;

    #pragma acc update host(matrix[:tol][:tol])
    // std::cout.precision(4);
    // for (int i = 0; i < tol; i++)
    // {
    //     for (int j = 0; j < tol; j++)
    //     {
    //         std::cout<<matrix[i][j]<<'\t';
            
    //     }
    //     std::cout << std::endl;
        
    // }
    
    #pragma acc exit data delete(matrix[:tol][:tol], new_matrix[:tol][:tol], error)

    for (int i = 0; i < tol; ++i){
        delete[] matrix[i];
        delete[] new_matrix[i];
    }
    delete[] matrix;
    delete new_matrix;

    
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
	long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << "Время работы: "<< microseconds << std::endl;
    return 0;
}
