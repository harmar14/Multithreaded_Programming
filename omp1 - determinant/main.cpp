#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <new>

#define EPSILON (numeric_limits<float>::epsilon() * (1e-7))

using namespace std;

void matrix_out(float* matrix, size_t n) {

    cout << "\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            cout << matrix[i * n + j] << " ";
        cout << "\n";
    }

}

float calc_det(float* matrix, size_t n) {

    float tmp;
    float det = 1; 

    auto start_time = chrono::steady_clock::now();

    for (int k = 0; k < n; ++k) {

        float pivot = matrix[k * n + k];
        int pivotRow = k;
        for (int row = k + 1; row < n; ++row) {
            if (fabs(matrix[row * n + k] - pivot) > EPSILON) { //difference between fabs and abs is that fabs returns float but abs returns int
                pivot = matrix[row * n + k];
                pivotRow = row;
            }
        }
        if (fabs(pivot - 0.0) <= EPSILON) {
            return 0.0;
        }
        if (pivotRow != k) {
            for (int i = 0; i < n; i++) {
                tmp = matrix[k * n + i];
                matrix[k * n + i] = matrix[pivotRow * n + i];
                matrix[pivotRow * n + i] = tmp;
            }
            det *= -1.0;
        }
        det *= pivot;

        for (int row = k + 1; row < n; ++row) {
            for (int col = k + 1; col < n; ++col) {
                matrix[row * n + col] -= matrix[row * n + k] * matrix[k * n + col] / pivot;
            }
        }

    }

    auto end_time = chrono::steady_clock::now();

    auto elapsed_ms = chrono::duration_cast<chrono::milliseconds>(end_time - start_time); //nanoseconds are more convenient for small matrices
    cout << "time(1 thread(s)) : " <<  elapsed_ms.count() << " ms\n";

    return det;

}

float calc_det_omp(float* matrix, size_t n, int thNum) {

    float tmp;
    float det = 1;

    auto start_time = chrono::steady_clock::now();

    for (int k = 0; k < n; ++k) {

        float pivot = matrix[k * n + k];
        int pivotRow = k;

        #pragma omp parallel for num_threads(thNum) schedule(static, 2)
        for (int row = k + 1; row < n; ++row) {
            #pragma omp critical
            if (fabs(matrix[row * n + k] - pivot) > EPSILON) {
                pivot = matrix[row * n + k];
                pivotRow = row;
            }
        }
        if (fabs(pivot - 0.0) <= EPSILON) {
            return 0.0;
        }
        if (pivotRow != k) {
            for (int i = 0; i < n; i++) {
                tmp = matrix[k * n + i];
                matrix[k * n + i] = matrix[pivotRow * n + i];
                matrix[pivotRow * n + i] = tmp;
            }
            det *= -1.0;
        }
        det *= pivot;

        #pragma omp parallel for num_threads(thNum) schedule(static, 2)
        for (int row = k + 1; row < n; ++row) {
            for (int col = k + 1; col < n; ++col) {
                matrix[row * n + col] -= matrix[row * n + k] * matrix[k * n + col] / pivot;
            }
        }

    }

    auto end_time = chrono::steady_clock::now();

    auto elapsed_ms = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    cout << "time(" << thNum << " thread(s)) : " << elapsed_ms.count() << " ms\n";

    return det;
}

int main(int argc, char* argv[]) {

    if (argc != 4) {
        cerr << "Wrong number of parameters";
        exit(1);
    }

    string nameIn = argv[1];
    string nameOut = argv[2];
    int threadsAmount = stoi(argv[3]);

    //opening input file
    ifstream input;
    input.open(nameIn);
    if (!input) {
        cerr << "Reading file error";
        exit(1);
    }

    //getting size of square matrix
    size_t n;
    input >> n;

    //reading matrix from file
    float* matrix = new (nothrow) float[n * n];

    if (matrix == nullptr) {
        cerr << "Memory can not be allocated";
        exit(1);
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            input >> matrix[i * n + j];
        }
    }
    
    input.close();

    //calling function to calculate determinant (Gauss method)
    float det;
    if (threadsAmount == -1) {
        det = calc_det(matrix, n);
    }
    else if (threadsAmount == 0 or threadsAmount > omp_get_max_threads()) {
        det = calc_det_omp(matrix, n, omp_get_max_threads());
    }
    else {
        det = calc_det_omp(matrix, n, threadsAmount);
    }

    //deleting matrix from memory
    delete[] matrix;

    //opening output file
    ofstream output;
    output.open(nameOut);
    if (!output) {
        cerr << "Writing file error";
        exit(1);
    }

    //writing result with two decimal places only
    output << fixed;
    output.precision(2);
    output << det << "\n";
    output.close();

    return 0;

}
