#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include <omp.h>
#include <stdio.h>
#include <chrono>
#include <new>

using namespace std;

void sort(int* array, int startIdx, int endIdx) {

    if (startIdx < endIdx) {

        int right = endIdx;
        int left = startIdx;
        int pivot = array[(int)((left + right) / 2)];

        while (left <= right) {
            while (array[left] < pivot) {
                left++;
            }
            while (array[right] > pivot) {
                right--;
            }
            if (left <= right) {
                int tmp = array[left];
                array[left] = array[right];
                array[right] = tmp;
                left++;
                right--;
            }
        }

        int divideIdx = left;

        sort(array, startIdx, divideIdx - 1);
        sort(array, divideIdx, endIdx);
    }

}

void sort_with_tasks(int* array, int startIdx, int endIdx, int thNum) {

    if (startIdx < endIdx) {

        int right = endIdx;
        int left = startIdx;
        int pivot = array[(int)((left + right) / 2)];

        while (left <= right) {
            while (array[left] < pivot) {
                left++;
            }
            while (array[right] > pivot) {
                right--;
            }
            if (left <= right) {
                int tmp = array[left];
                array[left] = array[right];
                array[right] = tmp;
                left++;
                right--;
            }
        }

        int divideIdx = left;

        //a way to reduce execution time
        if ( (endIdx - startIdx) < 1000 ) {
            if (startIdx < right) {
                sort_with_tasks(array, startIdx, divideIdx - 1, thNum);
            }
            if (left < endIdx) {
                sort_with_tasks(array, divideIdx, endIdx, thNum);
            }
        }
        else {
            #pragma omp parallel num_threads(thNum)
            {
                #pragma omp task
                sort_with_tasks(array, startIdx, divideIdx - 1, thNum);
                #pragma omp task
                sort_with_tasks(array, divideIdx, endIdx, thNum);
            }
        }
                
    }

}

void sort_with_sections(int* array, int startIdx, int endIdx, int thNum) {

    if (startIdx < endIdx) {

        int right = endIdx;
        int left = startIdx;
        int pivot = array[(int)((left + right) / 2)];

        while (left <= right) {
            while (array[left] < pivot) {
                left++;
            }
            while (array[right] > pivot) {
                right--;
            }
            if (left <= right) {
                int tmp = array[left];
                array[left] = array[right];
                array[right] = tmp;
                left++;
                right--;
            }
        }

        int divideIdx = left;

        if ((endIdx - startIdx) < 1000) {
            if (startIdx < right) {
                sort_with_sections(array, startIdx, divideIdx - 1, thNum);
            }
            if (left < endIdx) {
                sort_with_sections(array, divideIdx, endIdx, thNum);
            }
        }
        else {
            #pragma omp parallel num_threads(thNum)
            {
                #pragma omp sections
                {
                    #pragma omp section
                    sort_with_sections(array, startIdx, divideIdx - 1, thNum);
                    #pragma omp section
                    sort_with_sections(array, divideIdx, endIdx, thNum);
                }
            }
        }

    }

}

int* quick_sort(int* array, size_t n) {

    auto start_time = chrono::steady_clock::now();

    sort(array, 0, n - 1);

    auto end_time = chrono::steady_clock::now();

    auto elapsed_ms = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    cout << "time(1 thread(s)) : " << elapsed_ms.count() << " ms\n";

    return array;

}

int* quick_sort_with_tasks(int* array, size_t n, int thNum) {

    auto start_time = chrono::steady_clock::now();
    
    #pragma omp parallel num_threads(thNum)
    {
        #pragma omp single
        sort_with_tasks(array, 0, n - 1, thNum);
    }

    auto end_time = chrono::steady_clock::now();

    auto elapsed_ms = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    cout << "time(" << thNum << " thread(s)) : " << elapsed_ms.count() << " ms\n";

    return array;

}

int* quick_sort_with_sections(int* array, size_t n, int thNum) {

    auto start_time = chrono::steady_clock::now();

    #pragma omp parallel num_threads(thNum)
    {
        #pragma omp single
        sort_with_sections(array, 0, n - 1, thNum);
    }

    auto end_time = chrono::steady_clock::now();

    auto elapsed_ms = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    cout << "time(" << thNum << " thread(s)) : " << elapsed_ms.count() << " ms\n";

    return array;

}

void array_out(int* array, size_t n) {
    for (int i = 0; i < n; i++) {
        cout << array[i] << " ";
    }
    cout << "\n";
}

int main(int argc, char* argv[]) {

    if (argc != 5) {
        cerr << "Wrong number of parameters";
        exit(1);
    }
    
    string nameIn = argv[1];
    string nameOut = argv[2];
    int threadsAmount = stoi(argv[3]);
    int realization = stoi(argv[4]);
    // realizations:
    // 0 - without multithreading
    // 1 - with OMP sections (threadsAmount >= 0)
    // 2 - with OMP tasks (threadsAmount >= 0)

    if (threadsAmount > omp_get_max_threads() or threadsAmount == 0)
        threadsAmount = omp_get_max_threads();

    //opening input file
    ifstream input;
    input.open(nameIn);
    if (!input) {
        cerr << "Reading file error";
        exit(1);
    }

    //getting size of array
    size_t n;
    input >> n;

    //reading array from file
    int* array = new (nothrow) int[n];

    if (array == nullptr) {
        cerr << "Memory can not be allocated";
        exit(1);
    }

    for (int i = 0; i < n; i++) {
        input >> array[i];
    }

    input.close();

    //array_out(array, n);

    switch (realization) {
    case 0:
        //no OMP
        array = quick_sort(array, n);
        //array_out(array, n);
        break;
    case 1:
        //OMP sections
        array = quick_sort_with_sections(array, n, threadsAmount);
        break;
    case 2:
        //OMP tasks
        array = quick_sort_with_tasks(array, n, threadsAmount);
        break;
    default:
        cerr << "No " << realization << " realization. Choose 0, 1 or 2";
    }

    //opening output file
    ofstream output;
    output.open(nameOut);
    if (!output) {
        cerr << "Writing file error";
        delete[] array;
        exit(1);
    }

    //writing result to file
    for (int i = 0; i < n; i++) {
        output << array[i] << " ";
    }
    output << "\n";

    //deleting array from memory
    delete[] array;
    
    output.close();

    return 0;

}