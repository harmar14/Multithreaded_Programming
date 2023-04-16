#include <iostream>
#include <string>
#include <fstream>
#include <omp.h>
#include <stdio.h>
#include <chrono>

using namespace std;

int threadsAmount;

int* min_max(unsigned char* channel, int size, float coefficient)
{
    //coefficient determines the number of pixels to ignore
    int ignoreNum = size * coefficient;
    //histogram shows how many pixels have every level or brightness from 0 to 255
    int histogram[256] = {0};
    for (int i = 0; i < size; i++) {
        histogram[channel[i]]++;
    }
    //the brightest pixel and the darkest one
    int min = 255, max = 0;
    for (int i = 0; i < 256; i++) {
        if (histogram[i] != 0) {
            if (i < min) {
                min = i;
            }
            if (i > max) {
                max = i;
            }
        }
    }
    //applying pixel ignoring
    while (ignoreNum != 0) {
        if (histogram[min] < histogram[max]) {
            //ignoring pixels from the left (the brightest ones)
            ignoreNum -= histogram[min];
            if (ignoreNum < 0) {
                ignoreNum = 0;
            }
            else {
                min++;
            }
        }
        else {
            //ignoring pixels from the right (the darkest ones)
            ignoreNum -= histogram[max];
            if (ignoreNum < 0) {
                ignoreNum = 0;
            }
            else {
                max--;
            }
        }
    }
    //min and max should be returned
    return new(nothrow) int[2] {min, max};
}
int* min_max_omp(unsigned char* channel, int size, float coefficient)
{
    //coefficient determines the number of pixels to ignore
    int ignoreNum = size * coefficient;
    //histogram shows how many pixels have every level or brightness from 0 to 255
    int histogram[256] = { 0 };
    #pragma omp parallel
    {
        int localHist[256] = { 0 };
        #pragma omp for
        for (int i = 0; i < size; i++) {
            localHist[channel[i]]++;
        }
        #pragma omp critical
        for (int i = 0; i < 256; i++) {
            histogram[i] += localHist[i];
        }
    }
    //the brightest pixel and the darkest one
    int min = 255, max = 0;
    for (int i = 0; i < 256; i++) {
        if (histogram[i] != 0) {
            if (i < min) {
                min = i;
            }
            if (i > max) {
                max = i;
            }
        }
    }
    //applying pixel ignoring
    while (ignoreNum != 0) {
        if (histogram[min] < histogram[max]) {
            //ignoring pixels from the left (the brightest ones)
            ignoreNum -= histogram[min];
            if (ignoreNum < 0) {
                ignoreNum = 0;
            }
            else {
                min++;
            }
        }
        else {
            //ignoring pixels from the right (the darkest ones)
            ignoreNum -= histogram[max];
            if (ignoreNum < 0) {
                ignoreNum = 0;
            }
            else {
                max--;
            }
        }
    }
    //min and max should be returned
    return new(nothrow) int[2] {min, max};
}

unsigned char* contrast_improvement(unsigned char* imageData, int size, int min, int max) {
    for (int i = 0; i < size; i++)
    {
        int color = ((int)imageData[i] - min) * 255 / (max - min);
        if (color > 255) {
            color = 255;
        }
        if (color < 0) {
            color = 0;
        }
        imageData[i] = color;
    }
    return imageData;
}
unsigned char* contrast_improvement_omp(unsigned char* imageData, int size, int min, int max) {
    int color;
    #pragma omp parallel for num_threads(threadsAmount) schedule(static) private(color)
    for (int i = 0; i < size; i++)
    {
        color = ((int)imageData[i] - min) * 255 / (max - min);
        if (color > 255) {
            color = 255;
        }
        if (color < 0) {
            color = 0;
        }
        imageData[i] = color;
    }
    return imageData;
}

int main(int argc, char* argv[])
{
    //input example: MTPLab2.exe in.pnm out.pnm <threads_num> <coefficient - float [0.0, 0.5)>
    if (argc != 5) {
        cerr << "Wrong number of parameters";
        exit(1);
    }

    string nameIn = argv[1];
    string nameOut = argv[2];
    threadsAmount = stoi(argv[3]);
    float coefficient = stof(argv[4]);

    if (threadsAmount == 0 || threadsAmount > omp_get_max_threads()) {
        threadsAmount = omp_get_max_threads();
    }
    else if (threadsAmount < -1) {
        cerr << "Wrong number of threads";
        exit(1);
    }

    //preparing result data
    int size = 0;
    unsigned char* image = new(nothrow) unsigned char[size];
    //checking if memory was allocated successfully
    if (image == nullptr) {
        cerr << "Memory can not be allocated";
        delete[] image;
        exit(1);
    }

    //reading binary file
    ifstream fileIn(nameIn, ios::binary);

    if (!fileIn.is_open()) {
        cerr << "Reading file error";
        exit(1);
    }

    //getting file format (P5 or P6), width, height and max value
    string format;
    int width, height, max_val;
    fileIn >> format >> width >> height >> max_val;
    //skipping "\r" (code is 13)
    fileIn.get();

    if (format != "P5" && format != "P6") {
        cerr << "Invalid header";
        exit(1);
    }
    
    //if there is only one pixel, do nothing
    if (width == height == 1) {
        fileIn.close();
        ifstream input(nameIn, ios::binary);
        ofstream output(nameOut, ios::binary);
        output << input.rdbuf();
        input.close();
        output.close();
        cout << "time(" << abs(threadsAmount) << " thread(s)) : 0 ms\n";

        return 0;
    }

    size = height * width;

    if (format == "P5") {
        //P5 is grayscale
        unsigned char* imageData = new(nothrow) unsigned char[size];
        //checking if memory was allocated successfully
        if (imageData == nullptr) {
            cerr << "Memory can not be allocated";
            delete[] imageData;
            exit(1);
        }

        if (threadsAmount == -1) {
            //without OMP
            for (int i = 0; i < size; i++)
            {
                imageData[i] = fileIn.get();
            }
            fileIn.close();
            auto start_time = chrono::steady_clock::now();
            int* greyscale_min_max = min_max(imageData, size, coefficient);
            image = contrast_improvement(imageData, size, greyscale_min_max[0], greyscale_min_max[1]);
            auto end_time = chrono::steady_clock::now();
            auto elapsed_ms = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
            cout << "time(1 thread(s)) : " << elapsed_ms.count() << " ms\n";
        }
        else {
            //with OMP
            for (int i = 0; i < size; i++)
            {
                imageData[i] = fileIn.get();
            }
            fileIn.close();
            auto start_time = chrono::steady_clock::now();
            int* greyscale_min_max = min_max_omp(imageData, size, coefficient);
            image = contrast_improvement_omp(imageData, size, greyscale_min_max[0], greyscale_min_max[1]);
            auto end_time = chrono::steady_clock::now();
            auto elapsed_ms = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
            cout << "time(" << threadsAmount << " thread(s)) : " << elapsed_ms.count() << " ms\n";
        }
    }
    else if (format == "P6") {
        //P6 is a colored image, there are R-channel, G-channel and B-channel
        unsigned char* channelR = new(nothrow) unsigned char[size];
        if (channelR == nullptr) {
            cerr << "Memory can not be allocated";
            delete[] channelR;
            exit(1);
        }
        unsigned char* channelG = new(nothrow) unsigned char[size];
        if (channelG == nullptr) {
            cerr << "Memory can not be allocated";
            delete[] channelG;
            exit(1);
        }
        unsigned char* channelB = new(nothrow) unsigned char[size];
        if (channelB == nullptr) {
            cerr << "Memory can not be allocated";
            delete[] channelB;
            exit(1);
        }
        //full image data is also needed to send it to contrast_improvement function later
        unsigned char* imageData = new(nothrow) unsigned char[size * 3];
        if (imageData == nullptr) {
            cerr << "Memory can not be allocated";
            delete[] imageData;
            exit(1);
        }
        //every pixel is given by 3 values, so the number of all values is size * 3
        for (int i = 0; i < size; i++) {
            channelR[i] = fileIn.get();
            channelG[i] = fileIn.get();
            channelB[i] = fileIn.get();
            imageData[i * 3] = channelR[i];
            imageData[i * 3 + 1] = channelG[i];
            imageData[i * 3 + 2] = channelB[i];
        }
        fileIn.close();
        if (threadsAmount == -1) {
            //without OMP
            auto start_time = chrono::steady_clock::now();

            int min, max;
            //getting min and max brightness of each channel ignoring pixels according to coefficient value
            int* minmaxR = min_max(channelR, size, coefficient);
            if (minmaxR == nullptr) {
                cerr << "Memory can not be allocated";
                delete[] minmaxR;
                exit(1);
            }
            min = minmaxR[0];
            max = minmaxR[1];
            delete[] channelR, minmaxR;
            int* minmaxG = min_max(channelG, size, coefficient);
            if (minmaxG == nullptr) {
                cerr << "Memory can not be allocated";
                delete[] minmaxG;
                exit(1);
            }
            if (minmaxG[0] < min) {
                min = minmaxG[0];
            }
            if (minmaxG[1] > max) {
                max = minmaxG[1];
            }
            delete[] channelG, minmaxG;
            int* minmaxB = min_max(channelB, size, coefficient);
            if (minmaxB == nullptr) {
                cerr << "Memory can not be allocated";
                delete[] minmaxB;
                exit(1);
            }
            if (minmaxB[0] < min) {
                min = minmaxB[0];
            }
            if (minmaxB[1] > max) {
                max = minmaxB[1];
            }
            delete[] channelB, minmaxB;
            size *= 3;
            image = contrast_improvement(imageData, size, min, max);

            auto end_time = chrono::steady_clock::now();
            auto elapsed_ms = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
            cout << "time(1 thread(s)) : " << elapsed_ms.count() << " ms\n";
        }
        else {
            //with OMP
            auto start_time = chrono::steady_clock::now();

            int min, max;
            //getting min and max brightness of each channel ignoring pixels according to coefficient value
            int* minmaxR = min_max_omp(channelR, size, coefficient);
            if (minmaxR == nullptr) {
                cerr << "Memory can not be allocated";
                delete[] minmaxR;
                exit(1);
            }
            min = minmaxR[0];
            max = minmaxR[1];
            delete[] channelR, minmaxR;
            int* minmaxG = min_max_omp(channelG, size, coefficient);
            if (minmaxG == nullptr) {
                cerr << "Memory can not be allocated";
                delete[] minmaxG;
                exit(1);
            }
            if (minmaxG[0] < min) {
                min = minmaxG[0];
            }
            if (minmaxG[1] > max) {
                max = minmaxG[1];
            }
            delete[] channelG, minmaxG;
            int* minmaxB = min_max_omp(channelB, size, coefficient);
            if (minmaxB == nullptr) {
                cerr << "Memory can not be allocated";
                delete[] minmaxB;
                exit(1);
            }
            if (minmaxB[0] < min) {
                min = minmaxB[0];
            }
            if (minmaxB[1] > max) {
                max = minmaxB[1];
            }
            delete[] channelB, minmaxB;
            size *= 3;
            image = contrast_improvement_omp(imageData, size, min, max);

            auto end_time = chrono::steady_clock::now();
            auto elapsed_ms = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
            cout << "time(" << threadsAmount << " thread(s)) : " << elapsed_ms.count() << " ms\n";
        }
    }
    
    //writing the result to a file
    ofstream fileOut(nameOut, ios::binary);
    if (!fileOut.is_open()) {
        cerr << "Writing file error";
        exit(1);
    }
    fileOut << format << "\n" << width << " " << height << "\n" << max_val << "\n";
    for (size_t i = 0; i < size; i++) {
        fileOut << image[i];
    }
    delete[] image;
    fileOut.close();

    return 0;
}