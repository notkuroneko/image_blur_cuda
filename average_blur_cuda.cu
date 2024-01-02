#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <stdio.h>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;
using namespace cv;

#define THREAD_DIM = 32;

__global__ void average_blur(float *A, float *B, int r, int c, int idx){ 
    int rowID = threadIdx.y + blockIdx.y * blockDim.y; 	// Row address
	int colID = threadIdx.x + blockIdx.x * blockDim.x;	
    if (rowID<r&&colID<c) //check if index is within bounds
    {
        float sum=0; 
        for (int i=rowID-idx/2; i<=rowID+idx/2; i++) //average blur algorithm
        for (int j=colID-idx/2; j<=colID+idx/2; j++)
        if (rowID>=idx/2 && rowID<=r-idx/2 && colID>=idx/2 && colID<=c-idx/2) sum += A[i*r+j];
        B[rowID*r+colID] = (float)sum/(idx*idx);
    }
}

//uncomment for future attempts to create library for the program

//Mat average_blur(const char* image_path) {
int main(void) {
    //cudaError_t err = cudaSuccess;
    const char* image_path = "sui.jpeg"; //reads image from library into grayscale
	Mat image = imread(image_path, IMREAD_GRAYSCALE);
    int r = image.rows;
    int c = image.cols;
	//Mat result
	Mat blur(r, c, CV_8UC1);
    int innerMatrixIndex = 3; //the size of the inner matrix used for blurring

    const size_t d_size = sizeof(float) * size_t(r*c); //defining CUDA parameters
    const dim3 threadsPerBlock(32,32); 	// Must not exceed 1024 (max thread per block)
	const dim3 blocksPerGrid((c+threadsPerBlock.x-1)/threadsPerBlock.x,((r+threadsPerBlock.y-1)/threadsPerBlock.y));	
    cout <<"CUDA kernel launch with " << blocksPerGrid.x << " x " << blocksPerGrid.y << " blocks of " << threadsPerBlock.x << " x " << threadsPerBlock.y <<" threads" << endl;
    
    float *h_matImg = (float*) malloc(3*r*c/2 * sizeof(float)); //allocate memory on host
    float *h_blurImg = (float*) malloc(3*r*c/2 * sizeof(float));

    for (int i=0; i<r; i++)
    for (int j=0; j<c; j++)
    h_matImg[i*r+j]=(float)image.at<uchar>(i,j); //transfer grayscale data to host

    float *d_matImg, *d_blurImg;
    cudaMalloc((void **) &d_matImg, d_size); //allocate memory on device
    cudaMalloc((void **) &d_blurImg, d_size);
    cudaMemcpy(d_matImg, h_matImg, d_size, cudaMemcpyHostToDevice); //copy data from host to device
    printf("Data copied to CUDA\n");

    auto start = std::chrono::steady_clock::now(); 
    average_blur<<<blocksPerGrid, threadsPerBlock>>>(d_matImg, d_blurImg, r, c, innerMatrixIndex); //calling kernel function
    auto end = std::chrono::steady_clock::now();
    cout << "Time elapsed on CUDA:" << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "microseconds" << endl;
    cudaMemcpy(h_blurImg, d_blurImg, d_size, cudaMemcpyDeviceToHost); //copy data from device to host
    cudaThreadSynchronize();
    for (int i=0; i<r; i++) {
        for (int j=0; j<c; j++) {
            blur.at<uchar>(i,j)=h_blurImg[i*r+j]; //read data into grayscale image
            //cout << h_blurImg[i*r+j] << " ";
        }
        //cout << endl;
    }

    cudaFree(d_matImg); //free memory
    cudaFree(d_blurImg);
    free(h_blurImg);
    free(h_matImg);
    printf("Data released\n");
    
    namedWindow("image", WINDOW_AUTOSIZE); //show image
	imshow("image", image);
    namedWindow("blur", WINDOW_AUTOSIZE);
	imshow("blur", blur);
    waitKey(0);
    return 0;
    //return blur;
}

// $ /usr/bin/nvcc -o average_blur_cuda.out average_blur_cuda.cu `pkg-config --cflags --libs opencv4`
// $ ./average_blur_cuda.out