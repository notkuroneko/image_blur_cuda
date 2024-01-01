#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>

using namespace std;
using namespace cv;

__global__ void average_blur(float *A, float *B, int r, int c, int idx){
    int rowID = threadIdx.y + blockIdx.y * blockDim.y; 	// Row address
	int colID = threadIdx.x + blockIdx.x * blockDim.x;	
    if (rowID<r&&colID<c)
    {
        float sum=0, average;
        for (int i=rowID-idx/2; i<=rowID+idx/2; i++)
        for (int j=colID-idx/2; j<=colID+idx/2; j++)
        if (rowID>=idx/2 && rowID<=r-idx/2 && colID>=idx/2 && colID<=c-idx/2) sum += A[i*r+j];
        average = (float)sum/(idx*idx);
        B[rowID*r+colID] = average;
    }
}

//uncomment for future attempts to create library for the program

//Mat average_blur(const char* image_path) {
int main(void) {
    //cudaError_t err = cudaSuccess;
    const char* image_path = "Untitled.jpeg";
	Mat image = imread(image_path, IMREAD_GRAYSCALE);
    int r = image.rows;
    int c = image.cols;
	//Mat result
	Mat blur(r, c, CV_8UC1);
    int innerMatrixIndex = 3;

    const size_t d_size = sizeof(float) * size_t(r*c);
    const dim3 threadsPerBlock(32,32); 	// Must not exceed 1024 (max thread per block)
	const dim3 blocksPerGrid((r/32),(c/32));	
    
    float *h_matImg = (float*) malloc(2*r*c * sizeof(float));
    float *h_blurImg = (float*) malloc(2*r*c * sizeof(float));

    for (int i=0; i<r; i++)
    for (int j=0; j<c; j++)
    h_matImg[i*r+j]=(float)image.at<uchar>(i,j);

    float *d_matImg, *d_blurImg;
    cudaMalloc((void **) &d_matImg, d_size);
    cudaMalloc((void **) &d_blurImg, d_size);
    cudaMemcpy(d_matImg, h_matImg, d_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_blurImg, h_blurImg, d_size, cudaMemcpyHostToDevice);
    printf("Data copied to CUDA\n");

    average_blur<<<blocksPerGrid, threadsPerBlock>>>(d_matImg, d_blurImg, r, c, innerMatrixIndex);
    cudaMemcpy(h_blurImg, d_blurImg, d_size, cudaMemcpyDeviceToHost);
    for (int i=0; i<r; i++) {
        for (int j=0; j<c; j++) {
            blur.at<uchar>(i,j)=h_blurImg[i*r+j];
            //cout << h_blurImg[i*r+j] << " ";
        }
        //cout << endl;
    }

    cudaFree(d_matImg);
    cudaFree(d_blurImg);
    free(h_blurImg);
    free(h_matImg);
    printf("Data released\n");
    
    namedWindow("image", WINDOW_AUTOSIZE);
	imshow("image", image);
    namedWindow("blur", WINDOW_AUTOSIZE);
	imshow("blur", blur);
    waitKey(0);
    return 0;
    //return blur;
}

// $ /usr/bin/nvcc -o average_blur_cuda.out average_blur_cuda.cu `pkg-config --cflags --libs opencv4`
// $ ./average_blur_cuda.out