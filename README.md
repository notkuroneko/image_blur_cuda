OpenCV image blur algorithm using CUDA

Algorithm:
- Read image file into a grayscale 2D matrix.
- Allocate memory for the host & device matrices. 1 for the original image, and 1 for the blurred image.
- Copy from the host matrices to the device matrices.
- Launch the CUDA kernel to process the average blur algorithm.
- Copy the processed matrix from device back to host.
- Create a window to show the image files.

Execution:
- Open a terminal, create directory to the folder
- Run "$ <path_to_nvcc>/nvcc -o <file_name>.out <file_name>.cu". This will invoke the nvcc compiler and create an object of the CUDA file.
- Run "$ ./<file_name>.out" to execute the object file.

Remaining issues: 
- Only works for square image files (x by x pixels). Rectangular images turns from an axb image turns into a bxb or axa image if b<a or a<b respectively.

Future plans
- Processing a folder of image files
- Header creation

References:
- https://github.com/jcbacong/CUDA-matrix-addition/blob/master/main.cu (for matrix addition)
- https://github.com/thomasplantin/cuda-image-processing/blob/master/main.cu (for image processing algorithms using cuda)
- https://github.com/the-other-mariana/image-processing-algorithms/tree/master (for image processing algorithms using openmp)
