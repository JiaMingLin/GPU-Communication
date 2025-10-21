
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image_write.h"
#include "stb_image.h"

__global__
void grayscaleKernel(const unsigned char* input, unsigned char* output, int width, int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height){
        int index = y * width + x;
        char R = input[index * 3 + 0];
        char G = input[index * 3 + 1];
        char B = input[index * 3 + 2];
        char gray = 0.299 * R + 0.587 * G + 0.114 * B;
        output[index] = gray;
    }
}

void grayscale(const unsigned char* input, unsigned char* output, int width, int height){
    // 1. allocate memory on device(input and output)
    unsigned char* input_d = nullptr;
    unsigned char* output_d = nullptr;
    cudaMalloc((void**)&input_d, 3 * width * height * sizeof(unsigned char));
    cudaMalloc((void**)&output_d, width * height * sizeof(unsigned char));
    // copy input to device
    cudaMemcpy(input_d, input, 3 * width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // 2. launch the kernel
    dim3 blockSize(16, 16);
    int gridX = (width + blockSize.x - 1) / blockSize.x;   // ceil(width/16)
    int gridY = (height + blockSize.y - 1) / blockSize.y;   // ceil(height/16)
    dim3 gridSize(gridX, gridY);
    
    printf("Image: %dx%d, Grid: %dx%d, Block: %dx%d\n", width, height, gridX, gridY, blockSize.x, blockSize.y);
    
    grayscaleKernel<<<gridSize, blockSize>>>(input_d, output_d, width, height);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
        return;
    }
    // 3. copy the result back to host
    cudaMemcpy(output, output_d, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    
    // 4. free the memory
    cudaFree(input_d);
    cudaFree(output_d);
}

int main(){
    // read an png image from a file
    int width, height, channels;
    unsigned char* input = stbi_load("input/input.png", &width, &height, &channels, 3);
    
    if (!input) {
        printf("Failed to load image: %s\n", stbi_failure_reason());
        return 1;
    }
    
    printf("Loaded image: %dx%d, channels: %d\n", width, height, channels);
    unsigned char* output = new unsigned char[width * height];
    grayscale(input, output, width, height);

    stbi_write_png("output/grayscale_output.png", width, height, 1, output, width);
    stbi_image_free(input);
    delete[] output;

    return 0;
}