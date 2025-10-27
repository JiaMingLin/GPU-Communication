#include <cuda_runtime.h>
#include <cstdio>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define CHANNELS 3
#define KERNEL_SIZE 11
#define WINDOW_SIZE (KERNEL_SIZE/2)

__global__ void blurImageKernel(unsigned char *input, unsigned char *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int index = y * width + x;

    int red = 0, green = 0, blue = 0;
    int count = 0;
    for(int ky = -WINDOW_SIZE; ky <= WINDOW_SIZE; ky++){
        for(int kx = -WINDOW_SIZE; kx <= WINDOW_SIZE; kx++){
            int nx = x + kx, ny = y + ky;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height){
                int pixel_ind = (ny * width + nx) * CHANNELS;
                unsigned char R = input[pixel_ind + 0];
                unsigned char G = input[pixel_ind + 1];
                unsigned char B = input[pixel_ind + 2];
                red += R;
                green += G;
                blue += B;
                count++;
            }
        }
    }

    int out_ind = index * CHANNELS;
    output[out_ind + 0] = red / count;
    output[out_ind + 1] = green / count;
    output[out_ind + 2] = blue / count;

}

void blurImage(unsigned char *input, unsigned char *output, int width, int height) {
    unsigned char *input_d = nullptr, *output_d = nullptr;
    cudaMalloc((void**)&input_d, width * height * CHANNELS * sizeof(unsigned char));
    cudaMalloc((void**)&output_d, width * height * CHANNELS * sizeof(unsigned char));
    cudaMemcpy(input_d, input, width * height * CHANNELS * sizeof(unsigned char), cudaMemcpyHostToDevice);
    dim3 blockSize(16, 16);
    int gridX = (width + blockSize.x - 1) / blockSize.x;
    int gridY = (height + blockSize.y - 1) / blockSize.y;
    dim3 gridSize(gridX, gridY);
    blurImageKernel<<<gridSize, blockSize>>>(input_d, output_d, width, height);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
        return;
    }
    cudaMemcpy(output, output_d, width * height * CHANNELS * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaFree(input_d);
    cudaFree(output_d);
}

int main(){
    int width, height, channels;
    unsigned char* input = stbi_load("input/input.png", &width, &height, &channels, 3);
    if (!input) {
        printf("Failed to load image: %s\n", stbi_failure_reason());
        return 1;
    }
    unsigned char* output = new unsigned char[width * height * CHANNELS];
    blurImage(input, output, width, height);
    stbi_write_png("output/blur_output.png", width, height, CHANNELS, output, width * CHANNELS);
    stbi_image_free(input);
    delete[] output;   
    return 0;
}