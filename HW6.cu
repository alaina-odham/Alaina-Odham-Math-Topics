// Name: Alaina Odham
// Simple Julia CPU.
// nvcc HW6.cu -o temp -lglut -lGL
// glut and GL are openGL libraries.
/*
 What to do:
 This code displays a simple Julia fractal using the CPU.
 Rewrite the code so that it uses the GPU to create the fractal. 
 Keep the window at 1024 by 1024.
*/

// Include files
#include <stdio.h>
#include <GL/glut.h>

                                                                                    // Add these cuda headers to use the functions, math.h allows square root use
#include <cuda_runtime.h>
#include <math.h>

// Defines
#define MAXMAG 10.0 // If you grow larger than this, we assume that you have escaped.
#define MAXITERATIONS 200 // If you have not escaped after this many attempts, we assume you are not going to escape.
#define A  -0.824	//Real part of C
#define B  -0.1711	//Imaginary part of C

// Global variables
unsigned int WindowWidth = 1024;
unsigned int WindowHeight = 1024;

float XMin = -2.0;
float XMax =  2.0;
float YMin = -2.0;
float YMax =  2.0;

// Function prototypes
void cudaErrorCheck(const char*, int);

                                                                                                                    //remove cpu function, add gpu function
__global__ void juliaKernel(float *d_pixels, int width, int height, float XMin, float XMax, float YMin, float YMax) //create variables for the image dimensions and an array to save the colors to
{
	//assign each thread a pixel
    int i = blockIdx.x * blockDim.x + threadIdx.x; // x vlaue for pixels
    int j = blockIdx.y * blockDim.y + threadIdx.y; // y value for pixels

    if (i >= width || j >= height) return; //make sure to stay within image bounds

	//calculate step size between pixels to map (i,j) to (x,y)
    float stepSizeX = (XMax - XMin) / width;
    float stepSizeY = (YMax - YMin) / height;

	//map (i,j) to (x,y)
    float x = XMin + i * stepSizeX;
    float y = YMin + j * stepSizeY;

	//variables for drawing julia, uses complex numbers
    float mag, tempX; //magnitude of imaginary component and value of x for current calculation
    int count = 0; //number of iterations done

    mag = sqrtf(x * x + y * y); //magnitude formula, used to tell when we have 'escaped'

    while (mag < MAXMAG && count < MAXITERATIONS) //run until we have either escaped or done the max number of iterations (same calculation as before)
    {
        tempX = x;
        x = x * x - y * y + A;
        y = 2.0f * tempX * y + B;
        mag = sqrtf(x * x + y * y);
        count++;
    }

    float color = (count < MAXITERATIONS) ? 0.0f : 1.0f; //becomes black pixel if escaped, becomes red pixel if it did not escape

    int pixelIndex = (j * width + i) * 3;  //location of pixel
    d_pixels[pixelIndex + 0] = color; // Red (0.0f makes it black, 1.0f makes it red)
    d_pixels[pixelIndex + 1] = 0.0f;  // Green
    d_pixels[pixelIndex + 2] = 0.0f;  // Blue
}


void cudaErrorCheck(const char *file, int line)
{
	cudaError_t  error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
		exit(0);
	}
}



void display(void)                                                                                                 //change to use the gpu
{
    int numPixels = WindowWidth * WindowHeight * 3; //number of pixels in image x3 for rgb values to get total number of floats needed
    float *pixels = (float *)malloc(numPixels * sizeof(float)); //allocate memory for the image

    float *d_pixels;
    cudaMalloc((void **)&d_pixels, numPixels * sizeof(float)); //allocate memory for pixel data
	cudaErrorCheck(__FILE__, __LINE__);                                                                            //add error check

	//determine size of image and number of blocks/threads needed to cover the whole image
    dim3 blockSize(16, 16);
    dim3 gridSize((WindowWidth + blockSize.x - 1) / blockSize.x, (WindowHeight + blockSize.y - 1) / blockSize.y);

	//launch kernel
    juliaKernel<<<gridSize, blockSize>>>(d_pixels, WindowWidth, WindowHeight, XMin, XMax, YMin, YMax);
	cudaErrorCheck(__FILE__, __LINE__);                                                                            //add error check

    //move image data to cpu 
	cudaMemcpy(pixels, d_pixels, numPixels * sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);                                                                            //add error check

	//free used memory
    cudaFree(d_pixels);
	cudaErrorCheck(__FILE__, __LINE__);                                                                            //add error check

    glDrawPixels(WindowWidth, WindowHeight, GL_RGB, GL_FLOAT, pixels); //the opengl function for drawing images, lists size of image , uses 3 values per pixel (rgb), values are floats
    glFlush(); //tells it to display the image

    free(pixels); //frees up memory used
}


int main(int argc, char** argv)
{ 
   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(WindowWidth, WindowHeight);
	glutCreateWindow("Fractals--Man--Fractals");
   	glutDisplayFunc(display);
   	glutMainLoop();
}

