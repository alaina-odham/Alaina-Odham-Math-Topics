// Name: Alaina Odham
// Not simple Julia Set on the GPU
// nvcc HW7.cu -o temp -lglut -lGL

/*
 What to do:
 This code displays a simple Julia set fractal using the GPU.
 But it only runs on a window of 1024X1024.
 Extend it so that it can run on any given window size.
 Also, color it to your liking. I will judge you on your artisct flare. 
 Don't cute off your ear or anything but make Vincent wish he had, had a GPU.
*/

// Include files
#include <stdio.h>
#include <GL/glut.h>

#include <cuda_runtime.h>
#include <math.h>

// Defines
#define MAXMAG 10.0 // If you grow larger than this, we assume that you have escaped.

                                                                                                                //can edit the iterations as needed
#define MAXITERATIONS 200 // If you have not escaped after this many attempts, we assume you are not going to escape.


#define A  -0.824	//Real part of C
#define B  -0.1711	//Imaginary part of C

// Global variables

float timeStep = 0.0f;                                                                                          //to keep track of time

                                                                                                                //can change these values at will
unsigned int WindowWidth = 1024;
unsigned int WindowHeight = 1024;

float XMin = -2.0;
float XMax =  2.0;
float YMin = -2.0;
float YMax =  2.0;

// Function prototypes
void cudaErrorCheck(const char*, int);
                                                                                                                 //now added a time variable
__global__ void juliaKernel(float *d_pixels, int width, int height, float XMin, float XMax, float YMin, float YMax, float timeStep)
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

    mag = sqrtf(x * x + y * y);

    while (mag < MAXMAG && count < MAXITERATIONS)
    {
        tempX = x;
        x = x * x - y * y + A;
        y = 2.0f * tempX * y + B;
        mag = sqrtf(x * x + y * y);
        count++;
    }

                                                                                                                //messing around with colors
    
    float colorvariant = (float)i / (float)width;

    float cvarA = sinf(timeStep);                 //color change over time
    float cvarC = fmodf(timeStep * 1.0f, 1.0f)*9;   //smooth color change

    float cvar2 = (float)j / (float)height*cvarA;
    float cvar3 = (float)sinf(count+cvarA+(float)y*cvarA)+ cvarA;
    float cvar4;

    if((float)x+cvarA > (float)x*0.5f)
    {
        cvar4 = 1.0f+cvarA*.5;
    }
    else
    {
        cvar4 = 0.0f+cvarA*.5;
    }

    float cvar5 = 0.5f-cvar4;

    float cvar6;

    if((float)y*cvarA > (float)y*0.5f+cvarC+cvarA)
    {
        cvar6 = 0.5f+cvarA*.5;
    }
    else
    {
        cvar6 = 0.0f+cvarA*.5;
    }

    float cvar7 = 0.5f-cvar6;

    float cvar8 = (float)cosf((float)x +cvarA)+cvarA;

    float r = (count < MAXITERATIONS) ? cvar3 : 3.0f+cvarA*2+cvarC;
    float g = (count < MAXITERATIONS) ? cvar2+cvar7 : 3.0f-cvarA;
    float b = (count < MAXITERATIONS) ? cvar5-cvar8+cvarA : colorvariant*2+cvarA*.5;


    int pixelIndex = (j * width + i) * 3;  //location of pixel
    d_pixels[pixelIndex + 0] = r;  // Red
    d_pixels[pixelIndex + 1] = g;  // Green
    d_pixels[pixelIndex + 2] = b;  // Blue
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



void display(void)                                                                                                 
{

    timeStep += 0.01f;                                                                        //to have the time update as the image is being displayed

	//code you can use to mess around with window size, uncomment as needed
	//WindowWidth = 500;
    //WindowHeight = 500;
	//glutReshapeWindow(WindowWidth, WindowHeight); //must use this function when manually entering sizes or the window size will not mach image size

	//code that lets you resize window with your mouse instead of manually entering values, uncomment as needed
	WindowWidth = glutGet(GLUT_WINDOW_WIDTH);
    WindowHeight = glutGet(GLUT_WINDOW_HEIGHT);

	//code so you can see what the dimensions are
	//printf("\nWindowWidth = %d, \nWindowHeight = %d\n", WindowWidth, WindowHeight);

    int numPixels = WindowWidth * WindowHeight * 3; //number of pixels in image x3 for rgb values to get total number of floats needed
    float *pixels = (float *)malloc(numPixels * sizeof(float)); //allocate memory for the image

    float *d_pixels;
    cudaMalloc((void **)&d_pixels, numPixels * sizeof(float)); //allocate memory for pixel data
	cudaErrorCheck(__FILE__, __LINE__);                                                                            

	//determine size of image and number of blocks/threads needed to cover the whole image
    dim3 blockSize(16, 16);
    dim3 gridSize((WindowWidth + blockSize.x - 1) / blockSize.x, (WindowHeight + blockSize.y - 1) / blockSize.y);

	//launch kernel
    juliaKernel<<<gridSize, blockSize>>>(d_pixels, WindowWidth, WindowHeight, XMin, XMax, YMin, YMax, timeStep);                         //added timestep
	cudaErrorCheck(__FILE__, __LINE__);                                                                            

    //move image data to cpu 
	cudaMemcpy(pixels, d_pixels, numPixels * sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);                                                                            

	//free used memory
    cudaFree(d_pixels);
	cudaErrorCheck(__FILE__, __LINE__);                                                                            

    glDrawPixels(WindowWidth, WindowHeight, GL_RGB, GL_FLOAT, pixels);
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

    glutIdleFunc(display);                                                                                              //animation line, updates continuously

   	glutMainLoop();
}



