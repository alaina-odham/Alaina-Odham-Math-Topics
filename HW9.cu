// Name: Alaina Odham
// Vector Dot product on many block and useing shared memory
// nvcc HW9.cu -o temp
/*
 What to do:
 This code is the solution to HW8. It finds the dot product of vectors that are smaller than the block size.
 Extend this code so that it sets as many blocks as needed for a set thread count and vector length.
 Use shared memory in your blocks to speed up your code.
 You will have to do the final reduction on the CPU.
 Set your thread count to 200 (block size = 200). Set N to different values to check your code.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>
#include <math.h>                                                                                     //to use fabs() in error check

// Defines
#define N 9867                                                                                        //edit length of the vector
#define BLOCK_SIZE 200                                                                                //set thread count to 200

// Global variables
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU, *B_GPU, *C_GPU; //GPU pointers
float DotCPU, DotGPU;
dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize; //This variable will hold the Dimensions of your grid
float Tolerance = 0.01;

// Function prototypes
void cudaErrorCheck(const char *, int);
void setUpDevices();
void allocateMemory();
void innitialize();
void dotProductCPU(float*, float*, int);
__global__ void dotProductGPU(float*, float*, float*, int);
bool  check(float, float, float);
long elaspedTime(struct timeval, struct timeval);
void cleanUp();

// This check to see if an error happened in your CUDA code. It tell you what it thinks went wrong,
// and what file and line it occured on.
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

// This will be the layout of the parallel space we will be using.
void setUpDevices()
{
	BlockSize.x = BLOCK_SIZE;                                                          //set to 200
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = ((N - 1)/BLOCK_SIZE) +1;                                              //calculate blocks needed
	GridSize.y = 1;
	GridSize.z = 1;

	printf("Vector size: %d\n", N);
	printf("Block size: %d\n", BLOCK_SIZE);
	printf("Number of blocks: %d\n", GridSize.x);                                      //check output is correct
}

// Allocating the memory we will be using.
void allocateMemory()
{	
	// Host "CPU" memory.				
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(GridSize.x*sizeof(float));                                 //changed ouput to per block
	
	// Device "GPU" Memory
	cudaMalloc(&A_GPU,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B_GPU,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C_GPU,GridSize.x*sizeof(float));                                      //changed ouput to per block
	cudaErrorCheck(__FILE__, __LINE__);
}

// Loading values into the vectors that we will add.
void innitialize()
{
	for(int i = 0; i < N; i++)
	{		
		A_CPU[i] = (float)i;	
		B_CPU[i] = (float)(3*i);
	}
}

// Adding vectors a and b on the CPU then stores result in vector c.
void dotProductCPU(float *a, float *b, int n)                                         //remove c float
{
	DotCPU = 0.0f;                                                                    //define variable for CPU answer

    for(int id = 0; id < n; id++)                                                     //multiply each element and add as you go
    { 
        DotCPU += a[id] * b[id];                                                      //store directly in DotCPU
    }
}

// This is the kernel. It is the function that will run on the GPU.
// It adds vectors a and b on the GPU then stores result in vector c.
__global__ void dotProductGPU(float *a, float *b, float *c, int n)
{
	// Shared memory for this block
	__shared__ float sdata[200];       		                                          //add shared memory

	/*
	int id = threadIdx.x;                                                             //remove old method
	
	c[id] = a[id] * b[id];
	__syncthreads();
		
	int fold = blockDim.x;
	while(1 < fold)
	{
		if(fold%2 != 0)
		{
			if(id == 0 && (fold - 1) < n)
			{
				c[0] = c[0] + c[fold - 1];
			}
			fold = fold - 1;
		}
		fold = fold/2;
		if(id < fold && (id + fold) < n)
		{
			c[id] = c[id] + c[id + fold];
		}
		__syncthreads();
	}
	*/

	// Global thread ID                                                                 //new method
	int tid = blockIdx.x * blockDim.x + threadIdx.x;                                    //define thread id variables for use
	int localTid = threadIdx.x;
	
	if(tid < n)                                                                         //have each thread compute one element of the dot product
	{
		sdata[localTid] = a[tid] * b[tid];
	} 
	else                                                                                //out of bounds
	{
		sdata[localTid] = 0.0f;
	}
	
	// Synchronize to make sure all threads have loaded their data
	__syncthreads();
	
	// Perform reduction in shared memory
	// Modified to handle blocks with fewer elements than threads
	int activeElements = min(blockDim.x, n - blockIdx.x * blockDim.x);                  //define variable that determines elements left in current block

	for(int stride = 1; stride < activeElements; stride *= 2)                           //calculate stride needed
	{
	    if(localTid % (2 * stride) == 0 && localTid + stride < activeElements)          //stay within bounds
    	{
        	sdata[localTid] += sdata[localTid + stride];                                //add the elements and save them
    	}
    	__syncthreads();
	}
	
	// Thread 0 writes the result for this block
	if(localTid == 0) 
	{
		c[blockIdx.x] = sdata[0];                                                       //store final result
	}
}

// Checking to see if anything went wrong in the vector addition.
bool check(float cpuAnswer, float gpuAnswer, float tolerence)
{
	double percentError;
	
	percentError = fabs((gpuAnswer - cpuAnswer)/(cpuAnswer))*100.0;                     //change to fabs for floating points
	printf("\n\n CPU result: %f, GPU result: %f", cpuAnswer, gpuAnswer);                //print results
	printf("\n\n percent error = %lf\n", percentError);
	
	if(percentError < tolerence) 
	{
		return(true);
	}
	else 
	{
		return(false);
	}
}

// Calculating elasped time.
long elaspedTime(struct timeval start, struct timeval end)
{
	// tv_sec = number of seconds past the Unix epoch 01/01/1970
	// tv_usec = number of microseconds past the current second.
	
	long startTime = start.tv_sec * 1000000 + start.tv_usec; // In microseconds.
	long endTime = end.tv_sec * 1000000 + end.tv_usec; // In microseconds

	// Returning the total time elasped in microseconds
	return endTime - startTime;
}

// Cleaning up memory after we are finished.
void CleanUp()
{
	// Freeing host "CPU" memory.
	free(A_CPU); 
	free(B_CPU); 
	free(C_CPU);
	
	cudaFree(A_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(B_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(C_GPU);
	cudaErrorCheck(__FILE__, __LINE__);
}

int main()
{
	timeval start, end;
	long timeCPU, timeGPU;
	//float localC_CPU, localC_GPU;
	
	// Setting up the GPU
	setUpDevices();
	
	// Allocating the memory you will need.
	allocateMemory();
	
	// Putting values in the vectors.
	innitialize();
	
	// Adding on the CPU
	gettimeofday(&start, NULL);
	dotProductCPU(A_CPU, B_CPU, N);                                                      //simplify
	gettimeofday(&end, NULL);
	timeCPU = elaspedTime(start, end);
	
	/*                                                                                   //no longer needed
	if(BlockSize.x < N)
	{
		printf("\n\n Your vector size is larger than the block size.");
		printf("\n Because we are only using one block this will not work.");
		printf("\n Good Bye.\n\n");
		exit(0);
	}
	*/
	
	// Adding on the GPU
	gettimeofday(&start, NULL);
	
	// Copy Memory from CPU to GPU		
	cudaMemcpyAsync(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	dotProductGPU<<<GridSize,BlockSize>>>(A_GPU, B_GPU, C_GPU, N);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Copy Memory from GPU to CPU	
	cudaMemcpyAsync(C_CPU, C_GPU, GridSize.x*sizeof(float), cudaMemcpyDeviceToHost);             //change 1 to GridSize.x
	cudaErrorCheck(__FILE__, __LINE__);
	//DotGPU = C_CPU[0]; // C_GPU was copied into C_CPU.                                         //remove this line
	
	// Making sure the GPU and CPU wiat until each other are at the same place.
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);

	// Final reduction on CPU - sum all partial results from blocks                              //add up block results
	DotGPU = 0.0f;                                                                 //define result variable
	printf("\nDebug: Block results from GPU:\n");                                  //print each block result
	for(int id = 0; id < GridSize.x; id++) 
	{
		printf("Block %d result: %f\n", id, C_CPU[id]);
		DotGPU += C_CPU[id];                                                       //add up results
	}
	printf("Total GPU result after CPU reduction: %f\n", DotGPU);                  //print final result


	gettimeofday(&end, NULL);
	timeGPU = elaspedTime(start, end);
	
	// Checking to see if all went correctly.
	if(check(DotCPU, DotGPU, Tolerance) == false)
	{
		printf("\n\n Something went wrong in the GPU dot product.\n");
		printf("\n The time it took on the CPU was %ld microseconds", timeCPU);    //have it print times here too
		printf("\n The time it took on the GPU was %ld microseconds", timeGPU);
	}
	else
	{
		printf("\n\n You did a dot product correctly on the GPU");
		printf("\n The time it took on the CPU was %ld microseconds", timeCPU);
		printf("\n The time it took on the GPU was %ld microseconds", timeGPU);
	}
	
	// Your done so cleanup your room.	
	CleanUp();	
	
	// Making sure it flushes out anything in the print buffer.
	printf("\n\n");
	
	return(0);
}


