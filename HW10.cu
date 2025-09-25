// Name:
// Robust Vector Dot product 
// nvcc HW10.cu -o temp
/*
 What to do:
 This code is the solution to HW9. It computes the dot product of vectors of any length and uses shared memory to 
 reduce the number of calls to global memory. However, because blocks can't sync, it must perform the final reduction 
 on the CPU. 
 To make this code a little less complicated on the GPU lets do some pregame stuff and use atomic adds.
 1. Make sure the number of threads on a block are a power of 2 so we don't have to see if the fold is going to be
    even. Because if it is not even we had to add the last element to the first reduce the fold by 1 and then fold. 
    If it is not even tell your client what is wrong and exit.
 2. Find the right number of blocks to finish the job. But, it is possible that the grid demention is too big. I know
    it is a large number but it is finite. So use device properties to see if the grid is too big for the machine 
    you are on and while you are at it make sure the blocks are not too big too. Maybe you wrote the code on a new GPU 
    but your client is using an old GPU. Check both and if either is out of bound report it to your client then kindly
    exit the program.
 3. Always checking to see if you have threads working past your vector is a real pain and adds a bunch of time consumming
    if statments to your GPU code. To get around this find out how much you would have to add to your vector to make it 
    perfectly fit in your block and grid layout and pad it with zeros. Multipying zeros and adding zero do nothing to a 
    dot product. If you were lucky on HW8 you kind of did this but you just got lucky because most of the time the GPU sets
    everything to zero at start up. But!!!, you don't want to put code out where you are just lucky soooo do a cudaMemset
    so you know everything is zero. Then copy up the now zero values.
 4. In HW9 we had to do the final add "reduction" on the CPU because we can't sync blocks. Use atomic add to get around 
    this and finish the job on the GPU. Also you will have to copy this final value down to the CPU with a cudaMemCopy.
    But!!! We are working with floats and atomics with floats can only be done on GPUs with major compute capability 3 
    or higher. Use device properties to check if this is true. And, while you are at it check to see if you have more
    than 1 GPU and if you do select the best GPU based on compute capablity.
 5. Add any additional bells and whistles to the code that you thing would make the code better and more foolproof.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>

// Defines
#define N 4354649 // Length of the vector                                                            //edit N to test code, originally 100000
#define BLOCK_SIZE 1024 // Threads in a block                                                       //set to 1024,a power of 2, max of what most computers handle
                                                                                                    //set to an odd number to test if user warning works

// Global variables
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU, *B_GPU, *C_GPU; //GPU pointers
float DotCPU, DotGPU;
dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize; //This variable will hold the Dimensions of your grid
float Tolerance = 0.01;

int N_PADDED;                                                                                       //make variable for padded vectors

float *finalResult_GPU;                                                                             //make GPU pointer for final dot product result

// Function prototypes
void cudaErrorCheck(const char *, int);
void setUpDevices();
void allocateMemory();
void innitialize();
void dotProductCPU(float*, float*, int);
__global__ void dotProductGPU(float*, float*, float*, float*, int);                                 //add a float to the GPU prototype for the final result
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
	BlockSize.x = BLOCK_SIZE;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = (N - 1)/BlockSize.x + 1; // This gives us the correct number of blocks.
	GridSize.y = 1;
	GridSize.z = 1;

	N_PADDED = GridSize.x * BlockSize.x;                                                          //set padded vector size as the number of blocks needed times the block size

	printf("Vector size: %d\n", N);
	printf("Padded vector size: %d\n", N_PADDED);
	printf("Block size: %d\n", BLOCK_SIZE);
	printf("Number of blocks: %d\n", GridSize.x);                                                 //check if info is correct
}

// Allocating the memory we will be using.
void allocateMemory()
{	
	// Host "CPU" memory.				                                                          //change to use padded vector size
	A_CPU = (float*)malloc(N_PADDED*sizeof(float));
	B_CPU = (float*)malloc(N_PADDED*sizeof(float));
	C_CPU = (float*)malloc(N_PADDED*sizeof(float));
	
	// Device "GPU" Memory                                                                        //change to use padded vector size
	cudaMalloc(&A_GPU,N_PADDED*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B_GPU,N_PADDED*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C_GPU,N_PADDED*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);

	// Initialize GPU memory to zero                                                              //use cudaMemset to make sure everyithing is initially filled with zeros
	cudaMemset(A_GPU, 0, N_PADDED*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemset(B_GPU, 0, N_PADDED*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemset(C_GPU, 0, N_PADDED*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);

	// Final result memory                                                                        //allocate memory for our final result
	cudaMalloc(&finalResult_GPU, sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);

	// Initialize Final result memory
	cudaMemset(finalResult_GPU, 0, sizeof(float));
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

	// Pad remaining elements with zeros                                                          //add a for loop to do the padding
	for(int i = N; i < N_PADDED; i++)                   // after the data ends at N, run until we reach the end of our padded vector length                                              
	{
		A_CPU[i] = 0.0f;                                //set each element in this area to zero
		B_CPU[i] = 0.0f;
	}
}

// Adding vectors a and b on the CPU then stores result in vector c.
void dotProductCPU(float *a, float *b, float *C_CPU, int n)
{
	for(int id = 0; id < n; id++)
	{ 
		C_CPU[id] = a[id] * b[id];
	}
	
	for(int id = 1; id < n; id++)
	{ 
		C_CPU[0] += C_CPU[id];
	}
}

// This is the kernel. It is the function that will run on the GPU.
// It adds vectors a and b on the GPU then stores result in vector c.
__global__ void dotProductGPU(float *a, float *b, float *c, float *finalResult, int n)            //add final result float
{
	int threadIndex = threadIdx.x;
	int vectorIndex = threadIdx.x + blockDim.x*blockIdx.x;
	__shared__ float c_sh[BLOCK_SIZE];
	
	c_sh[threadIndex] = (a[vectorIndex] * b[vectorIndex]);                    //multiplication step between bectors a and b, stores answers in c
	__syncthreads();
	
	int fold = blockDim.x;
	while(1 < fold)                                                           //folds until only one element left
	{

		/*                                
		if(fold%2 != 0)                                                       //remove this section that occurs when fold is odd since we are set to 1024
		{
			if(threadIndex == 0 && (vectorIndex + fold - 1) < n) 
			{
				c_sh[0] = c_sh[0] + c_sh[0 + fold - 1];
			}
			fold = fold - 1;
		}
		*/


		fold = fold/2;
		if(threadIndex < fold /*&& (vectorIndex + fold) < n*/)                //with the padding we can remove the part that checks our bounds, should be perfect
		{
			c_sh[threadIndex] = c_sh[threadIndex] + c_sh[threadIndex + fold];
			
		}
		__syncthreads();
	}
	
	c[blockDim.x*blockIdx.x] = c_sh[0];

	// Use atomic add for final result                                        //add use of atomic add operation
	if(threadIndex == 0)                                                      //do when there is only one result left in every block
	{
		atomicAdd(finalResult, c_sh[0]);                                      //add all of the values stored in thread zero in each block to the finalResult             
	}
}

// Checking to see if anything went wrong in the vector addition.
bool check(float cpuAnswer, float gpuAnswer, float tolerence)
{
	double percentError;
	
	percentError = abs((gpuAnswer - cpuAnswer)/(cpuAnswer))*100.0;
	printf("\n\n CPU result: %f, GPU result: %f", cpuAnswer, gpuAnswer);                            //print results
	printf("\n\n percent error = %lf\n", percentError);
	
	if(percentError < Tolerance) 
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

	cudaFree(finalResult_GPU);                                              //add cleanup for the final result done with atomic add
	cudaErrorCheck(__FILE__, __LINE__);
}

int main()
{
	if (BLOCK_SIZE%2 == 1)                                                  //quit if blocksize is odd
	{
		printf("The block size is not even so this code will not be accurate.");
		return(0);
	}

	cudaDeviceProp prop;                                                    //use properties fetched in homework 5

	int count;
	cudaGetDeviceCount(&count);                                             //get number of GPU on current machine
	printf("You have %d GPU in this machine\n\n", count);

	int bestGPU = 0;                                                        //make variables used in GPU comparison
	int bestComputeCapability = 0;

	for (int i=0; i < count; i++)
	{
		cudaGetDeviceProperties(&prop, i);                                  //print all relevant info for each GPU
		printf("Info for GPU %d\n", i);
		printf("Name: %s\n", prop.name);
		printf("Compute capability: %d.%d\n", prop.major, prop.minor);
		printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
		printf("Max blocks in grid: %d\n", prop.maxGridSize[0]);
		
		int computeCapability = prop.major * 10 + prop.minor;               //convert compute capability into an integer for easy comparison

		if (computeCapability > bestComputeCapability)                      //if the GPU currently being looked at is better than all the previous ones, set it as such
    	{
        	bestComputeCapability = computeCapability;
        	bestGPU = i;
    	}
	}
	
	if (bestComputeCapability < 30)                                         //end here if the best GPU is not past version 3.0  // mult *(.25) to test
	{
		printf("This computer does not have a GPU with good enough compute capability to run this code.");
		return(0);
	}

    cudaSetDevice(bestGPU);                                                 //use the GPU determined as best
    cudaErrorCheck(__FILE__, __LINE__);

    cudaGetDeviceProperties(&prop, bestGPU);                                //print which GPU was chosen
    printf("\nGPU %d, %s, was determined as best to use in this case.\n\n", bestGPU, prop.name);

	if (prop.maxThreadsPerBlock < BLOCK_SIZE)                               //end here if the computer has less than 1024 threads per block // mult *(.5) to test
	{
		printf("The GPU does not have enough threads per block to support this code.");
		return(0);
	}
	if (prop.maxGridSize[0] < ((N - 1)/BlockSize.x + 1))                    //end here if the computer has less blocks than we need // mult *(.0000000001) to test
	{
		printf("The GPU does not have enough blocks in the grid to support this code.");
		return(0);
	}

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
	dotProductCPU(A_CPU, B_CPU, C_CPU, N);
	DotCPU = C_CPU[0];
	gettimeofday(&end, NULL);
	timeCPU = elaspedTime(start, end);
	
	// Adding on the GPU
	gettimeofday(&start, NULL);
	
	// Copy Memory from CPU to GPU	                                                                      //change to use padded vectors
	cudaMemcpyAsync(A_GPU, A_CPU, N_PADDED*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(B_GPU, B_CPU, N_PADDED*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);

	// set final result to zero before computation                                                        //use cudaMemset to start the atomic add with zero
	cudaMemset(finalResult_GPU, 0, sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	
	dotProductGPU<<<GridSize,BlockSize>>>(A_GPU, B_GPU, C_GPU, finalResult_GPU, N_PADDED);                //change to use padded vectors, add final result variable
	cudaErrorCheck(__FILE__, __LINE__);

	// Copy final result                                                                                  //use cudaMemcpy to get final result got using atomic add
	cudaMemcpy(&DotGPU, finalResult_GPU, sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Copy Memory from GPU to CPU	
	cudaMemcpyAsync(C_CPU, C_GPU, N_PADDED*sizeof(float), cudaMemcpyDeviceToHost);                        //change to use padded vectors
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Making sure the GPU and CPU wiat until each other are at the same place.
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);
	
	/*                                                                                                    //remove CPU reduction, use atomic add instead
	DotGPU = 0.0;
	printf("\nBlock results from GPU:\n");
	for(int i = 0; i < N; i += BlockSize.x)
	{
		printf("Block %d result: %f\n", i/BlockSize.x, C_CPU[i]);
		DotGPU += C_CPU[i]; // C_GPU was copied into C_CPU. 
	}
	printf("Total GPU result after CPU reduction: %f\n", DotGPU);
	*/

	gettimeofday(&end, NULL);
	timeGPU = elaspedTime(start, end);
	
	// Checking to see if all went correctly.
	if(check(DotCPU, DotGPU, Tolerance) == false)
	{
		printf("\n\n Something went wrong in the GPU dot product.\n");
		printf("\n The time it took on the CPU was %ld microseconds", timeCPU);                           //print times for when it goes wrong as well
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


