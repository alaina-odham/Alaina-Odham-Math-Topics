// Name: Alaina Odahm
// GPU random walk. 
// nvcc P_GPURandomWalk.cu -o temp

/*
 What to do:
 This code runs a random walk for 10,000 steps on the CPU.

 1. Use cuRAND to run 20 random walks simultaneously on the GPU, each with a different seed.
    Print out all 20 final positions.

 2. Use cudaMallocManaged(&variable, amount_of_memory_needed);
    This allocates unified memory, which is automatically managed between the CPU and GPU.
    You lose some control over placement, but it saves you from having to manually copy data
    to and from the GPU.
*/

/*
 Purpose:
 To learn how to use cuRAND and unified memory.
*/

/*
 Note:
 The maximum signed int value is 2,147,483,647, so the maximum unsigned int value is 4,294,967,295.

 RAND_MAX is guaranteed to be at least 32,767. When I checked it on my laptop (10/6/2025), it was 2,147,483,647.
 rand() returns a value in [0, RAND_MAX]. It actually generates a list of pseudo-random numbers that depends on the seed.
 This list eventually repeats (this is called its period). The period is usually 2³¹ = 2,147,483,648,
 but it may vary by implementation.

 Because RAND_MAX is odd on this machine and 0 is included, there is no exact middle integer.
 Casting to float as in (float)RAND_MAX / 2.0 divides the range evenly.
 Using integer division (RAND_MAX / 2) would bias results slightly toward the positive side by one value out of 2,147,483,647.

 I know this is splitting hares (sorry, rabbits), but I'm just trying to be as accurate as possible.
 You might do this faster with a clever integer approach, but I’m using floats here for clarity.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>
#include <curand_kernel.h>                                                               //add this to use GPU random number generator

// Defines

// Globals
int NumberOfRandomSteps = 10000;
float MidPoint = (float)RAND_MAX/2.0f;

// Function prototypes
int getRandomDirection();
int main(int, char**);

__global__ void randomWalkKernel(int *posX, int *posY, int steps, unsigned long long s); //new gpu function prototype

int getRandomDirection()
{	
	int randomNumber = rand();
	
	if(randomNumber < MidPoint) return(-1);
	else return(1);
}

__global__ void randomWalkKernel(int *posX, int *posY, int steps, unsigned long long s) //two arrays to store the final x and y positions, variable for number of steps to take, and seed float
{
    int id = threadIdx.x;                                                               //one thread per walker

    curandState state;                                                                  //create cuRAND state, variable to hold random numbers
    curand_init(s + id, 0, 0, &state);                                                  //each walk at different seed, dont bother with subsequences, 0 to start at begining of sequence

    for (int i = 0; i < steps; i++)                                                     //run 10000 times
	{
        float randX = curand_uniform(&state);                                           //get a random float between zero and one for x and y, get new random number from state
        float randY = curand_uniform(&state);

        if (randX < 0.5f)                                                               //move left
		{
    		posX[id] += -1;
		} 
		else                                                                            //move right
		{
    		posX[id] += 1;
		}

		if (randY < 0.5f)                                                               //move backwards
		{
	    	posY[id] += -1;
		} 
		else                                                                            //move forward
		{
    		posY[id] += 1;
		}
    } 
}


int main(int argc, char** argv)
{
	/*
	srand(time(NULL));
	*/

	unsigned long long seed = time(NULL);                                              //base seed on time
	
	printf(" RAND_MAX for this implementation is = %d \n", RAND_MAX);                  //print randmax

	/*
	int positionX = 0;
	int positionY = 0;
	*/

	int *positionX, *positionY;                                                        //allocate memory for 20 walks
	cudaMallocManaged(&positionX, 20 * sizeof(int));
	cudaMallocManaged(&positionY, 20 * sizeof(int));

	for (int i = 0; i < 20; i++)                                                      //initialize positions at (0,0)
	{
    	positionX[i] = 0;
    	positionY[i] = 0;
	}	

	/*
	for(int i = 0; i < NumberOfRandomSteps; i++)
	{
		positionX += getRandomDirection();
		positionY += getRandomDirection();
	}
	*/

	randomWalkKernel<<<1, 20>>>(positionX, positionY, NumberOfRandomSteps, seed);    //launch 20 threads, one random walk each
	cudaDeviceSynchronize();

	for (int i = 0; i < 20; i++)                                                     //print results
	{
    	printf("Walker %d final position = (%d, %d)\n", i, positionX[i], positionY[i]);
	}

	cudaFree(positionX);                                                             //clean up
	cudaFree(positionY);

	/*
	printf("\n Final position = (%d,%d) \n", positionX, positionY);
	*/

	return 0;
}

