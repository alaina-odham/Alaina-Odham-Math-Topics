// Name: Alaina Odham
// Device query
// nvcc HW5.cu -o temp
/*
 What to do:
 This code prints out useful information about the GPU(s) in your machine, 
 but there is much more data available in the cudaDeviceProp structure.

 Extend this code so that it prints out all the information about the GPU(s) in your system. 
 Also, and this is the fun part, be prepared to explain what each piece of information means. 
*/

// Include files
#include <stdio.h>

// Defines

// Global variables

// Function prototypes
void cudaErrorCheck(const char*, int);

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

int main()
{
	cudaDeviceProp prop;

	int count;
	cudaGetDeviceCount(&count);
	cudaErrorCheck(__FILE__, __LINE__);
	printf(" You have %d GPUs in this machine\n", count);
	
	for (int i=0; i < count; i++) {
		cudaGetDeviceProperties(&prop, i);
		cudaErrorCheck(__FILE__, __LINE__);
		printf(" ---General Information for device %d ---\n", i);
		printf("Name: %s\n", prop.name);                                   //what GPU you have
		printf("Compute capability: %d.%d\n", prop.major, prop.minor);     //what archetectude cuda is computing with, larger numbers are newer and better
		printf("Clock rate: %d\n", prop.clockRate);                        //clock speed of gpu in kHz
		printf("Device copy overlap: ");                                   //when enabled can move data and run kernels simultaneously
		if (prop.deviceOverlap) printf("Enabled\n");
		else printf("Disabled\n");
		printf("Kernel execution timeout : ");                             //if enabled will automatically stop running a kernel if it takes too long
		if (prop.kernelExecTimeoutEnabled) printf("Enabled\n");
		else printf("Disabled\n");
		printf(" ---Memory Information for device %d ---\n", i);
		printf("Total global mem: %ld\n", prop.totalGlobalMem);            //memory shared across the whole grid
		printf("Total constant Mem: %ld\n", prop.totalConstMem);           //easy access memory for the threads
		printf("Max mem pitch: %ld\n", prop.memPitch);                     //max width allowed when allocating memory
		printf("Texture Alignment: %ld\n", prop.textureAlignment);         //the alignment requirement to access things in the texture memory (memory address starts with a multiple of this number)
		printf(" ---MP Information for device %d ---\n", i);
		printf("Multiprocessor count : %d\n", prop.multiProcessorCount);   //Number of SMs (streaming multiprocessors) on the GPU
		printf("Shared mem per mp: %ld\n", prop.sharedMemPerBlock);        //shared memory for a block
		printf("Registers per mp: %d\n", prop.regsPerBlock);               //number of registers avaliable for the threads in a block to use, typically holds temporary variables
		printf("Threads in warp: %d\n", prop.warpSize);                    //a warp breaks down a block into groups of threads, typically 32
		printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);                                                         //max threads per block
		printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);   //max dimensions a thread can have
		printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);           //max dimensions for a grid
		printf("\n");



        //missing general info
        printf("Integrated GPU: %s\n", prop.integrated ? "Yes" : "No");                //if the CPU and GPU on the same chip
        printf("Can map host memory: %s\n", prop.canMapHostMemory ? "Yes" : "No");     //if the GPU can make and access CPU memory or if it must make copies
        printf("Compute mode: %d\n", prop.computeMode);                                //determines how many cuda kernals can be run, or forbid it entirely
        printf("Concurrent kernels: %s\n", prop.concurrentKernels ? "Yes" : "No");     //if different kernels can run at the same time
        printf("ECC enabled: %s\n", prop.ECCEnabled ? "Yes" : "No");                   //if error correcting code is enabled, helps prevent corruption
        printf("PCI Bus ID: %d\n", prop.pciBusID);                                     //describes physical location of the GPU, broadest field
        printf("PCI Device ID: %d\n", prop.pciDeviceID);                               //also describes the physical location of the GPU
        printf("PCI Domain ID: %d\n", prop.pciDomainID);                               //this describes the physical location of the GPU too, narrowest field
                                                                                       //PCI stands for Peripheral Component Interconnect system
                                                                                       //this info helps select a specific GPU to run when you have multiple GPUs

        //missing memory info
        printf("Texture pitch alignment: %lu bytes\n", (unsigned long)prop.texturePitchAlignment);   //for a 2D texture, defines bytes between starting each row of pixels
        printf("Global L1 cache supported: %s\n", prop.globalL1CacheSupported ? "Yes" : "No");       //stores often accessed info from the global memory for easy access
        printf("Unified addressing: %s\n", prop.unifiedAddressing ? "Yes" : "No");                   //makes it so the same pointer accesses the same info on both the GPU and CPU

        //missing multiprocessor info
        printf("Max shared memory per multiprocessor: %lu bytes\n", (unsigned long)prop.sharedMemPerMultiprocessor); //total memory shared by blocks in one streaming multiprocessor
        printf("Max registers per multiprocessor: %d\n", prop.regsPerMultiprocessor);                                //registers avaliable in one streaming multiprocessor

        //other missing info
        printf("Warp shuffle instructions supported (cooperativeLaunch): %s\n", prop.cooperativeLaunch ? "Yes" : "No");  //allows blocks to communicate with each other
        printf("Managed memory: %s\n", prop.managedMemory ? "Yes" : "No");                                               //if a managed memory accessible by both the CPU and GPU is supported
        printf("Concurrent managed access: %s\n", prop.concurrentManagedAccess ? "Yes" : "No");                          //if the managed memory can be accessed by the GPU and CPU at the same time
        printf("Can execute host callable device functions (canUseHostPointerForRegisteredMem): %s\n", prop.canUseHostPointerForRegisteredMem ? "Yes" : "No");
                                                                                                                         //lets cuda access pre existing host memory without copying it to the GPU first
        printf("Compute preemption supported: %s\n", prop.computePreemptionSupported ? "Yes" : "No");                    //lets you interrupt a kernel and switch to a different one
        printf("Pageable memory access: %s\n", prop.pageableMemoryAccess ? "Yes" : "No");                                //simpler but more inefficient way for GPU to access CPU memory, uses unpinned memory
        printf("Direct managed memory access from host: %s\n", prop.directManagedMemAccessFromHost ? "Yes" : "No");      //If the CPU can access GPU memory

        //The takeaway is to avoid page faults and migration. 
        //If the the GPU tries to access memory on a page, but that page is on the CPU at that time, we get a page fault.
        //It then "migrates" that page from the CPU to the GPU for use.
        //This takes up time and hurts performance, so we try to avoid this.

	}	
	return(0);
}

