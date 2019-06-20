/**
 * 
 * 
 * 
 * 
 * Designed and Developed By:
	Tahir Mustafa	-	tahir.mustafa53@gmail.com / k132162@nu.edu.pk
	Akhtar Zaman	-	k132168@nu.edu.pk
	Jazib ul Hassan	-	k132138@nu.edu.pk
	Mishal Gohar	-	k132184@nu.edu.pk
 * 
 * For BS(CS) Final Year Project 2017, NUCES-FAST
 * Under the supervision of:
	Dr Jawwad Shamsi (HOD CS Department)
	Miss Nausheen Shoaib
 * With due gratitude to NVIDIA Research Lab, NUCES-FAST

 * This code is the intellectual property of the authors,
 * available for use under Academic and Educational purposes
 * only.
 * The Authors reserve the rights to this code
 * and related material.
 * 
 * Copyrights 2017
 * 
 * 
 * 
 * 
*/
#include<sys/socket.h>
#include<arpa/inet.h> // for inet_ntoa()
#include<net/ethernet.h>
#include<netinet/ip_icmp.h>   //Provides declarations for icmp header
#include<netinet/udp.h>   //Provides declarations for udp header
#include<netinet/tcp.h>   //Provides declarations for tcp header
#include<netinet/ip.h>    //Provides declarations for ip header

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "kmp_gpu.cuh"
#include <stdio.h>
#include <iostream>

using namespace std;

/*********************************************************/
// GPU Vars
__device__ __constant__ char *			dPatterns = NULL;						// The Patterns on GPU. Used As String[] / Char[][]
__device__ __constant__ short *     	dJumps = NULL;							// The jump arrays for each pattern, on the GPU memory
__device__ __constant__ short *			dPatternLen = NULL;						// The patterns lengths, on GPU memory
__device__ __constant__ int             dNumPatterns = 0;

char *			d_patterns_ptr = NULL;
short *     	d_jumps_ptr = NULL;
short *			d_pattern_len_ptr = NULL;
/*********************************************************/


/*********************************************************/
// Host Vars
int				nPatterns = 0;						// The number of patterns, on Host RAM
short *			hPatternLen = NULL;					// Length of patterns, on Host RAM
char **			hPatternText = NULL;				// Patterns to scan, on Host RAM
/*********************************************************/


/*********************************************************

	Allocates the variables on Host


*/
bool allocate_GPU_pattern_vars(int patterns, int size)
{
	try {
		hPatternLen = new short[patterns];
		hPatternText = new char*[patterns];
		nPatterns = patterns;
		
		cudaError_t res;
		res = cudaMalloc((void**)&d_patterns_ptr, sizeof(char)*size);
		if(res != cudaSuccess) {
	        delete[] hPatternLen;
	        delete[] hPatternText;
		    return false;
		}
		res = cudaMalloc((void**)&d_jumps_ptr, sizeof(short)*size);
		if(res != cudaSuccess) {
	        delete[] hPatternLen;
	        delete[] hPatternText;
		    cudaFree(d_patterns_ptr);
		    return false;
		}
		res = cudaMalloc((void**)&d_pattern_len_ptr, sizeof(short)*patterns);
		if(res != cudaSuccess) {
	        delete[] hPatternLen;
	        delete[] hPatternText;
		    cudaFree(d_patterns_ptr);
		    cudaFree(d_jumps_ptr);
		    return false;
		}
		return true;
	} catch(exception e) {
	    cout << e.what() << endl;
	    if(hPatternLen != NULL) {
	        delete[] hPatternLen;
	    }
	    if(hPatternText != NULL) {
	        delete[] hPatternText;
	    }
		return false;
	}
}
/*********************************************************/


/*********************************************************

	Free the allocated GPU and host memories


*/
void free_memory()
{
	int i;
	if(d_patterns_ptr != NULL)
	{
	    cudaFree(d_patterns_ptr);
	}
	if(hPatternText != NULL)
	{
		for(i=0; i<nPatterns; i++)
		{
			if(hPatternText[i] != NULL)
				delete[] hPatternText[i];
		}
		delete[] hPatternText;
	}
	if(d_jumps_ptr != NULL)
	{
	    cudaFree(d_jumps_ptr);
	}
	if(hPatternLen != NULL)
	{
		delete[] hPatternLen;
	}
	if(d_pattern_len_ptr != NULL)
	{
		cudaFree(d_pattern_len_ptr);
	}
	nPatterns = 0;
}
/*********************************************************/


/********************************************************

	An general function which allocates a GPU variable
	Copies the given data, and returns the pointer to
	the copied data.


*/
void* copy_to_gpu(void *data, int size)
{
	char *dVar;
	cudaError_t cudaRes;
	if((cudaRes = cudaMalloc((void **)&dVar, size)) != cudaSuccess ) {
        printf("Memory Allocation Failed!\n%s\n", cudaGetErrorString(cudaRes));
		return NULL;
    }
	if((cudaRes = cudaMemcpy(dVar, data, size, cudaMemcpyHostToDevice)) != cudaSuccess)
	{
        printf("Memory Allocation Failed!\n%s\n", cudaGetErrorString(cudaRes));
		cudaFree(dVar);
		return NULL;
	}
	return (void*) dVar;
}
/*********************************************************/


/********************************************************

	Given a vector of string patterns
	Prepocess them and create jump tables
	Then copy them to GPU


*/
int load_patterns_to_gpu(vector<string> pattern, int size)
{
	if(!allocate_GPU_pattern_vars(pattern.size(), size)) {
		return -1;
	}
	
	char* ptrn = new char[size];
	short* jmps = new short[size];
	memset(ptrn, 0, sizeof(char)*size);
	memset(jmps, 0, sizeof(short)*size);
	int i, j;
	for(i=0; i<pattern.size(); i++) {
		try
		{
		    short n = hPatternLen[i] = (short) pattern[i].length();
	    	hPatternText[i] = new char[n+1];
	    	strncpy(hPatternText[i], pattern[i].c_str(), n+1);
	    	hPatternText[i][n] = '\0';
	    	
	    	// Pre Process the pattern
	    	short *jump = new short[n];
			preProcess(pattern[i].c_str(), n, jump);
			
			// Column major store
			for(j=0; j<n; j++) {
			    ptrn[(j*pattern.size())+i] = pattern[i][j];
			    jmps[(j*pattern.size())+i] = jump[j];
			}
			//cout << pattern[i] << " " << hPatternText[i] << endl;
			
			delete[] jump;
		}
		catch(exception e)
		{
		    cout << e.what() << endl;
			return -3;
		}
	}
	cudaError_t res;
	res = cudaMemcpy(d_patterns_ptr, ptrn, sizeof(char)*size, cudaMemcpyHostToDevice);
	if(res != cudaSuccess) {
	    cout << 1 << " " << cudaGetErrorString(res) << endl;
	    delete[] jmps;
	    delete[] ptrn;
	    return -2;
	}
	res = cudaMemcpy(d_jumps_ptr, jmps, sizeof(short)*size, cudaMemcpyHostToDevice);
	if(res != cudaSuccess) {
	    cout << 2 << " " <<  cudaGetErrorString(res) << endl;
	    delete[] jmps;
	    delete[] ptrn;
	    return -2;
	}
	res = cudaMemcpy(d_pattern_len_ptr, hPatternLen, sizeof(short)*nPatterns, cudaMemcpyHostToDevice);
	if(res != cudaSuccess) {
	    cout << 3 << " " <<  cudaGetErrorString(res) << endl;
	    delete[] jmps;
	    delete[] ptrn;
	    return -2;
	}
	res = cudaMemcpyToSymbol(dPatterns, &d_patterns_ptr, sizeof(char*), 0, cudaMemcpyHostToDevice);
	if(res != cudaSuccess) {
	    cout << 4 << " " <<  cudaGetErrorString(res) << endl;
	    delete[] jmps;
	    delete[] ptrn;
	    return -2;
	}
	res = cudaMemcpyToSymbol(dJumps, &d_jumps_ptr, sizeof(short*), 0, cudaMemcpyHostToDevice);
	if(res != cudaSuccess) {
	    cout << 5 << " " <<  cudaGetErrorString(res) << endl;
	    delete[] jmps;
	    delete[] ptrn;
	    return -2;
	}
	res = cudaMemcpyToSymbol(dPatternLen, &d_pattern_len_ptr, sizeof(short*), 0, cudaMemcpyHostToDevice);
	if(res != cudaSuccess) {
	    cout << 6 << " " <<  cudaGetErrorString(res) << endl;
	    delete[] jmps;
	    delete[] ptrn;
	    return -2;
	}
	res = cudaMemcpyToSymbol(dNumPatterns, &nPatterns, sizeof(int), 0, cudaMemcpyHostToDevice);
	if(res != cudaSuccess) {
	    cout << 7 << " " <<  cudaGetErrorString(res) << endl;
	    delete[] jmps;
	    delete[] ptrn;
	    return -2;
	}
	
	
	return 1;
}
/*********************************************************/


/********************************************************

	KMP Preprocess function
	IDK how it works
*/
void preProcess(const  char* pat, int m, short* jump)
{
    short i = 1, j = 0;

    jump[0] = 0;

    while (i < m) {
        if (pat[i] == pat[j]) {
            jump[i] = j + 1;
            i++;
            j++;
        } else if (j > 0) {
            j = jump[j - 1];
        } else {
            jump[i] = 0;
            i++;
        }
    }
}

/********************************************************
	The GPU packet processor kernel.
	Launches the string matcher for part of the data
	respective of the thread.
********************************************************/

__global__ void process_packet_gpu(char* buffer, int len, int* results)
{
    extern __shared__ char localBuf[];
    int i, stride;
    for(i = threadIdx.x; i < len; i += blockDim.x) {
        localBuf[i] = buffer[i];
    }
    __syncthreads();
    i = (blockIdx.x * blockDim.x + threadIdx.x);
    stride = (blockDim.x * blockDim.y * blockDim.z * gridDim.x);
    while(i < dNumPatterns)
    {
        kmpSearch((char*) localBuf, len, i, results);
        i += stride;
    }
}

/********************************************************

	GPU based KMP Search
	same as host based.


*/

__device__ void kmpSearch(
							char*		text,
							int			n,
							int			patternId,
							int *		results
						  )
{
    int j=0, i=0, ptr = patternId, m = dPatternLen[patternId];
    while (i < n) {
        if (text[i] == dPatterns[ptr]) {
            i++;
            j++;
            ptr += dNumPatterns;
        } else if (j > 0) {
            ptr -= dNumPatterns;
            j = dJumps[ptr];
            ptr = (dNumPatterns * j) + patternId;
        } else {
            i++;
        }
        if (j == m) {
        	results[0] = i-j;
        	results[1] = patternId;
        }
    }
}

