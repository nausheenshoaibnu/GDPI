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
#include "rk_gpu.cuh"
#include <stdio.h>
#include <iostream>

using namespace std;

/*********************************************************/
// GPU Vars
__device__ __constant__ char *			dPatterns;						// The Patterns on GPU. Used As String[] / Char[][]
__device__ __constant__ int *         	dHash = NULL;							// The jump arrays for each pattern, on the GPU memory
__device__ __constant__ short *			dPatternLen;						// The patterns lengths, on GPU memory
__device__ __constant__ int             dNumPatterns;

extern char *			d_patterns_ptr;
extern short *			d_pattern_len_ptr;
int *                   d_hash_ptr;
/*********************************************************/


/*********************************************************/
// Host Vars
extern int				nPatterns;						// The number of patterns, on Host RAM
extern short *			hPatternLen;					// Length of patterns, on Host RAM
extern char **			hPatternText;				// Patterns to scan, on Host RAM
/*********************************************************/


/*********************************************************

	Allocates the variables on Host


*/
bool allocate_GPU_pattern_vars_rk(int patterns, int size)
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
		res = cudaMalloc((void**)&d_hash_ptr, sizeof(int)*patterns*2);
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
		    cudaFree(d_hash_ptr);
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
void free_memory_rk()
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
	if(d_hash_ptr != NULL)
	{
	    cudaFree(d_hash_ptr);
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

	Given a vector of string patterns
	Prepocess them and create jump tables
	Then copy them to GPU
*/

extern void PrintTheUnprintable (const char * data , int Size, FILE* file);
int load_patterns_to_gpu_rk(vector<string> pattern, int size)
{
	if(!allocate_GPU_pattern_vars_rk(pattern.size(), size)) {
		return -1;
	}
	
	char* ptrn = new char[size];
	int* hashes = new int[nPatterns * 2];
	memset(ptrn, 0, sizeof(char)*size);
	int i, j;
	for(i=0; i<pattern.size(); i++) {
		try
		{
		    short n = hPatternLen[i] = (short) pattern[i].length();
	    	hPatternText[i] = new char[n+1];
	    	strncpy(hPatternText[i], pattern[i].c_str(), n+1);
	    	hPatternText[i][n] = '\0';
	    	
	    	// Pre Process the pattern
			hashes[i] = rkHash(pattern[i].c_str(), n);
			hashes[i + nPatterns] = calculatePow(n);
			// Column major store
			for(j=0; j<n; j++) {
			    ptrn[(j*nPatterns)+i] = pattern[i][j];
			}
			/*
			if(n <= 5)
			{
			    char text[] = "web = https://slate.com";
			    int len = strlen(text);
			    int hsh = rkHash(text, n);
			    printf("%d %d\n", len, hsh);
			    
			        char* ptr = &ptrn[i];
			    if(hsh == hashes[i])
			    {
			        for(j=0; j<n; j++, ptr += nPatterns)
			        {
			            
			            if(text[j] != *ptr)
			            {
			                break;
			            }
			        }
			        printf("Found at %d: \n", (j == n ? i : -1));
			    } else {
			        int x;
			        for(x=1; x<len-n; x++) {
			            hsh = reHash(text+x, n, hsh, hashes[i + nPatterns]);
			            printf("%d %d\n", x,hsh);
			            if(hsh == hashes[i])
			            {
			                ptr = &ptrn[i];
			                for(j=0; j<n; j++, ptr += nPatterns)
			                {
			                    if(text[x+j] != *ptr)
			                    {
			                        break;
			                    }
			                }
			                printf("Found at %d: \n", (j == n ? i : -1));
			            }
			        }
			    }
			}
			*/
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
	    delete[] hashes;
	    delete[] ptrn;
	    return -2;
	}
	res = cudaMemcpy(d_hash_ptr, hashes, sizeof(int)*nPatterns*2, cudaMemcpyHostToDevice);
	if(res != cudaSuccess) {
	    cout << 2 << " " <<  cudaGetErrorString(res) << endl;
	    delete[] hashes;
	    delete[] ptrn;
	    return -2;
	}
	res = cudaMemcpy(d_pattern_len_ptr, hPatternLen, sizeof(short)*nPatterns, cudaMemcpyHostToDevice);
	if(res != cudaSuccess) {
	    cout << 3 << " " <<  cudaGetErrorString(res) << endl;
	    delete[] hashes;
	    delete[] ptrn;
	    return -2;
	}
	res = cudaMemcpyToSymbol(dPatterns, &d_patterns_ptr, sizeof(char*), 0, cudaMemcpyHostToDevice);
	if(res != cudaSuccess) {
	    cout << 4 << " " <<  cudaGetErrorString(res) << endl;
	    delete[] hashes;
	    delete[] ptrn;
	    return -2;
	}
	res = cudaMemcpyToSymbol(dHash, &d_hash_ptr, sizeof(int*), 0, cudaMemcpyHostToDevice);
	if(res != cudaSuccess) {
	    cout << 5 << " " <<  cudaGetErrorString(res) << endl;
	    delete[] hashes;
	    delete[] ptrn;
	    return -2;
	}
	res = cudaMemcpyToSymbol(dPatternLen, &d_pattern_len_ptr, sizeof(short*), 0, cudaMemcpyHostToDevice);
	if(res != cudaSuccess) {
	    cout << 6 << " " <<  cudaGetErrorString(res) << endl;
	    delete[] hashes;
	    delete[] ptrn;
	    return -2;
	}
	res = cudaMemcpyToSymbol(dNumPatterns, &nPatterns, sizeof(int), 0, cudaMemcpyHostToDevice);
	if(res != cudaSuccess) {
	    cout << 7 << " " <<  cudaGetErrorString(res) << endl;
	    delete[] hashes;
	    delete[] ptrn;
	    return -2;
	}
	
	return 1;
}
/*********************************************************/

/********************************************************
	The GPU packet processor kernel.
	Launches the string matcher for part of the data
	respective of the thread.
********************************************************/

__global__ void process_packet_gpu_rk(char* buffer, int len, int* results)
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
        rkSearch((char*) localBuf, len, i, results);
        i += stride;
    }
}

int calculatePow(int M)
{
	int x = 1;
    // The value of h would be "pow(d, M-1)%q"
    for (int i = 0; i < M-1; i++) {
        x = (x << 8)%Q;
	}
    return (int) x;
}

/********************************************************

	GPU based RK Search
	same as host based.


*/

__device__ int d_strncmp(const char* t, char*p, int n)
{
    int i;
    for(i=0; i<n; i++, p += dNumPatterns)
    {
        if(t[i] != *p)
        {
            return 0;
        }
    }
    return 1;
}

__device__ void rkSearch(
							const char*		text,
							const int		n,
							const int		patternId,
							int *    		results
						  )
{
    const int m = dPatternLen[patternId];
    const int ptrnHash = dHash[patternId];
    const int pow = dHash[patternId + dNumPatterns];
    int h = rkHash(text, m);
    if(h == ptrnHash && d_strncmp(text, dPatterns+patternId, m) ) {
        results[0] = 0;
        results[1] = patternId;
        return;
    }
    for(int i = 1; i<n-m; i++)
    {
        h = reHash(text+i, m, h, pow);
        if(h == ptrnHash && d_strncmp(text+i, dPatterns+patternId, m) )
        {
            results[0] = i;
            results[1] = patternId;
            return;
        }
    }
}

/************************************************************
*		The RK Algorithm
*/
__host__ __device__ int rkHash(const char* str, int len)
{
	int hh = 0;
    // Calculate the hash value of pattern and first
    // window of text
    for (int i = 0; i < len; i++)
    {
    	hh = ((hh << 8) + str[i])%Q;
    }
    
    return hh;
}
__host__ __device__ int reHash(const char* str, int len, int pre, int h)
{
	pre = (pre - *(str-1)*h)%Q;
	pre = (((pre << 8)%Q) + str[len-1])%Q;
	if(pre < 0)
	    pre += Q;
	return pre;
}
