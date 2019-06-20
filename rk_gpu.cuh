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
#pragma once

#ifndef RK_GPU_H
#define RK_GPU_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <string>

using namespace std;

#ifndef NULL
	#define NULL 0
#endif
#ifndef null
	#define null NULL
#endif

// d is the number of characters in input alphabet
#define D 256
#define Q 7996369

// PreProcess and Save to GPU. Allocates dPattern[] and dJump[]
int load_patterns_to_gpu_rk(vector<string> pattern, int);

// Creates dPatterns[] and dJumps[], DONT USE EXPLICITLY.
bool allocate_GPU_pattern_vars_rk(int patterns, int);

// Last function to free all variables
void free_memory_rk();

// Packet processing
__global__ void process_packet_gpu_rk(char* buffer, int len, int* results);

/************************************************************
*		The RK Algorithm
*/

int calculatePow(int M);

__host__ __device__ int rkHash(const char* str, int len);
__host__ __device__ int reHash(const char* str, int len, int pre, int h);
__device__ void rkSearch(
							const char*		text,
							const int		n,
							const int		patternId,
							int *    		results
						  );


/*************************************************************/

#endif
