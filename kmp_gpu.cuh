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

#ifndef KMP_GPU_H
#define KMP_GPU_H

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


// PreProcess and Save to GPU. Allocates dPattern[] and dJump[]
int load_patterns_to_gpu(vector<string> pattern, int);

// Creates dPatterns[] and dJumps[], DONT USE EXPLICITLY.
bool allocate_GPU_pattern_vars(int patterns, int);

// Last function to free all variables
void free_memory();

// Allocate and copy data to GPU. Return allocated pointer.
void* copy_to_gpu(void *data, int size);

// Packet processing gpu pe hogayi
__global__ void process_packet_gpu(char* buffer, int len, int* results);

/************************************************************/
/*
*
*		The KMP Algorithm <3
*
*/
void preProcess(const  char* pat, int m, short* jump);

__device__ void kmpSearch(
							char*		text,
							int			n,
							int			patternId,
							int *		results
						  );


/*************************************************************/

#endif
