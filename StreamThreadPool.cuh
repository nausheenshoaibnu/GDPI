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
#ifndef STREAM_THREAD_POOL_CUH
#define STREAM_THREAD_POOL_CUH

#ifndef null
    #define null 0
#endif

#ifndef NULL 
    #define NULL 0 
#endif

#define THREAD_POOL_DEFAULT_SIZE 4

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "list.cuh"
#include "kmp_gpu.cuh"
#include "rk_gpu.cuh"
#include "config.cuh"
#include "reset.h"

/*********************************************************/
// Host Vars
extern int				nPatterns;						// The number of patterns, on Host RAM
extern short*			hPatternLen;					// Length of patterns, on Host RAM
extern char **			hPatternText;					// Patterns to scan, on Host RAM
/*********************************************************/

void PrintTheUnprintable (const char * data , int Size, FILE* file);

class StreamThreadPool;

typedef struct threadPoolArgs_t {
    int id;
    StreamThreadPool *pool;
} threadPoolArgs;

typedef struct process_packet_args_t {
    u_char * buffer;
    int size;
} PACKET_T;

class StreamThreadPool {
  protected:
    pthread_t *threads;
    cudaStream_t *streams;
    FILE* Logfile;
    
    bool running;
    int size;
    
    Queue<PACKET_T*> *workQueue;
    
    StreamThreadPool() { StreamThreadPool(THREAD_POOL_DEFAULT_SIZE); }
    
  public:
    long bytes;
    float packetTime;
    int counter;
    
    StreamThreadPool(int s): workQueue() {
        threads = new pthread_t[s];
        streams = new cudaStream_t[s];
        workQueue = new Queue<PACKET_T*>();
        size = s;
        packetTime = bytes = counter = 0;
        logfile = stdout;
    }
    
    bool setLogFile(char* filename) {
        Logfile = fopen(filename, "a");
        if(!Logfile) {
            Logfile = stdout;
            return false;
        }
        // Time reference from
        // www.stackoverflow.com/questions/5141960/get-the-current-time-in-c
        time_t mytime;
        mytime = time(NULL);
        return true;
    }
    
    void startPool() {
        running = true;
        fprintf(logfile, "\r\n\r\n*** Program started on : %s\r\n", ctime(&mytime));
        int i;
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
        int rc;
        cudaError_t res;
        for(i=0; i<size; i++) {
            res = cudaStreamCreate(&streams[i]);
            if(res != cudaSuccess) {
                printf("Could'nt Create Stream %d\n", (i+1));
            }
            
            threadPoolArgs *args = new threadPoolArgs;
            args->id = i;
            args->pool = this;
            rc = pthread_create(&threads[i], &attr, StreamThreadPool::threadWork, (void*) args);
            if(rc) {
                printf("Couldn't Spawn ThreadPool Thread: %d\n", (i+1));
            }
        }
        pthread_attr_destroy(&attr);
    }
    
    int addToQueue(PACKET_T* buffer) {
        return workQueue->add(buffer);
    }
    
    void stopPool() {
        running = false;
        printf("\r\nStopping Threads\r\n");
        cudaDeviceSynchronize();
        printf("\r\nStopped\r\n");
    }
    
    static void* threadWork(void* args) {
        static const int threadsPerBlock = Config::getThreads();
        static const int blocks = Config::getBlocks();
        static const int algo = Config::getAlgo();
        int id;
	    char* dBuffer;
	    int * dResults;
	    float time;
        cudaError_t cudaRes;
        
        cudaEvent_t startEv, stopEv;
        
        u_char *buffer = NULL;
        int packetSize = 0;
        PACKET_T *arg;
        
        StreamThreadPool *pool;
        {
            threadPoolArgs *arg = (threadPoolArgs*) args;
            id = arg->id;
            pool = arg->pool;
            delete arg;
        }
        printf("Worker %d Running\n", (id+1));
        while(pool->running) {
            arg = pool->workQueue->remove();
            if(arg != NULL) {
                buffer = arg->buffer;
                packetSize = arg->size;
                delete arg;
                //printf("Worker %d got: %d\n", (id+1), packetSize);
                cudaEventCreate(&startEv);
                cudaEventCreate(&stopEv);
                
                cudaEventRecord(startEv, pool->streams[id]);
                // Allocate memory for the packet
                if(Config::getZeroCopy()) {
                cudaRes = cudaHostGetDevicePointer((void**) &dBuffer, (void*) buffer, 0);
                    // Memory Allocation for packet successful?
                    if(cudaRes != cudaSuccess)
	                {
		                printf("\nThread Pool cudaGetDevicePointer Error Occurred\n%s\n", cudaGetErrorString(cudaRes));
                        cudaFreeHost(buffer);
                        pthread_exit(NULL);
	                }
	            } else {
	                cudaRes = cudaMalloc((void**) &dBuffer, sizeof(char)*packetSize);
                    // Memory Allocation for packet successful?
                    if(cudaRes != cudaSuccess)
	                {
		                printf("\nThread Pool cudaMalloc for dBuffer Error Occurred\n%s\n", cudaGetErrorString(cudaRes));
                        cudaFreeHost(buffer);
                        pthread_exit(NULL);
	                }
	                cudaRes = cudaMemcpyAsync(dBuffer, buffer, sizeof(char)*packetSize, cudaMemcpyHostToDevice, pool->streams[id]);
                    if(cudaRes != cudaSuccess)
	                {
		                printf("\nThread Pool cudaMemcpy for dBuffer Error Occurred\n%s\n", cudaGetErrorString(cudaRes));
                        cudaFreeHost(buffer);
                        cudaFree(dBuffer);
                        pthread_exit(NULL);
	                }
	            }
	            // Allocate GPU memory for results of pattern search and check results
	            cudaRes = cudaMalloc((void**) &dResults, sizeof(int)*2);
	            // Memory Allocation for packet successful?
	            if(cudaRes != cudaSuccess)
	            {
	                printf("\nThread Pool cudaMalloc for results Error Occurred\n%s\n", cudaGetErrorString(cudaRes));
	                cudaFreeHost(buffer);
	                cudaFree(dBuffer);
	                pthread_exit(NULL);
	            }
	            // Set Results to -1, -1
	            cudaRes = cudaMemsetAsync(dResults, -1, sizeof(int)*2, pool->streams[id]);
	            if(cudaRes != cudaSuccess)
	            {
	                printf("\nProcess Packet cudaMemset Error Occurred\n%s\n", cudaGetErrorString(cudaRes));
	                cudaFreeHost(buffer);
	                cudaFree(dBuffer);
	                pthread_exit(NULL);
	            }
	            switch(algo) {
	                case 1:
	                    process_packet_gpu<<<blocks , threadsPerBlock, packetSize, pool->streams[id]>>>(dBuffer, packetSize, dResults);
	                break;
	                case 2:
	                    process_packet_gpu_rk<<<blocks , threadsPerBlock, packetSize, pool->streams[id]>>>(dBuffer, packetSize, dResults);
	                break;
	                default:
	                    printf("Algo ID: %d unknown\nExiting\n", algo);
	                    cudaFreeHost(buffer);
	                    cudaFree(dBuffer);
	                    pthread_exit(NULL);
	                break;
	            }
	            
	            int *hN;
	            cudaMallocHost(&hN, sizeof(int) * 2);
	            // Retrieve Results
	            cudaRes = cudaMemcpyAsync(hN, dResults, sizeof(int)*2, cudaMemcpyDeviceToHost, pool->streams[id]);
	            if(cudaRes != cudaSuccess)
	            {
	                printf("\nThreadPool Results CopyBack Error Occurred\n%s\n", cudaGetErrorString(cudaRes));
	            }
	            // All threads Synch
	            cudaEventRecord(stopEv, pool->streams[id]);
	            cudaStreamSynchronize(pool->streams[id]);
	            cudaEventElapsedTime(&time, startEv, stopEv);
	            pool->bytes += packetSize;
	            pool->packetTime += time;
	            pool->counter ++;
	            
	            if(hN[0] > -1) {
	                fprintf(logfile, "\nThreat Found: ");
	                PrintTheUnprintable(hPatternText[hN[1]], hPatternLen[hN[1]], logfile);
	                fprintf(logfile, " -> %d\n", hN[0]);
	                got_packet(buffer);
	            }
	            
	            // Release memories
	            cudaFreeHost(buffer);
	            cudaFreeHost(hN);
	            cudaFree(dResults);
	            if(!Config::getZeroCopy()) {
	                cudaFree(dBuffer);
	            }
	            cudaEventDestroy(startEv);
	            cudaEventDestroy(stopEv);
	            //printf("%d consumed\r\n", (id+1));
            } else {
                //printf("Worker %d got NULL\r\n", (id+1));
                sleep(1);
            }
        }
        
        printf("Worker %d Exiting\n", (id+1));
        pthread_exit(NULL);
    }
    
    ~StreamThreadPool() {
        printf("Destroying Thread Pool\r\n");
        if(running) {
            stopPool();
        }
        cudaDeviceSynchronize();
        while(!workQueue->isEmpty()) {
            PACKET_T* temp = workQueue->remove();
            cudaFreeHost(temp->buffer);
            delete temp;
        }

        delete workQueue;
        
        for(int i=0; i<this->size; i++) {
            cudaStreamDestroy(streams[i]);
        }
        
        delete[] this->streams;
        delete[] this->threads;
        printf("Destroyed Thread Pool\r\n");
        
        // Time reference from
        // www.stackoverflow.com/questions/5141960/get-the-current-time-in-c
        time_t mytime;
        mytime = time(NULL);
        fprintf(logfile, "\r\n\r\n*** Program stopped on : %s\r\n", ctime(&mytime));
        if(logfile != stdout) {
            fclose(logfile);
            logfile = NULL;
        }
    }
};

void PrintTheUnprintable (const char * data , int Size, FILE* file)
{
    int i;
    for(i=0 ; i < Size ; i++)
    {
        if(data[i]>=32 && data[i]<=128)
            fprintf(file , "%c",(unsigned char)data[i]); //if its a number or alphabet
        else
            fprintf(file , "."); //otherwise print a dot
    }
}

#endif
