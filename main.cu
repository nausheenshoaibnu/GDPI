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

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h> // for exit()
#include <string.h> //for memset
#include <signal.h>
#include <math.h>
#include <time.h>
#include <algorithm>
#include <unistd.h>

#include <pthread.h>
#include <pcap.h>

#include "StreamThreadPool.cuh"
#include "config.cuh"

using namespace std;

//#define DEBUG 1

/*********************************************************/

/*********************************************************/
extern int nPatterns;
long maxP = 0;
int counter = 0;

struct timeval startTime, stopTime;
long bytes = 0;
float packetTime = 0;
/*********************************************************/

void loadConfig(const char* file);

/*
	Prints pkt data
*/
void PrintData (const u_char * , int);


/*
	This function is called by libpcap to process packets.
	It will call GPU kernel and do the rest there.
*/
void process_packet(u_char *, const struct pcap_pkthdr *, const u_char *);

/*
	Initialize variables, both host and gpu.
	Load patterns, prepare tables and copy to gpu.
*/
void setUpFirewall();

pcap_t *handle = NULL; //Handle of the device that shall be sniffed
FILE* logfile;
StreamThreadPool *pool = NULL;

/*
	Interrupt handler.
	Only way to exit the program is via CTRL+C
	So this listens for it, and performs proper exit
	Deallocates memory
*/
void intHandlerToFreeMemory(int dummy) {
	if(handle != NULL) {
		pcap_breakloop(handle);
		pcap_close(handle);
	}
	if(pool != NULL) {
	    pool->stopPool();
	} else {
	    switch(Config::getAlgo())
	    {
	        case 1:
                free_memory();
	            break;
	        case 2:
                free_memory_rk();
	            break;
	        default:
	            break;
	    }
	    exit(0);
	}
	counter = pool->counter;
	bytes = pool->bytes;
	packetTime = pool->packetTime;
	gettimeofday(&stopTime, NULL);
	long t = ((stopTime.tv_sec - startTime.tv_sec)*1000000L + stopTime.tv_usec) - startTime.tv_usec;
	float seconds = t / 1000000.0f;
	fprintf(logfile, "\r\nPackets: %d\n", counter);
	fprintf(logfile, "Time: %f\n", seconds);
	fprintf(logfile, "Packets Per Second: %f\n", (counter/seconds));
	fprintf(logfile, "MB Per Second: %f\n", ((bytes/1048576.0f)/seconds));
	fprintf(logfile, "Avg Packet Process Time: %f\n", packetTime/(counter*1000));
	
	printf("\nFreeing Memory\n");
	cudaDeviceSynchronize();
	delete pool;
	sleep(1);
	switch(Config::getAlgo())
	{
	    case 1:
            free_memory();
	    break;
	    case 2:
            free_memory_rk();
	    break;
	    default:
	    break;
	}
	if(logfile != stdout) {
	    fclose(logfile);
	}
	printf("\nFreed Memory\n");
	//cudaProfilerStop();
	exit(0);
}

int main(int argc, char** argv)
{
    loadConfig("config.cfg");
    int i, len;
    for(i=1; i < argc; i++) {
        if(strncmp(argv[i], "--streams=", 10) == 0) {
            int s = atoi(argv[i]+10);
            if(s > 0)
                Config::setStreams(s);
            else {
                printf("Invalid Streams ( >= 1 ): \"%d\"\n", s);
                exit(-1);
            }
        } else if(strncmp(argv[i], "--threads=", 10) == 0) {
            int t = atoi(argv[i] + 10);
            if( t > 0 && t <= 1024 )
                Config::setThreads(t);
            else {
                printf("Thread count (1-1024) of of bounds: \"%d\"\n", t);
                exit(-1);
            }
        } else if(strncmp(argv[i], "--blocks=", 9) == 0) {
            int b = atoi(argv[i] + 9);
            if( b > 0 )
                Config::setBlocks(b);
            else {
                printf("Block count (1-8) of of bounds \"%d\"\n", b);
                exit(-1);
            }
        } else if(strncmp(argv[i], "--logfile=", 10) == 0) {
            char* temp = argv[i] + 11;
            len = strlen(temp);
            temp[--len] = '\0';
            Config::setLogFile(temp);
        } else if(strncmp(argv[i], "--patterns=", 11) == 0) {
            char* temp = argv[i] + 12;
            len = strlen(temp);
            temp[--len] = '\0';
            Config::setPatterns(temp);
        } else if(strncmp(argv[i], "--zerocopy=", 11) == 0) {
            char z = argv[i][11];
            if(z == 'y' || z == 'n') {
                Config::setZeroCopy(z);
            } else {
                printf("ZeroCopy=y/n -> \'%c\' is invlaid\n", z);
                exit(-1);
            }
        } else if(strncmp(argv[i], "--algo=", 7) == 0) {
            /*
            char* temp = argv[i] + 7;
            len = strlen(temp);
            temp[--len] = '\0';
            Config::setAlgo(temp);
            */
            Config::setAlgo(argv[i]);
        }
    }
	setUpFirewall();
	
    #ifdef DEBUG
    printf("Setup Firewall finished\r\n");
    #endif
    {
    	pcap_if_t *alldevsp , *device;
    	
    	char errbuf[PCAP_ERRBUF_SIZE] , *devname , devs[100][100];
    	int count = 1 , n;
     
    	//First get the list of available devices
    	printf("Finding available devices ... ");
    	if( pcap_findalldevs( &alldevsp , errbuf) )
		{
		    printf("Error finding devices : %s" , errbuf);
		    exit(1);
		}
		printf("Done");
		 
		//Print the available devices
		printf("\nAvailable Devices are :\n");
		for(device = alldevsp ; device != NULL ; device = device->next)
		{
		    printf("%d. %s - %s\n" , count , device->name , device->description);
		    if(device->name != NULL)
		    {
		        strcpy(devs[count] , device->name);
		    }
		    count++;
		}
		 
		//Ask user which device to sniff
		printf("Enter the number of the device you want to sniff : ");
		scanf("%d" , &n);
		devname = devs[n];
		 
		//Open the device for sniffing
		printf("Opening device %s for sniffing ... " , devname);
		handle = pcap_open_live(devname , 65536 , 1 , /*Packet Read Timeout*/ 0, errbuf);
		 
		if (handle == NULL) 
		{
		    fprintf(stderr, "Couldn't open device %s : %s\n" , devname , errbuf);
		    intHandlerToFreeMemory(-1);
		    return -1;
		}
		printf("Done\n");
		
		pcap_freealldevs(alldevsp);
    }
    
    gettimeofday(&startTime, NULL);
    
    #ifdef DEBUG
    printf("Creating Thread Pool\r\n");
    #endif
    pool = new StreamThreadPool(Config::getStreams());
    pool->setLogFile(Config::getLogFile());
    logfile = fopen(Config::getLogFile(), "a");
    pool->startPool();
    #ifdef DEBUG
    printf("Thread Pool Created\r\n");
    #endif
    
    #ifdef DEBUG
    printf("Invoking PCAP LOOP\r\n");
    #endif
    //cudaProfilerStart();
    //Put the device in sniff loop
    pcap_loop(handle , -1 , process_packet , NULL);
    
    #ifdef DEBUG
    printf("PCAP LOOP Ended\r\n");
    #endif
    
    pthread_exit(NULL);
}

inline void nCopy(char* dest, char* source, int size) {
    int i;
    for(i=0; i<size; i++) {
        dest[i] = source[i];
    }
}

void process_packet(u_char *args, const struct pcap_pkthdr *header, const u_char *buffer)
{
    int size = header->len;
    PACKET_T *pkt = new PACKET_T;
    pkt->size = size;
    cudaHostAlloc(&(pkt->buffer), (sizeof(char) * size), cudaHostAllocMapped);
    nCopy((char*) pkt->buffer, (char*) buffer, size);
    pool->addToQueue(pkt);
}

/*
	Prints the packet data like:
	HEXADECIMAL		ASCII
	such that non printable characters are replaced with '.'
*/

void PrintData (const u_char * data , int Size)
{
    int i , j;
    for(i=0 ; i < Size ; i++)
    {
        if( i!=0 && i%16==0)   //if one line of hex printing is complete...
        {
            fprintf(logfile , "         ");
            for(j=i-16 ; j<i ; j++)
            {
                if(data[j]>=32 && data[j]<=128)
                    fprintf(logfile , "%c",(unsigned char)data[j]); //if its a number or alphabet
                 
                else fprintf(logfile , "."); //otherwise print a dot
            }
            fprintf(logfile , "\n");
        } 
         
        if(i%16==0) fprintf(logfile , "   ");
            fprintf(logfile , " %02X",(unsigned int)data[i]);
                 
        if( i==Size-1)  //print the last spaces
        {
            for(j=0;j<15-i%16;j++) 
            {
              fprintf(logfile , "   "); //extra spaces
            }
             
            fprintf(logfile , "         ");
             
            for(j=i-i%16 ; j<=i ; j++)
            {
                if(data[j]>=32 && data[j]<=128) 
                {
                  fprintf(logfile , "%c",(unsigned char)data[j]);
                }
                else
                {
                  fprintf(logfile , ".");
                }
            }
             
            fprintf(logfile ,  "\n" );
        }
    }
}

void loadConfig(const char* file)
{
    ifstream fin;
    fin.open(file);
    
    if(!fin.is_open())
    {
        printf("Couldn't open configuration file: %s\n", file);
        return;
    }
    printf("Parsing Config File\n");
    string str = "";
    while(getline(fin, str))
    {
        if(strncmp(str.c_str(), "streams=", 8) == 0) {
            int s = atoi(str.c_str()+8);
            if(s > 0)
                Config::setStreams(s);
            else {
                printf("Config: Invalid Streams ( >= 1 ): \"%d\"\n", s);
            }
        } else if(strncmp(str.c_str(), "threads=", 8) == 0) {
            int t = atoi(str.c_str() + 8);
            if( t > 0 && t <= 1024 )
                Config::setThreads(t);
            else {
                printf("Config: Thread count (1-1024) out of bounds: \"%d\"\n", t);
            }
        } else if(strncmp(str.c_str(), "blocks=", 7) == 0) {
            int b = atoi(str.c_str() + 7);
            if( b > 0 )
                Config::setBlocks(b);
            else {
                printf("Config: Block count (1-8) of of bounds \"%d\"\n", b);
            }
        } else if(strncmp(str.c_str(), "logfile=", 8) == 0) {
            char temp2[2048];
            const char* temp = str.c_str() + 9;
            int len = strlen(temp);
            strncpy(temp2, temp, len);
            temp2[--len] = '\0';
            Config::setLogFile(temp2);
        } else if(strncmp(str.c_str(), "patterns=", 9) == 0) {
            char temp2[2048];
            const char* temp = str.c_str() + 10;
            int len = strlen(temp);
            strncpy(temp2, temp, len);
            temp2[--len] = '\0';
            Config::setPatterns(temp2);
        } else if(strncmp(str.c_str(), "zerocopy=", 9) == 0) {
            char z = str.c_str()[9];
            if(z == 'y' || z == 'n') {
                Config::setZeroCopy(z);
            } else {
                printf("ZeroCopy=y/n -> \'%c\' is invlaid\n", z);
            }
        } else if(strncmp(str.c_str(), "algo=", 5) == 0) {
        /*
            char temp2[100];
            const char* temp = str.c_str() + 5;
            int len = strlen(temp);
            strncpy(temp2, temp, len);
            temp2[--len] = '\0';
            */
            Config::setAlgo(str);
        }
    }
    fin.close();
    printf("Config file parsed successfully\n");
}

string hexToString(const string &inp) {
	char acc = 0;
	string out = "";
	int j;
	for(j=0; j<inp.length()-1; j+=2) {
	    acc = 0;
		acc += ((inp[j] >= 'a' ? (inp[j]-'a') + 10 : inp[j]-'0')*16);
		acc += ((inp[j+1] >= 'a' ? (inp[j+1]-'a') + 10 : inp[j+1]-'0'));
		
		out += (char)acc;
	}
	return out;
}

void setUpFirewall()
{
    struct sigaction sa;
    sa.sa_handler = intHandlerToFreeMemory;
    sa.sa_flags = 0;
    sigemptyset(&sa.sa_mask);
    if(sigaction(SIGINT, &sa, NULL) == -1) {
        perror("SIGACTION");
        exit(-1);
    }
    {
        printf("Algorithm: %s\n", (Config::getAlgo() == 1 ? "KMP" : "Rabin Karp"));
        printf("Streams: %d\n", Config::getStreams());
        printf("Threads: %d\n", Config::getThreads());
        printf("Blocks: %d\n", Config::getBlocks());
        printf("Patterns: %s\n", Config::getPatterns());
        printf("LogFile: %s\n", Config::getLogFile());
        printf("ZeroCopy: %c\n", (Config::getZeroCopy() ? 'y' : 'n'));
        FILE* t = fopen(Config::getLogFile(), "w");
        if( t == NULL ) {
            printf("LogFile set to stdout\n");
            logfile = stdout;
        } else {
            logfile = t;
        }
        if(Config::getZeroCopy()) {
            cudaSetDeviceFlags(cudaDeviceMapHost);
        }
    }
    printf("Loading Patterns\n");
    ifstream f2;
    f2.open(Config::getPatterns());
    if(!f2.is_open())
    {
        printf("Couldn't open patterns file %s\n", Config::getPatterns());
        exit(-1);
    }
    vector<string> pattern;
    
    int size = 0;
    string str="";
	// Get the patterns
    while (getline(f2, str))
	{
	    int i = 0;
	    while(true)
	    {
	        i = str.find("\n", i);
	        if (i == std::string::npos) break;
	        str.replace(i, 1, "");
	    }
	    i = 0;
	    while(true)
	    {
	        i = str.find("\r", i);
	        if (i == std::string::npos) break;
	        str.replace(i, 1, "");
	    }
		str = hexToString(str.substr(0, str.length()));
		pattern.push_back(str);
		if(str.length() > size) {
		    size = str.length();
		}
	}
	std::sort(pattern.begin(), pattern.end(), std::greater<std::string>());
	size = size * pattern.size();
	int lp = 1;
	switch(Config::getAlgo())
	{
		case 1:
			lp = load_patterns_to_gpu(pattern, size);
			break;
		case 2:
			lp = load_patterns_to_gpu_rk(pattern, size);
			break;
		default:
			break;
	}
	
	if(lp != 1)
	{
		switch(lp)
		{
			case -1:
				cout << "Couldn't Allocate GPU vars :(" << endl;
				break;
			case -2:
				cout << "Couldn't Copy Patterns to GPU vars :(" << endl;
				break;
			case -3:
				cout << "Memory Allocation exception :(" << endl;
				break;
		}
		switch(Config::getAlgo())
		{
			case 1:
			    free_memory();
				break;
			case 2:
			    free_memory_rk();
				break;
			default:
				break;
		}
		exit(-1);
	}
	f2.close();
	cout << nPatterns << " Patterns Loaded!" << endl;
	
}
