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

#ifndef CONFIG_H
#define CONFIG_H

class Config {
    private:
        static int streams,
            threads,
            blocks;
        static char patternsFile[1024], logFile[1024];
        static char zeroCopy;
        static int algo;
    public:
        static int getStreams() {
            return streams;
        }
        static int setStreams(int s) {
            return streams = s;
        }
        static int getAlgo() {
            return algo;
        }
        static int setAlgo(string a) {
            if(a.find("kmp") != std::string::npos) {
                algo = 1;
            } else if(a.find("rk") != std::string::npos) {
                algo = 2;
            } else {
                algo = 1;
            }
            return algo;
        }
        static int getThreads() {
            return threads;
        }
        static int setThreads(int t) {
            return threads = t;
        }
        static int getBlocks() {
            return blocks;
        }
        static int setBlocks(int b) {
            return blocks = b;
        }
        static char* getPatterns() {
            return patternsFile;
        }
        static char* setPatterns(char p[]) {
            int i = 0;
            while(p[i] != '\0') {
                patternsFile[i] = p[i];
                i++;
            }
            patternsFile[i] = '\0';
            return patternsFile;
        }
        static char* getLogFile() {
            return logFile;
        }
        static char* setLogFile(char p[]) {
            int i = 0;
            while(p[i] != '\0') {
                logFile[i] = p[i];
                i++;
            }
            logFile[i] = '\0';
            return logFile;
        }
        
        static char getZeroCopy() {
            return zeroCopy;
        }
        static char setZeroCopy(char z) {
            zeroCopy = (z == 'y');
            return zeroCopy;
        }
};
int Config::algo = 1;
int Config::streams = 4;
int Config::threads = 512;
int Config::blocks = 4;
char Config::patternsFile[] = "/home/ubuntu/Desktop/Patterns/pattern.txt";
char Config::logFile[] = "";
char Config::zeroCopy = 'y';
#endif
