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
#ifndef LIST_CUH
#define LIST_CUH

#include <pthread.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef null
    #define null 0
#endif

#ifndef NULL 
    #define NULL 0 
#endif

template<class T>
class list_node {
public:
    T data;
    list_node *next;
    
    list_node() {
        data = NULL;
        next = NULL;
    }
    list_node(T buf) {
        data = buf;
        next = NULL;
    }
    list_node(T buf, list_node *n) {
        data = buf;
        next = n;
    }
    
};

template<class T>
class Queue {

  private:
    pthread_mutex_t lock;
    //pthread_cond_t condition;
    
    list_node<T> *head, *tail;
    int size;
  
  public:

    Queue() {
        size = 0;
        head = NULL;
        tail = NULL;
        pthread_mutex_init(&lock, NULL);
        //pthread_cond_init(&condition, NULL);
    }
    
    int add(T data) {
        pthread_mutex_lock(&lock);
        if(head != NULL) {
            tail = tail->next = new list_node<T>(data);
        } else {
            head = tail = new list_node<T>(data);
        }
        size++;
        //pthread_cond_signal(&condition);
        pthread_mutex_unlock(&lock);
        
        return 1;
    }
    
    T peek() {
        T toRet = NULL;
        pthread_mutex_lock(&lock);
        /*
        while(head == NULL) {
            pthread_cond_wait(&condition, &lock);
        }
        */
        if(head == NULL) {
            toRet = NULL;
        } else {
            toRet = head->data;
        }
        pthread_mutex_unlock(&lock);
        return toRet;
    }
    
    T remove() {
        T toRet = NULL;
        pthread_mutex_lock(&lock);
        /*
        while(head == NULL) {
            pthread_cond_wait(&condition, &lock);
        }
        */
        if(head == NULL) {
            toRet = NULL;
        } else {
            toRet = head->data;
            list_node<T>* temp = head;
            head = head->next;
            delete temp;
            size--;
        }
        if(head == NULL) {
            tail = NULL;
        }
        pthread_mutex_unlock(&lock);
        return toRet;
    }
    
    int isEmpty() {
        return size <= 0;
    }
    
    
    ~Queue() {
        pthread_mutex_lock(&lock);
        list_node<T>* temp = head;
        while(temp != null) {
            temp = temp->next;
            delete head;
            head = temp;
        }
        tail = NULL;
        size = 0;
        //pthread_cond_broadcast(&condition);
        pthread_mutex_unlock(&lock);
        pthread_mutex_destroy(&lock);
        //pthread_cond_destroy(&condition);
    }
};

#endif
