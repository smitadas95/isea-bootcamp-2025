#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdint.h>
#include <x86intrin.h>
#include <unistd.h>

#define MATRIX_SIZE 64   
#define NOP_COUNT 100000 

volatile int ready = 0;      
volatile int bit_to_send = 0; 

static inline uint64_t rdtsc() {
    return __rdtsc();
}


void* sender(void* arg) {
    
    while (ready == 0);

    if (bit_to_send == 1) {
        int A[MATRIX_SIZE][MATRIX_SIZE], B[MATRIX_SIZE][MATRIX_SIZE], C[MATRIX_SIZE][MATRIX_SIZE] = {0};

        for (int x = 0; x < MATRIX_SIZE; x++)
            for (int y = 0; y < MATRIX_SIZE; y++)
                for (int z = 0; z < MATRIX_SIZE; z++)
                    C[x][y] += A[x][z] * B[z][y];

    } else {
        for (int j = 0; j < NOP_COUNT; j++) {
            __asm__ __volatile__("nop");
        }
    }
    ready = 0;
    return NULL;
}

void* receiver(void* arg) {
    uint64_t start, end;

    ready = 1;
    start = rdtsc();   
    while (ready == 1);
    end = rdtsc();
    
    uint64_t cycles_taken = end - start;
    
    /*
     * Your code here. Infer which bit has been transmitted
    */

    //printf("Sent Bit: %d | Received Bit: %d (Cycles: %llu)\n", bit_to_send, received_bit, cycles_taken);

    return NULL;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <bit>\n", argv[0]);
        return 1;
    }

    bit_to_send = atoi(argv[1]);
    if (bit_to_send != 0 && bit_to_send != 1) {
        printf("Error: Bit must be 0 or 1.\n");
        return 1;
    }

    pthread_t sender_thread, receiver_thread;

    pthread_create(&sender_thread, NULL, sender, NULL);
    pthread_create(&receiver_thread, NULL, receiver, NULL);

    pthread_join(sender_thread, NULL);
    pthread_join(receiver_thread, NULL);

    return 0;
}
