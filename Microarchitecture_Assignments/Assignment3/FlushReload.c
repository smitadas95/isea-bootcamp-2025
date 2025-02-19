#include <emmintrin.h>
#include <x86intrin.h>
#include <stdint.h>
#include <stdio.h>

#define CACHE_HIT_THRESHOLD (90) //this can be chosen by executing the cache_timing_experiment code and selecting any minimum cache access time,
#define DELTA 64

uint8_t array[256 * 4096];
int temp;

void flushCache()
{

  /* Bring the data to RAM. Prevent Copy-on-write.
   COW is a memory management technique that allows multiple processes to share the same memory pages for an array. 
   We will initialize the array so we 'force' the creation  // of a page. We want to ensure that the array exists in the physical memory.
   Purpose: Clears the cache by writing data and flush cache lines to ensure accesses come from main memory. Write your code to do this operations. */

}

void victim(uint8_t secret)
{

  temp = array[secret * 4096 + DELTA];
}
void reloadCache()
{

  int junk = 0;
  register uint64_t time1, time2;
  volatile uint8_t *addr;
  int i;
  for (i = 0; i < 256; i++)
  {

    /* First create a pointer to each potential secret value and check for its memory access time. 
    If the memory access time is less than euqal to the CACHE_HIT_THRESHOLD, then it is cached 
    and print the secret and address. Write your code here. */
  }
}
int main(int argc, const char **argv)
{

  uint8_t secret = 23;
  flushCache();
  victim(secret);
  reloadCache();
  return 0;
}
