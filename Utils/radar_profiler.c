#define _GNU_SOURCE

#include "radar_profiler.h"
#include "radar_utils.h"
#include <malloc.h>
#include <stdlib.h>
#include <string.h>

void profiler_flush_cache(void) {
    const int cache_size = 32 * 1024 * 1024; // 32MB 가짜 데이터
    volatile char *dummy = (char*)malloc(cache_size);
    if (dummy) {
        for (int i = 0; i < cache_size; i += 64) { dummy[i] = 1; }
        free((void*)dummy);
    }
}

int profiler_start_mem(void) {
    malloc_trim(0); // 힙 메모리 파편화 정리
    return get_current_ram_usage_kb();
}

double profiler_end_mem_mb(int base_mem_kb) {
    int peak_mem = get_current_ram_usage_kb();
    return (peak_mem - base_mem_kb) / 1024.0;
}

void* allocate_and_clear_aligned(size_t size, size_t alignment) {
    void *ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) == 0) {
        memset(ptr, 0, size);
    }
    return ptr;
}