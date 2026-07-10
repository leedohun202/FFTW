#ifndef RADAR_PROFILER_H
#define RADAR_PROFILER_H

#include <stddef.h>

// ---------------------------------------------------------
// [메모리 및 시간 프로파일링 헬퍼 함수]
// ---------------------------------------------------------

// 1. 메모리 측정 시작 (캐시 비우기 및 현재 RAM 측정)
int profiler_start_mem(void);

// 2. 메모리 측정 종료 (증가한 RAM 사용량을 MB 단위로 반환)
double profiler_end_mem_mb(int base_mem_kb);

// 3. CPU 캐시(L2/L3) 강제 플러시 (Cold Cache 상태 유도)
void profiler_flush_cache(void);

// 4. 정렬된 메모리 할당 및 0 초기화 통합 헬퍼 (찌꺼기 오염 방지)
void* allocate_and_clear_aligned(size_t size, size_t alignment);

#endif // RADAR_PROFILER_H