# ====================================================================
# 🛰️ 3D Radar Pipeline: 완전 자율형 개별 유닛 가속 빌드 시스템
# ====================================================================

CC = gcc

# 🔥 [라즈베리파이 5 네이티브 가속 옵션 수용 - A76 명시]
CFLAGS = -O3 -mcpu=cortex-a76 -mtune=cortex-a76 -ffast-math -ftree-vectorize -fopenmp -Wall \
         -I./Globals -I./FFT -I./FFT_int16 -I./FFT_RADIX4 -I./Pipeline -I./Utils

# 🧵 [OpenMP와 라이브러리 자원 충돌 방지 통합 플래그]
LDFLAGS = -lfftw3f -lfftw3f_threads -lpthread -lm -fopenmp

# ====================================================================
# 📦 공통 오브젝트 자동 탐지 (와일드카드 & 필터링 안전 매핑)
# ====================================================================
# 1. 모든 관련 폴더의 .c 파일을 싹 긁어옵니다.
ALL_SRCS = $(wildcard Globals/*.c) \
           $(wildcard FFT/*.c) \
           $(wildcard Pipeline/*.c) \
           $(wildcard FFT_RADIX4/*.c) \
           $(wildcard FFT_int16/*.c) \
           $(wildcard FFT_int16_RADIX4/*.c) \
           $(wildcard Utils/*.c)

# 2. 💥 [핵심 필터링] 그 중에서 'run_' 이나 'main_' 으로 시작하는 
#    런처(Launcher) 전용 C 파일들은 공통 오브젝트에서 쏙 뺍니다! (충돌 원천 차단)
COMMON_SRCS = $(ALL_SRCS)

COMMON_OBJS = $(COMMON_SRCS:.c=.o)

# ====================================================================
# 🎯 바이너리 그룹 타겟 정의 (.PHONY)
# ====================================================================
.PHONY: all fftw3 custom clean

all: fftw3 custom

fftw3: radar_test_fftw3_est radar_test_fftw3_meas radar_test_fftw3_pat
custom: radar_test_float_r2 radar_test_float_r4 radar_test_int16_r2 radar_test_int16_r4

# ====================================================================
# 🎯 [1] FFTW3 진영: 3가지 플래닝 모드별 개별 사출 규칙
# ====================================================================
radar_test_fftw3_est: Main/main_fftw3_estimate.c $(COMMON_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

radar_test_fftw3_meas: Main/main_fftw3_measure.c $(COMMON_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

radar_test_fftw3_pat: Main/main_fftw3_patient.c $(COMMON_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# ====================================================================
# 🎯 [2] Custom 가속기 진영: 독자적인 단일 Standalone 사출 규칙
# ====================================================================
radar_test_float_r2: Main/main_float_r2.c $(COMMON_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

radar_test_float_r4: Main/main_float_r4.c $(COMMON_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

radar_test_int16_r2: Main/main_int16_r2.c $(COMMON_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

radar_test_int16_r4: Main/main_int16_r4.c $(COMMON_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# ====================================================================
# 🧼 일반 컴파일 패턴 규칙 및 청소
# ====================================================================
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(COMMON_OBJS) radar_test_* Utils/run_*.o