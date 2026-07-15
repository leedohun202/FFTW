# ====================================================================
# 🛰️ 3D Radar Pipeline: 완전 자율형 개별 유닛 가속 빌드 시스템
# ====================================================================

CC = gcc

# 🔥 [라즈베리파이 5 네이티브 가속 옵션 수용 - A76 명시]
# 💡 [수정] 헤더 탐색 경로에 -I./Main 을 추가하여 testbench 헤더 유연성을 높였습니다.
CFLAGS = -O3 -mcpu=cortex-a76 -mtune=cortex-a76 -ffast-math -ftree-vectorize -fopenmp -Wall \
         -I./Globals -I./FFT -I./FFT_int16 -I./FFT_RADIX4 -I./Pipeline -I./Utils

# 🧵 [OpenMP와 라이브러리 자원 충돌 방지 통합 플래그]
LDFLAGS = -lfftw3f -lfftw3f_threads -lpthread -lm -fopenmp

# ====================================================================
# 📦 공통 오브젝트 자동 탐지 (와일드카드 & 필터링 안전 매핑)
# ====================================================================
ALL_SRCS = $(wildcard Globals/*.c) \
           $(wildcard FFT/*.c) \
           $(wildcard Pipeline/*.c) \
           $(wildcard FFT_RADIX4/*.c) \
           $(wildcard FFT_int16/*.c) \
           $(wildcard FFT_int16_RADIX4/*.c) \
           $(wildcard Utils/*.c)

COMMON_SRCS = $(ALL_SRCS)

COMMON_OBJS = $(COMMON_SRCS:.c=.o)

# ====================================================================
# 🎯 바이너리 그룹 타겟 정의 (.PHONY)
# ====================================================================
# 💡 [수정] .PHONY 목록에 testbench를 추가했습니다.
.PHONY: all fftw3 custom testbench clean

all: fftw3 custom testbench

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
# 🎯 [3] 💥 [신규 추가] 7대 엔진 통합 자율 인지 테스트벤치 사출 규칙
# ====================================================================
testbench: radar_testbench

radar_testbench: Main/main_testbench.c $(COMMON_OBJS)
	@echo "🚀 [Makefile] 7대 가속 엔진 통합 엣지케이스 테스트벤치 컴파일 중..."
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "🟩 빌드 완료! ./radar_testbench 명령어로 실행하세요."

# ====================================================================
# 🧼 일반 컴파일 패턴 규칙 및 청소
# ====================================================================
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(COMMON_OBJS) radar_test_* radar_testbench Utils/run_*.o
	@echo "🧹 [Makefile] 기존 실행 파일 및 오브젝트 가 청소되었습니다."