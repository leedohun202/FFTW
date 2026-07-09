CC = gcc
CFLAGS = -O3 -mcpu=cortex-a76 -mtune=cortex-a76 -ffast-math -ftree-vectorize -fopenmp -Wall \
         -I./Globals -I./FFT  -I./FFT_int16 -I./FFT_RADIX4 -I./Pipeline -I./Utils
LDFLAGS = -lfftw3f -lfftw3f_threads -lpthread -lm

# 🎯 타겟 바이너리 정의
TARGET_EST  = radar_test_est
TARGET_MEAS = radar_test_meas
TARGET_PAT  = radar_test_pat

SRCS = main.c \
       $(wildcard Globals/*.c) \
       $(wildcard FFT/*.c) \
       $(wildcard Pipeline/*.c) \
       $(wildcard Utils/*.c) \
	   $(wildcard FFT_RADIX4/*.c) \
	   $(wildcard FFT_int16/*.c)
OBJS = $(SRCS:.c=.o)

# 1️⃣ 기본 타겟 (ESTIMATE 모드: 0.1초 탈출 디버깅용)
all: CFLAGS += -DFFTW_MODE_ESTIMATE
all: clean_objs $(TARGET_EST)

$(TARGET_EST): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET_EST) $(OBJS) $(LDFLAGS)

# 2️⃣ MEASURE 타겟 (속도 측정 및 밸런스 모드)
measure: CFLAGS += -DFFTW_MODE_MEASURE
measure: clean_objs $(TARGET_MEAS)

$(TARGET_MEAS): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET_MEAS) $(OBJS) $(LDFLAGS)

# 3️⃣ PATIENT 타겟 (극한의 고문 및 최종 릴리즈 모드)
patient: CFLAGS += -DFFTW_MODE_PATIENT
patient: clean_objs $(TARGET_PAT)

$(TARGET_PAT): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET_PAT) $(OBJS) $(LDFLAGS)

# 패턴 규칙 및 빌드 전 안전 세탁
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean_objs:
	rm -f $(OBJS)

clean:
	rm -f $(OBJS) $(TARGET_EST) $(TARGET_MEAS) $(TARGET_PAT)