# FFT demo — 두 백엔드로 빌드 가능
#   make            커스텀 myfft (외부 의존성 없음) → ./fft_demo
#   make run        커스텀 빌드 후 실행
#   make fftw       실제 libfftw3f 링크        → ./fft_demo_fftw   (FFTW 설치 필요)
#   make run-fftw   FFTW 빌드 후 실행
#   make compare    두 백엔드 실행 결과를 diff (정확성 대조)
#   make clean

CC      ?= gcc
CFLAGS  ?= -std=gnu11 -O2 -Wall -I.
LDLIBS_COMMON = -lm

TARGET_CUSTOM = fft_demo
TARGET_FFTW   = fft_demo_fftw

RADAR_SRCS = $(wildcard Globals/*.c) $(wildcard FFT/*.c) $(wildcard FFT_RADIX4/*.c)

.PHONY: all run fftw run-fftw compare clean

all: $(TARGET_CUSTOM)

# --- 커스텀 백엔드 (기본, 무의존) ---
$(TARGET_CUSTOM): demo.c myfft.c myfft.h fft_backend.h $(RADAR_SRCS)
	$(CC) $(CFLAGS) demo.c myfft.c $(RADAR_SRCS) -o $(TARGET_CUSTOM) $(LDLIBS_COMMON)

run: $(TARGET_CUSTOM)
	./$(TARGET_CUSTOM)

# --- 실 FFTW 백엔드 (골든 레퍼런스) ---
# demo 가 fftwf_init_threads 등을 호출하므로 -lfftw3f_threads -pthread 필요
# (레이더 Makefile 의 링크 구성과 동일). 순서 주의: threads 를 먼저.
LDLIBS_FFTW = -lfftw3f_threads -lfftw3f -pthread $(LDLIBS_COMMON)
$(TARGET_FFTW): demo.c fft_backend.h
	$(CC) $(CFLAGS) -DUSE_FFTW demo.c -o $(TARGET_FFTW) $(LDLIBS_FFTW)

fftw: $(TARGET_FFTW)

run-fftw: $(TARGET_FFTW)
	./$(TARGET_FFTW)

# --- 두 백엔드 출력 대조 ---
compare: $(TARGET_CUSTOM) $(TARGET_FFTW)
	@./$(TARGET_CUSTOM) > out_custom.txt; \
	 ./$(TARGET_FFTW)   > out_fftw.txt; \
	 sed '1d' out_custom.txt > c.txt; sed '1d' out_fftw.txt > f.txt; \
	 if diff -q c.txt f.txt >/dev/null; then echo "OK: 커스텀 == FFTW (헤더 줄 제외 동일)"; \
	 else echo "DIFF 발견:"; diff c.txt f.txt; fi; \
	 rm -f c.txt f.txt

clean:
	rm -f $(TARGET_CUSTOM) $(TARGET_FFTW) $(TARGET_CUSTOM).exe $(TARGET_FFTW).exe \
	      out_custom.txt out_fftw.txt
