# FFT demo 벤치마크 Makefile
# -------------------------------------------------------------------
#  make r2         : Radix-2 커스텀 백엔드 빌드 (custom_fft_all_radix2.c 사용)
#  make r4         : Radix-4 커스텀 백엔드 빌드 (custom_fft_all_radix4.c 사용)
#  make fftw       : 실제 libfftw3f 링크 빌드 (골든 레퍼런스)
#
#  make run-r2     : Radix-2 백엔드 빌드 후 실행
#  make run-r4     : Radix-4 백엔드 빌드 후 실행
#  make run-fftw   : FFTW 백엔드 빌드 후 실행
#
#  make compare    : Radix-2와 FFTW 실행 결과를 diff (정확성 대조)
#  make clean      : 빌드 산출물 삭제
# -------------------------------------------------------------------

CC      ?= gcc
CFLAGS  ?= -std=gnu11 -O2 -Wall -I.
LDLIBS_COMMON = -lm

TARGET_R2   = fft_demo_r2
TARGET_R4   = fft_demo_r4
TARGET_FFTW = fft_demo_fftw

RADAR_SRCS = $(wildcard Globals/*.c)

.PHONY: all r2 r4 fftw run-r2 run-r4 run-fftw compare2 compare4 clean

all: r2 r4

# --- [1] Radix-2 커스텀 백엔드 ---
r2: $(TARGET_R2)
$(TARGET_R2): demo.c myfft_radix2.c custom_fft_all_radix2.c myfft.h fft_backend.h $(RADAR_SRCS)
	$(CC) $(CFLAGS) demo.c myfft_radix2.c custom_fft_all_radix2.c $(RADAR_SRCS) -o $(TARGET_R2) $(LDLIBS_COMMON)

run-r2: $(TARGET_R2)
	./$(TARGET_R2)

# --- [2] Radix-4 커스텀 백엔드 ---
r4: $(TARGET_R4)
$(TARGET_R4): demo.c myfft_radix4.c custom_fft_all_radix4.c myfft.h fft_backend.h $(RADAR_SRCS)
	$(CC) $(CFLAGS) demo.c myfft_radix4.c custom_fft_all_radix4.c $(RADAR_SRCS) -o $(TARGET_R4) $(LDLIBS_COMMON)

run-r4: $(TARGET_R4)
	./$(TARGET_R4)

# --- [3] 실 FFTW 백엔드 (골든 레퍼런스) ---
# demo가 fftwf_init_threads 등을 호출하므로 -lfftw3f_threads -pthread 필요
LDLIBS_FFTW = -lfftw3f_threads -lfftw3f -pthread $(LDLIBS_COMMON)
fftw: $(TARGET_FFTW)
$(TARGET_FFTW): demo.c fft_backend.h
	$(CC) $(CFLAGS) -DUSE_FFTW demo.c -o $(TARGET_FFTW) $(LDLIBS_FFTW)

run-fftw: $(TARGET_FFTW)
	./$(TARGET_FFTW)

# --- [4] Radix-2 vs FFTW 출력 대조 ---
compare2: $(TARGET_R2) $(TARGET_FFTW)
	@echo "=== Radix-2 vs FFTW 결과 대조 시작 ==="
	@./$(TARGET_R2) > out_r2.txt || true
	@./$(TARGET_FFTW) > out_fftw.txt || true
	@sed '1d' out_r2.txt > c2.txt
	@sed '1d' out_fftw.txt > f.txt
	@if diff -q c2.txt f.txt >/dev/null; then \
		echo "[PASS] OK: Radix-2 커스텀 == FFTW (헤더 줄 제외 동일)"; \
	else \
		echo "[FAIL] DIFF 발견:"; \
		diff c2.txt f.txt || true; \
	fi
	@rm -f out_r2.txt out_fftw.txt c2.txt f.txt

# --- [5] Radix-4 vs FFTW 출력 대조 ---
compare4: $(TARGET_R4) $(TARGET_FFTW)
	@echo "=== Radix-4 vs FFTW 결과 대조 시작 ==="
	@./$(TARGET_R4) > out_r4.txt || true
	@./$(TARGET_FFTW) > out_fftw.txt || true
	@sed '1d' out_r4.txt > c4.txt
	@sed '1d' out_fftw.txt > f.txt
	@if diff -q c4.txt f.txt >/dev/null; then \
		echo "[PASS] OK: Radix-4 커스텀 == FFTW (헤더 줄 제외 동일)"; \
	else \
		echo "[FAIL] DIFF 발견 (부동소수점 오차 확인 요망):"; \
		diff c4.txt f.txt || true; \
	fi
	@rm -f out_r4.txt out_fftw.txt c4.txt f.txt

clean:
	rm -f $(TARGET_R2) $(TARGET_R4) $(TARGET_FFTW) *.exe out_*.txt c2.txt c4.txt f.txt

