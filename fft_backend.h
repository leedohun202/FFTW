/*
 * fft_backend.h — FFT 백엔드 선택 스위치
 * -----------------------------------------------------------------------------
 * demo.c 는 FFTW-API(fftwf_*)만 사용하므로, 이 헤더 하나로 백엔드를 바꾼다.
 *   - 기본(커스텀)   : #include "myfft.h"   → 외부 의존성 없음
 *   - USE_FFTW 정의 시: #include <fftw3.h>  → 실제 libfftw3f (골든 레퍼런스)
 *
 * 빌드:
 *   make          # 커스텀(myfft) — 무의존, 바로 실행
 *   make fftw     # 실 FFTW  (gcc ... -DUSE_FFTW -lfftw3f) — FFTW 설치 필요
 * -----------------------------------------------------------------------------
 */
#ifndef FFT_BACKEND_H
#define FFT_BACKEND_H

#ifdef USE_FFTW
  #include <fftw3.h>
#else
  #include "myfft.h"
#endif

#endif /* FFT_BACKEND_H */
