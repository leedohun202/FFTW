## 코드 특징

FFT/ : real과 imaginary 2개의 float array를 input으로 받는 radix-2 FFT 코드.
FFT_RADIX4/ : real과 imaginary 2개의 float array를 input으로 받는 radix-4 FFT 코드.
Globals/ : FFT에 사용될 global parameter를 정의하고 초기화. 각 샘플 크기에 따른 FFT 코드에 사용될 bit reversal과 삼각함수 LUT를 맨 처음에 한꺼번에 만든다.

float[2]인 input fftwf_complex를 바로 사용하지는 못하고, FFT/와 FFT_RADIX4/의 코드 특성상 fftwf_complex를 real과 imag로 변환하고 custom fft에 넣고 돌렸다가 다시 합치게 된다.

속도는 fftw에 비해 느려졌지만 일단 기존 코드와 호환이 되며, 정확도는 그대로이다.

# FFT 데모 — FFTW 대체 커스텀 FFT (빌드·실행 가능)

레이더 알고리즘은 전부 걷어내고, 파이프라인이 실제로 쓰는 **FFT 사용 패턴만** 뽑아낸
최소 실행 프로그램입니다. 간단한 입력을 생성 → FFT 실행 → 결과 출력 + **자체 정답 검증**.

동료가 커스텀 FFT를 붙여 실 FFTW 결과와 대조할 수 있는 **테스트 벤치**입니다.

---

## 빠른 시작

```bash
make run       # 커스텀 myfft 로 빌드+실행 (외부 의존성 없음)
```

기대 출력(검증 완료):

```
=== FFT demo  [backend: myfft (custom radix-2)] ===

[1] c2c forward 1D  (N=8, angle-FFT 패턴)
    |X[k]| =        0.00  0.00  8.00  0.00  0.00  0.00  0.00  0.00
    [PASS] peak at expected bin (2)

[2] c2c backward + 수동 /N  (N=8, iFFT 패턴 — 비정규화 규약)
    복원 최대오차 = 1.19e-07
    [PASS] IFFT(FFT(x))/N == x

[3] r2c forward  (N=16 → 9 bins, range-FFT 패턴 — 반스펙트럼)
    |R[k]| =        0.00  0.00  0.00  8.00  0.00  0.00  0.00  0.00  0.00
    [PASS] peak at expected bin (3)

[4] 배치 strided c2c + execute_dft  (N=8, howmany=4, doppler 패턴)
    batch 0 |X|=    8.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
    batch 1 |X|=    0.00  8.00  0.00  0.00  0.00  0.00  0.00  0.00
    batch 2 |X|=    0.00  0.00  8.00  0.00  0.00  0.00  0.00  0.00
    batch 3 |X|=    0.00  0.00  0.00  8.00  0.00  0.00  0.00  0.00
    [PASS] each batch peaks at its own bin (0..3)
    [PASS] execute_dft rebinding consistent (outA == outB)

[5] in-place c2c  (N=16, zpFFT 패턴 — 입력버퍼=출력버퍼)
    in-place vs out-of-place 차이 = 0.00e+00
    [PASS] peak at expected bin (5)
    [PASS] in-place == out-of-place

=== 결과: 7 PASS, 0 FAIL ===
```

---

## 두 백엔드 & 정확성 대조

같은 `demo.c` 가 백엔드만 바꿔 빌드됩니다 (`fft_backend.h` 가 스위치).

```bash
make            # 커스텀 myfft            → ./fft_demo
make fftw       # 실제 libfftw3f (레퍼런스) → ./fft_demo_fftw   (FFTW 설치 필요)
make compare    # 두 백엔드 실행 결과 diff (헤더 줄 제외 동일해야 정상)
```

> `make fftw` 는 `libfftw3f`, `libfftw3f_threads` 가 필요합니다
> (Ubuntu: `sudo apt install libfftw3-dev`). 이 데모 개발 환경에는 FFTW 가 없어
> 커스텀 경로만 실행 검증했고, FFTW 링크 커맨드는 레이더 `Makefile` 과 동일하게 맞춰 뒀습니다.

**활용법**: 커스텀 FFT를 수정할 때마다 `make compare` 로 실 FFTW 와 수치 일치를 확인하세요.

---

## 각 테스트가 대표하는 레이더 변환

| 데모 테스트 | 레이더 원본 | 검증 포인트 |
|---|---|---|
| [1] c2c forward 1D | `plan_angleL/R` (2D AoA) | 단일 순변환 |
| [2] c2c backward + `/N` | `plan_iFFT` | **비정규화 규약** — 역변환 후 호출부가 직접 `/N` |
| [3] r2c forward | `plan_range_fft` | **반스펙트럼** N/2+1 출력 |
| [4] 배치 strided + `execute_dft` | `plan_doppler_fft` | 배치·strided 입력, **plan 버퍼 재바인딩** |
| [5] in-place c2c | `plan_zpFFTL/R` | **입력=출력** 동일 버퍼 |

---

## 파일

| 파일 | 설명 |
|---|---|
| `demo.c` | 5개 변환 패턴 데모 + 입력 생성 + 자체 검증. FFTW-API(`fftwf_*`)만 사용. |
| `myfft.h` | FFTW 호환 shim 헤더 (fftwf_* 부분집합). |
| `myfft.c` | **동작하는** radix-2 커스텀 FFT (c2c/r2c/배치/in-place, 비정규화). |
| `fft_backend.h` | 백엔드 선택 (`-DUSE_FFTW` 여부). |
| `Makefile` | `make` / `make fftw` / `make compare` / `make clean`. |

---

## 구현 제약 (커스텀 FFT를 레이더로 옮길 때)

1. **2의 승수 길이만** 지원(radix-2). 레이더 실제 크기(range 4095, doppler 12, angle 2000,
   iFFT/zpFFT 런타임)는 2의 승수가 아니므로, 호출부에서 zero-pad 로 크기를 맞추고
   다운스트림 빈↔물리량(거리/속도/각도) 매핑을 재보정해야 함. (별도 포팅 가이드 참조)
2. **비정규화**: forward/backward 모두 스케일 1. 역변환의 `/N` 은 호출부 책임(테스트 [2]).
3. **레이아웃 계약**: `fftwf_complex = float[2]` 인터리브 유지(구조체/시그니처/memcpy 의존).
4. **성능**: 현재 `myfft.c` 는 정확성 우선 스칼라 참조 구현. 실시간 경로(doppler 배치)는
   이후 트위들 캐시/실FFT packing/SIMD 등으로 최적화 필요.

---

## myfft 를 레이더 코드에 실제로 붙이려면

1. `myfft.h`, `myfft.c` 를 레이더 저장소로 복사.
2. `#include <fftw3.h>` → `#include "myfft.h"` (5곳: 두 analyze 파일 + `rd_mem_mng_util.h`,
   `data_save_util.h`, `data_save_util.c`).
3. `Makefile` 에서 `-lfftw3f_threads -lfftw3f` 제거, `SRCS_RADAR_COMMON` 에 `src/myfft.c` 추가.
4. 위 "구현 제약 1" 에 따라 5개 변환 크기를 2의 승수로 조정.
5. 기존 FFTW 바이너리와 출력 대조 검증.
