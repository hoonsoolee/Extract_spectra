# 초분광 노지 작물 분석 파이프라인 — 사용 설명서

## 개요

이 도구는 초분광 노지 작물 이미지의 픽셀을 자동으로 분류하고 클래스별 반사율 스펙트럼을 추출합니다. 결과는 CSV 파일과 인터랙티브 HTML 리포트로 저장됩니다.

---

## 1. 설치

### 요구 사항
- Python 3.9 이상
- Git

### 설치 순서

```bash
# 1. 저장소 클론
git clone https://github.com/hoonsoolee/Extract_spectra.git
cd Extract_spectra

# 2. 패키지 설치
pip install -r requirements.txt
```

> **참고:** `Autoencoder` 또는 `1D-CNN` 방법을 사용하려면 PyTorch가 추가로 필요합니다:
> ```bash
> pip install torch
> ```

---

## 2. 데이터 준비

초분광 이미지 파일을 `data/` 폴더(또는 원하는 폴더)에 넣으세요.

**지원 형식:**

| 형식 | 확장자 |
|------|--------|
| ENVI | `.hdr` (+ `.raw` / `.bil` / `.bip` / `.bsq`) |
| GeoTIFF | `.tif` / `.tiff` |
| HDF5 | `.h5` / `.hdf5` |
| MATLAB | `.mat` |

---

## 3. GUI 실행 (권장)

```bash
# 한국어 UI
python -m streamlit run app.py

# 영어 UI
python -m streamlit run app_en.py
```

브라우저가 자동으로 `http://localhost:8501`에서 열립니다.

### 사이드바 설정

| 항목 | 설명 |
|------|------|
| **데이터 소스** | *로컬 폴더* 선택 후 데이터 폴더 경로 입력 |
| **처리 모드** | *단일 파일 선택* — 폴더 스캔 후 파일 하나 선택 분석; *전체 배치 처리* — 모든 파일 순차 처리, 파일별 리포트 생성 |
| **분류 방법** | 아래 표 참조 |
| **클래스 수** | 감지할 클러스터/클래스 수 |
| **출력 폴더** | 결과 저장 위치 (기본값: `./output`) |
| **파일 수 제한** | 빠른 테스트 시 1~2로 설정 |

### 분류 방법

| 방법 | 유형 | 라벨 필요? | 특징 |
|------|------|-----------|------|
| **Hybrid** | 비지도 | 불필요 | **기본 추천.** NDVI → 밝기 → K-means |
| **K-Means** | 비지도 | 불필요 | 탐색적 분석 |
| **SAM** | 비지도 / 지도 | 선택 | 조명 불변 스펙트럼 각도 매핑 |
| **Random Forest** | 지도 | **필요** | 라벨 있을 때 높은 정확도 |
| **Autoencoder** | 비지도 | 불필요 | PyTorch 필요 |
| **1D-CNN** | 지도 | **필요** | PyTorch 필요 |

### 출력 파일

분석 완료 후 다음 파일이 생성됩니다:

```
output/
└── <파일명>/
    ├── spectra.csv       # 클래스별 평균·표준편차 반사율 스펙트럼
    ├── class_map.png     # 분류 맵 이미지
    └── report_YYYYMMDD_HHMMSS.html   # 인터랙티브 HTML 리포트
```

`.html` 파일을 브라우저에서 열면 다음 내용을 확인할 수 있습니다:
- RGB / CIR 합성 이미지
- 통합 분류 맵
- 클래스별 분류 이미지 (색상 강조)
- 인터랙티브 반사율 스펙트럼 차트
- 클러스터 품질 지표 (Silhouette, Davies-Bouldin)
- 식생 분리도 평가 (NDVI 기반 Recall / Precision / F1)
- **파일별 처리 시간**

---

## 4. 픽셀 라벨링 도구 (지도학습용)

**Random Forest** 또는 **1D-CNN**을 사용하려면 라벨이 필요합니다.

1. GUI의 **픽셀 라벨링** 탭으로 이동합니다.
2. 이미지 파일 경로(또는 폴더)를 입력하고 **로드**를 클릭합니다.
3. 클래스 이름과 색상을 설정합니다.
4. 이미지에서 픽셀을 클릭하여 클래스 라벨을 지정합니다.
5. **저장**을 클릭하여 `labels.csv`를 내보냅니다.
6. **분석 실행** 탭에서 `labels.csv` 경로를 입력하고 *Random Forest* 또는 *1D-CNN*을 선택합니다.

---

## 5. CLI (명령행 인터페이스)

```bash
# ./data 폴더의 모든 파일 처리 (기본 설정)
python main.py --local-folder ./data

# 단일 파일, K-Means, 8클래스
python main.py --local-folder ./data --method kmeans --n-clusters 8

# GitHub 저장소에서 처리
python main.py --github-repo owner/repo --github-folder data/2024

# 파일 목록만 확인 (처리 없음)
python main.py list --local-folder ./data
```

---

## 6. Hybrid 방법 — 클래스 ID

기본 **Hybrid** 방법 사용 시 다음 클래스 ID가 자동 부여됩니다:

| ID | 클래스 |
|----|--------|
| 0 | 배경 (Background) |
| 1 | 햇빛 받는 잎 (Sunlit Leaves) |
| 2 | 그림자 잎 (Shadowed Leaves) |
| 3 | 토양 (Soil) |
| 4 | 기타 (Other) |

---

## 7. 사용 팁

- **처음 사용 시** — *단일 파일 선택* 모드로 이미지 하나만 먼저 테스트한 후 전체 배치를 실행하세요.
- **처리 시간**이 완료 배너와 HTML 리포트에 표시됩니다 — 파일 하나의 시간을 기준으로 전체 배치 소요 시간을 가늠하세요.
- **불량 밴드** (1340–1460 nm, 1790–1960 nm)는 자동으로 제거됩니다.
- 모든 처리는 **CPU**로 실행됩니다. 큰 이미지(1000×1000 px 이상)는 파일당 수 분이 걸릴 수 있습니다.

---

## 8. 문제 해결

| 문제 | 해결 방법 |
|------|-----------|
| `ModuleNotFoundError` | `pip install -r requirements.txt` 실행 |
| `Unsupported format` 오류 | 파일 확장자가 지원 형식인지 확인 |
| ENVI 파일 로드 실패 | `.hdr`와 데이터 파일(`.raw`/`.bil` 등)이 같은 폴더에 있는지 확인 |
| 분류 맵이 비어 있음 | NDVI 임계값을 낮춰보세요 (기본값 0.15) |
| 처리 속도가 너무 느림 | `config.yaml`에서 `spatial_downsample: 2` 설정 |

---

*노지 초분광 작물 분석을 위해 개발되었습니다. 문의사항은 연구실로 연락하세요.*
