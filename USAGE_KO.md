# 초분광 노지 작물 분석 파이프라인 — 사용 설명서

처음 사용하는 경우에는 CERES, ROI 스펙트럼, 클러스터링 검수, 패널 보정과
결과파일을 실제 버튼 순서로 설명한 [한국어 실전 빠른 사용법](quick_start_ko.html)을
먼저 보세요.

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

한글·영문 UI는 같은 분석 코드를 실행합니다. 영문 UI는 표시 문구만
번역하므로 CERES, ROI, 패널 보정, 클러스터링, 리포트 기능이 동일합니다.

브라우저가 자동으로 `http://localhost:8501`에서 열립니다.

### 사이드바 설정

| 항목 | 설명 |
|------|------|
| **데이터 소스** | *로컬 폴더* 선택 후 데이터 폴더 경로 입력 |
| **처리 모드** | *단일 파일 선택* — 폴더 스캔 후 파일 하나 선택 분석; *전체 배치 처리* — 모든 파일 순차 처리, 파일별 리포트 생성 |
| **분류 방법** | 아래 표 참조 |
| **클래스 수** | 감지할 클러스터/클래스 수 |
| **결과 리포트** | 빠른 필드 QC, 연구용 표준 또는 사용자 지정 항목 선택 |
| **출력 폴더** | 결과 저장 위치 (기본값: `./output`) |
| **파일 수 제한** | 빠른 테스트 시 1~2로 설정 |

**분석 시작**을 누르면 계산은 별도 작업 프로세스에서 실행됩니다. 실행 중에는
사이드바 또는 분석 상태 영역의 **⏹️ 실행 중지**를 눌러 즉시 중지할 수 있습니다.
중지 전에 완성된 파일은 결과 폴더에 남고, 아직 작성 중이던 파일은 최종 결과로
간주하지 마세요. 실행 로그와 경과 시간은 화면에서 약 2초마다 갱신됩니다.

### 분류 방법

| 방법 | 유형 | 라벨 필요? | 특징 |
|------|------|-----------|------|
| **Hybrid** | 비지도 | 불필요 | **기본 추천.** NDVI → 밝기 → K-means |
| **K-Means** | 비지도 | 불필요 | 탐색적 분석 |
| **SAM** | 비지도 / 지도 | 선택 | 조명 불변 스펙트럼 각도 매핑 |
| **HDBSCAN** | 비지도 | 불필요 | 클러스터 수 자동 결정; 불규칙 형태 클러스터에 최적 |
| **GMM** | 비지도 | 불필요 | 확률적 소프트 클러스터링; 겹치는 클래스에 유리 |
| **NMF** | 비지도 | 불필요 | 스펙트럼 언믹싱; 혼합 픽셀 분석에 적합 |
| **Random Forest** | 지도 | **필요** | 라벨 있을 때 높은 정확도 |
| **Autoencoder** | 비지도 | 불필요 | PyTorch 필요 |
| **1D-CNN** | 지도 | **필요** | PyTorch 필요 |

### 출력 파일

분석 완료 후 다음 파일이 생성됩니다:

완료 화면의 **최근 분석 결과 열기** 영역에서 HTML 리포트를 선택해 기본 브라우저로
바로 열거나, 결과 폴더를 Windows 파일 탐색기로 열 수 있습니다. Illinois Campus
Cluster처럼 원격 서버에서 실행할 때는 **HTML 리포트 다운로드**를 사용하세요.

```
output/
└── <파일명>/
    ├── spectra_{method}.csv                  # 예: spectra_kmeans.csv
    ├── spectra_{method}_reflectance.csv      # 보정 적용 시 과학용 반사율
    ├── spectra_{method}_raw_dn.csv           # 같은 클러스터의 보정 전 DN
    ├── processing_manifest.json              # 보정·정규화·원본 이력
    ├── report_config.json                     # 이번 실행에서 선택한 리포트 항목
    ├── class_map_{method}.png                # 예: class_map_hybrid.png
    ├── rgb.png / ndvi.png                    # 선택한 경우만 생성
    ├── cluster_map.png / cluster_overlay.png # 선택한 경우만 생성
    └── report_YYYYMMDD_HHMMSS_{method}.html  # 예: report_20260408_130351_kmeans.html
```

배치 처리에서는 선택에 따라 출력 루트에 `daily_report_<timestamp>.html`과
`daily_summary_<timestamp>.csv`도 생성됩니다. 하루 전체 파일의 보정파일, NDVI 평균·중앙값,
식생 비율, 클래스 수, 클러스터 품질 및 처리 시간을 한 표에서 비교할 수 있습니다.

### 팀·플랏 일일 통합 패키지

배치 모드에서 **팀·플랏 일일 통합 리포트**를 켜고 실제 측정일과 팀 이름을 입력하면
파일별 결과를 다시 계산하지 않고 다음 공유용 패키지를 만듭니다. 원본 CERES/BIL 큐브를
다시 RAM에 올리지 않으므로 추가 메모리 부담은 작습니다.

```text
output/team_reports/<측정일>_<팀>_<생성시각>/
├── Team_Report.html
├── Field_Results.xlsx
├── Field_Summary.csv
├── Warnings.csv
├── plots_overview.png
├── plots_ndvi.png
├── plot_ndvi_comparison.png
├── Images/                  # 공유에 필요한 플랏별 이미지 복사본
└── Details/                 # 플랏별 상세 HTML 복사본
```

- 모든 NDVI 타일은 동일한 `-1~1` 색상범위를 사용합니다.
- 전체 픽셀을 섞지 않고 **각 플랏의 중앙값과 IQR**을 비교합니다.
- 팀 통계에는 `value_units=reflectance`, `calibration_qc_status=PASS`, NDVI 계산 가능 조건을
  모두 만족한 플랏만 포함됩니다. `REVIEW`, `FAIL`, 미보정 파일은 `Warnings`로 분리됩니다.
- `Field_Results.xlsx`에는 Dashboard, README, Field Summary, Cluster Summary,
  Reflectance Spectra, Warnings 시트가 생성됩니다.

파일명이 플랏 ID가 아닌 경우 선택형 메타데이터 CSV를 사용합니다:

```csv
filename,plot_id,treatment,genotype,replicate,team,measurement_date
scene_001.bil,AP3-4,Control,WT,1,Team A,2026-08-27
```

### 선택형 결과 리포트

- **빠른 필드 QC**: RGB, NDVI, 클러스터 맵·오버레이, 평균·중앙값 스펙트럼,
  보정 QC 및 처리 시간을 저장합니다. 무거운 분리도 평가는 건너뛰므로 기본 추천입니다.
- **연구용 표준 리포트**: CIR, 클러스터별 분리 이미지, 평균/중앙값/표준편차/IQR,
  NDVI·GNDVI·NDRE·PRI, 클러스터 품질과 식생 분리도까지 포함합니다.
- **사용자 지정**: 이미지, 스펙트럼 통계, 식생지수, HTML·CSV·PNG 및 배치 요약을
  체크박스로 직접 선택합니다.

식생지수는 **보정 반사율**과 필요한 파장 밴드가 모두 있을 때만 계산됩니다. 원본 DN이거나
파장이 부족한 경우 임의 값을 만들지 않고 HTML과 manifest에 계산하지 못한 이유를 기록합니다.

출력 파일명에 방법 이름이 포함되므로 같은 폴더에 여러 방법의 결과를 함께 저장해도 덮어쓰기가 발생하지 않습니다.

결과 표는 현재 `.xlsx` 통합문서가 아니라 **Excel에서 바로 열 수 있는 UTF-8 CSV**입니다.
스펙트럼 CSV는 한 행이 한 파장이고, 파일명 접미사에 따라 값의 의미가 달라집니다.

| 파일 | 의미 |
|------|------|
| `spectra_{method}_reflectance.csv` | 유효한 보정이 적용된 과학 분석용 반사율 |
| `spectra_{method}_raw_dn.csv` | 보정 전 센서 DN; 보정 전후 비교와 오류 진단용 |
| `spectra_{method}_processed.csv` | 보정이 없을 때 정규화/전처리된 상대값; 절대 반사율이 아님 |
| `spectra_{method}.csv` | 현재 실행에서 추출된 대표 파일; 정확한 단위는 `value_units` 열로 확인 |
| `daily_summary_*.csv` | 배치 파일별 NDVI, 식생 비율, 클래스 수, 품질지표와 처리시간 |
| `all_roi_cluster_spectra*.csv` | 모든 ROI·클러스터 스펙트럼을 합친 표 |
| `cluster_summary.csv` | ROI별 클러스터 픽셀 수와 면적 비율(`fraction`, 0–1) |

각 스펙트럼 파일은 **클래스별·밴드별 7개 통계값**을 포함합니다:

| 열 접미사 | 설명 |
|-----------|------|
| `mean` | 클래스 내 모든 픽셀의 평균 반사율 |
| `std` | 표준편차 (스펙트럼 변동성) |
| `median` | 밴드별 중앙값 |
| `q25` / `q75` | 25번째 / 75번째 백분위수 |
| `mna` | **Medoid-Neighbourhood Average** — 클래스 중앙값에 유클리드 거리가 가장 가까운 100개 픽셀의 평균. 이상치를 제거하면서 실제 픽셀만 평균. |
| `sam_avg` | **SAM-Neighbourhood Average** — 중앙값에 스펙트럼 각도가 가장 작은 100개 픽셀의 평균. 조명 불변; Vcmax · 생화학 특성 매칭에 권장. |

단, ROI의 `cluster_spectra*` 파일은 비교·병합하기 쉬운 long 형식으로
`ROI × 클러스터 × 파장`마다 한 행을 사용하며 `mean`, `median`, `std`, `q25`, `q75`를 저장합니다.

열 이름은 `{파일명}_{방법}_{클래스}_{통계}` 형식(예: `AP3-4_kmeans_Sunlit_Leaves_sam_avg`)이므로 여러 파일·방법의 CSV를 합쳐도 열 충돌이 없습니다. `mna`·`sam_avg`에 사용하는 이웃 수(기본값 100)는 `config.yaml`의 `extraction.n_neighbors`에서 변경할 수 있습니다.

각 스펙트럼 CSV 앞부분에는 `value_units`, `calibration_applied`,
`calibration_profile`, `calibration_method`, `dark_source_type`, 패널 요약과
`normalization_mode`가 함께 기록됩니다. 보정 반사율 CSV에는 각 파장의 경험선 계수
`calibration_a`, `calibration_b`도 저장됩니다. raw-DN 파일은
`calibration_applied=false`이지만 대응되는 보정파일을 `paired_calibration_profile`에 남깁니다.
논문용 반사율은 `_reflectance.csv`에서 `value_units=reflectance`,
`calibration_applied=true`, `calibration_qc_status=PASS`를 우선 확인합니다. `REVIEW`는
점프·포화 경고를 검토한 뒤 사용하고 `FAIL`은 사용하지 않습니다.

`.html` 파일을 브라우저에서 열면 다음 내용을 확인할 수 있습니다:
- RGB / CIR 합성 이미지
- 통합 분류 맵
- 클래스별 분류 이미지 (색상 강조)
- 인터랙티브 반사율 스펙트럼 차트
- 클러스터 품질 지표 (Silhouette, Davies-Bouldin) + 색상 코딩 해석 박스
- 식생 분리도 평가 (NDVI 기반 Recall / Precision / F1) + 개선 안내
- **파일별 처리 시간**

리포트의 RGB, 클러스터 맵, 오버레이, 클러스터별 이미지는 넓은 화면에서도 2열로 크게
표시됩니다. 이미지를 클릭하면 화면에 맞춰 확대되며 바깥쪽, 닫기 버튼 또는 `Esc`로 닫습니다.

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

## 7. 패널 ROI + Dark 자동 반사율 보정

GUI의 **패널 보정** 탭에서 다음 순서만 수행합니다.

1. 패널이 찍힌 영상 또는 패널이 함께 나온 현장 영상을 엽니다.
2. 각 패널 안쪽을 Box/Lasso로 지정하고 인증 반사율을 입력합니다. 예: 99%=0.990, 50%=0.500.
3. Dark 준비 방법을 선택합니다.
   - Dark 영상이 없으면 기본 **수동 상수 DN 100**을 그대로 사용하거나 확인한 평균값으로 바꿉니다.
   - Dark 영상이 있으면 **실측 Dark 파일**을 선택하고 같은 integration time과 gain의 영상을 불러옵니다.
4. **자동 반사율 보정 계산**을 누릅니다.

프로그램은 파장별 포화와 Dark 대비 SNR을 검사합니다. 밝은 패널이 포화에 가까워지면
가중치를 부드럽게 낮추고, 같은 파장에서 정상인 낮은 반사율 패널을 공통 회귀식에
자동으로 사용합니다. 모든 패널이 불량인 밴드는 추정값으로 채우지 않고 NaN으로 표시합니다.

수동 상수 Dark는 모든 파장의 Dark가 거의 평평하고 약 100 DN이라는 사전 확인을 바탕으로 한
대체 방법입니다. 프로그램은 보정파일에 `dark_source_type=synthetic_constant`와 사용 DN을
기록합니다. 빠른 선별·리포트에는 사용할 수 있지만, 논문용 최종 정량 결과에는 같은 센서 설정의
실측 Dark를 권장합니다.

보정계수는 `output/calibration/`에 자동 저장됩니다. QC PASS/REVIEW만 **분석 실행**과
**ROI 스펙트럼**에 연결되고 FAIL은 자동 적용이 차단됩니다. 연결된 경우 추가 정규화는 꺼져
계산된 반사율을 유지합니다. **현재 영상 반사율 BIL 만들기**를
누르면 `output/reflectance/`에 float32 BIL/HDR 및 보정 이력 JSON이 저장됩니다.

앱을 다시 시작해도 분석할 영상과 같은 이름의 보정파일을 원본 폴더 및
`output/calibration/`에서 찾아 밴드 수·파장축을 검사한 뒤 자동 적용합니다. 과거에 이미 만든
CSV는 자동으로 바뀌지 않으므로, `processing_manifest.json`의 `calibration`이 `null`인 결과는
새 보정파일로 다시 분석해야 합니다.

여러 시각의 White를 자동 선택하거나 기존 `.npz`를 직접 지정하는 기능은 각 화면의
**고급 설정**에 있습니다.

---

## 8. 사용 팁

### CERES 컨테이너를 바로 확인하는 방법

1. **로컬 폴더 → 단일 파일 선택 → 폴더 스캔**에서 `.ceres` 파일을 고릅니다.
2. **CERES 내부 목록 읽기**를 누르면 픽셀 전체를 읽지 않고 레코드 헤더만 검사하여
   `A/VNIR`, `A/SWIR`, `B/VNIR` 같은 실제 촬영 항목을 표시합니다. 같은 파일은 인덱스를
   캐시하므로 다음에는 즉시 열립니다. 2024 CBDF v1과 신규 CBDF v2를 모두 인식합니다.
3. 항목을 하나 고르고 **빠른 미리보기**를 누릅니다. VNIR은 가시광 RGB, SWIR은 가색 3개 밴드만 직접 읽어
   전체 CERES나 전체 초분광 큐브를 RAM에 올리지 않습니다.
4. 분석할 항목이 맞으면 **선택 항목 분석 준비**를 누릅니다. 선택한 센서/구간만
   `output/_ceres_cache`의 uint16 BIL/HDR로 만들고 일반 분석 입력에 연결합니다.
5. 표시된 메모리 예상표를 보고 공간 다운샘플링을 선택한 뒤 분석을 시작합니다.

현재 단계는 안전한 **탐색·선택·선택 구간 변환**까지 지원합니다. CERES 하루 폴더를 BIL 없이
완전 자동 처리하려면 표본 기반 클러스터 모델 학습 후 레코드를 두 번 스트리밍하는 방식이
필요하며, 이는 다음 단계입니다. 현재 배치 분석은 준비된 BIL/HDR을 대상으로 실행하세요.

### 보정 QC와 클러스터링 입력

- 새 보정파일은 생성 즉시 `PASS`, `REVIEW`, `FAIL`로 점검됩니다. 패널 복원 오차,
  ROI 균일도, 패널 간 계수 차이, 인접 파장 계단, 패널 가중치 급변을 함께 검사합니다.
- `FAIL` 파일은 감사용으로 저장되지만 전체 분석에 자동 연결되지 않습니다.
- 합성 상수 Dark를 사용한 결과는 측정 Dark가 아니므로 최소 `REVIEW`입니다.
- 기본 K-Means 클러스터링은 원본 DN의 스펙트럼 구조로 수행하고, 같은 클러스터 마스크를
  보정된 큐브에 적용해 반사율 스펙트럼과 지수를 저장합니다. Hybrid만 보정파일이 있으면
  NDVI·밝기 임계값 때문에 반사율을 클러스터 입력으로 자동 사용합니다.
- 분석 완료 후 **클러스터링 결과 이미지 검수**에서 분석 RGB, 클러스터 컬러맵,
  RGB 오버레이를 한 화면에서 비교할 수 있습니다. 표시할 클러스터, 색상 투명도와 경계선을
  조절하고 **클러스터별 단독 이미지**로 잎·그림자·과반사·토양 분리를 확인합니다.

- **Windows 경로 선택** — 데이터 폴더, 결과 폴더, ROI/패널/Dark 파일 옆의 `🪟 선택`을
  누르면 Windows 탐색기 창에서 경로를 고를 수 있습니다. 파일은 웹으로 업로드되지 않고
  로컬 경로만 전달되므로 수십 GB 자료에도 사용할 수 있습니다. Illinois Campus Cluster처럼
  원격 Linux 서버에서 실행할 때는 버튼이 비활성화되며 기존처럼 서버 경로를 직접 입력합니다.
- **처음 사용 시** — *단일 파일 선택* 모드로 이미지 하나만 먼저 테스트한 후 전체 배치를 실행하세요.
- **처리 시간**이 완료 배너와 HTML 리포트에 표시됩니다 — 파일 하나의 시간을 기준으로 전체 배치 소요 시간을 가늠하세요.
- **불량 밴드** (1340–1460 nm, 1790–1960 nm)는 자동으로 제거됩니다.
- 모든 처리는 **CPU**로 실행됩니다. 큰 이미지(1000×1000 px 이상)는 파일당 수 분이 걸릴 수 있습니다.

---

## 9. 전역 클러스터링 + 여러 구역(ROI)별 스펙트럼

전체 영상을 한 번 클러스터링한 뒤, 플롯·처리구별 ROI에서 그 전역 클러스터의 스펙트럼을
따로 저장합니다. 특정 ROI만 결과가 나쁘면 그 ROI에만 별도 설정으로 재클러스터링합니다.

```bash
python -m streamlit run app_roi_clustering.py
```

1. ENVI 데이터의 `.hdr` 또는 `.bil` 경로를 입력합니다. 어느 쪽을 선택해도 companion 파일을 자동 탐색합니다.
2. 큰 파일은 공간 다운샘플링 `4`로 먼저 로드합니다.
3. **구역 나누기** 탭에서 Box, Lasso 또는 **Polygon 클릭 ROI**로 여러 ROI를 만들고 이름을 지정합니다.
4. Polygon은 잎 둘레의 꼭짓점을 차례로 클릭한 뒤 **Polygon 완료**를 누르고 구역을 추가합니다. 마지막 점 취소와 전체 지우기도 가능합니다.
5. 패널에서 만든 `.npz`는 같은 영상 이름이면 자동 연결되며, 화면의 초록색 적용 배너로 확인합니다.
6. **전체 클러스터링 후 ROI 스펙트럼 추출**을 누릅니다. 모든 ROI가 같은 전역 클러스터 기준을 공유합니다.
7. 결과가 나쁜 ROI만 구역별 설정값으로 재클러스터링하고, 기존 전역 결과와 비교한 뒤 채택합니다.
8. 결과 탭에서 클러스터 맵과 평균/중앙값 스펙트럼을 확인하고 HTML 리포트를 저장합니다.

출력 폴더에는 ROI 정의와 실행 설정을 기록한 `analysis_manifest.json`, 전체/ROI별 스펙트럼
CSV, 클러스터 맵 PNG/NPZ, 요약 CSV, `report.html`이 생성됩니다. 평균과 표준편차는 ROI의
전체 유효 픽셀로 계산하고, median 및 IQR은 대용량 메모리 사용을 막기 위해 클러스터별
대표 표본으로 계산합니다.

다운샘플링된 화면의 ROI 좌표는 다운샘플링된 픽셀 좌표입니다. 동일 ROI를 원본 해상도에서
재사용할 때는 좌표에 다운샘플링 배율을 적용해야 합니다.

메인 앱의 **ROI 스펙트럼** 화면은 `ROI 이미지를 전체 폭으로 보기`가 기본입니다. 세로 또는
가로로 매우 긴 라인스캔 영상은 행/열 표시 구간 슬라이더가 자동으로 나타나며, 잘라서 보이는
구간에서 선택한 ROI는 현재 불러온 큐브의 전체 좌표로 자동 환산됩니다.

작은 부분을 확대해 ROI로 잡으려면 `🔍 확대`를 선택하고 이미지에서 드래그하거나 마우스 휠을
사용합니다. 원하는 크기로 확대된 뒤 Box/Lasso로 드래그하거나 Polygon 모드에서 잎 윤곽을
점으로 클릭합니다. `↩️ 확대 초기화`로 전체 화면으로 돌아갑니다.
확대 상태는 모드를 전환해도 유지됩니다.

스펙트럼 아래의 표시 범위에서는 **900 nm까지만 보기**, 전체 파장 복원, 파장 범위 슬라이더와
Y축 자동/0–1/직접 입력을 사용할 수 있습니다. 이 설정은 화면 그래프만 확대하며, 계산 결과와
저장되는 CSV에는 전체 밴드가 그대로 유지됩니다.

보정파일이 연결된 경우 ROI를 선택하면 스펙트럼 위에 **보정 반사율 / 원본 DN /
원본·보정 비교** 버튼이 나타납니다. 비교 모드는 반사율과 DN의 크기가 달라도 형태를 비교할 수
있도록 좌우 Y축을 따로 사용합니다. **보정 이상 밴드·계수 진단**을 열면 `R = a×DN+b`의
`a`, `b`, 무효·음수 gain, 극단적 gain 및 ROI 평균 반사율이 -0.05–1.20 범위를 벗어난 밴드를
확인할 수 있습니다. 진단은 값을 자동으로 고치지 않으므로 패널 포화, White/Dark 선택과 해당
파장의 보정계수를 검토하는 용도로 사용하세요.

---

## 10. 문제 해결

| 문제 | 해결 방법 |
|------|-----------|
| `ModuleNotFoundError` | `pip install -r requirements.txt` 실행 |
| `Unsupported format` 오류 | 파일 확장자가 지원 형식인지 확인 |
| ENVI 파일 로드 실패 | `.hdr`와 데이터 파일(`.raw`/`.bil` 등)이 같은 폴더에 있는지 확인 |
| 분류 맵이 비어 있음 | NDVI 임계값을 낮춰보세요 (기본값 0.15) |
| 처리 속도가 너무 느림 | `config.yaml`에서 `spatial_downsample: 2` 설정 |

---

*노지 초분광 작물 분석을 위해 개발되었습니다. 문의사항은 연구실로 연락하세요.*
