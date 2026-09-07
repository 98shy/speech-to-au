# Speech → AU25/AU26 (lip-sync only)

이 패키지는 **립싱크 관련 AU25(Lips Part)와 AU26(Jaw Drop) 두 개만** 예측합니다.
다른 AU(눈썹/눈/뺨 등 표정 관련)는 이 패키지의 범위가 아니며, 애초에 신뢰할 수
있는 성능이 나오지 않아 의도적으로 제외했습니다.

## 왜 AU25/AU26만인가

`experiment/grid_speechau_v1_1`에서 4개 predictor(TCN / TCN+GPT2 / Mamba /
Diffusion) x 3개 feature(MFCC / WavLM / MFCC+WavLM)로 9개 AU
(AU10/12/14/15/17/20/23/25/26)를 전부 실험한 결과, **AU25와 AU26만** 모델·피처에
관계없이 일관되게 ICC(3,1) 0.6~0.7 수준의 유의미한 예측력을 보였습니다. 나머지
AU는 전부 ICC(3,1)이 0 근방으로, 사실상 "아무거나 예측 안 하는 것"과 통계적으로
구분이 안 되는 수준입니다. 이런 AU를 렌더링 팀에 넘기면 오히려 잘못된 애니메이션을
만들게 되므로 명시적으로 제외했습니다.

## 파이프라인

```
raw wav (16kHz)
  -> audio_feature_extractor.extract_combined()   # MFCC(13) + WavLM-base-plus(768) = 781차원, 25fps
  -> AU25: Mamba(+combined) 체크포인트
  -> AU26: TCN(+combined) 체크포인트
  -> AU25/AU26 강도 곡선 (0~5 스케일, 25fps)
```

두 AU는 서로 다른 모델에서 나옵니다 — AU25는 Mamba, AU26은 TCN. 이는 각 AU별로
9개 AU를 동시에 예측하는 원래 모델(각각 9-AU 출력을 가짐)에서 **해당 AU 채널만
뽑아 쓰는 것**입니다.

## 사용법

```bash
python infer.py --wav path/to/speech.wav --out predictions.npz
```

`predictions.npz`에는 `timestamp_sec`, `AU25`, `AU26` (모두 길이 T, 25fps 정렬)가
들어 있습니다.

## 성능 (GRID 고정 test split, 2887발화)

| AU | 모델 | MAE | PCC | ICC(3,1) |
|---|---|---|---|---|
| AU25 | Mamba+Combined | 0.335 | 0.702 | 0.702 |
| AU26 | TCN+Combined | 0.236 | 0.701 | 0.693 |

## 한계 및 주의사항 (자세한 건 config.json 참고)

1. **AU10/12/14/15/17/20/23은 지원하지 않습니다.** 시도했지만 신뢰할 수 없었습니다.
2. **눈썹/눈 관련 AU는 아예 데이터셋에 없습니다.**
3. GRID 코퍼스(짧고 깨끗한 단일 화자 낭독체, ~3초)로만 검증했습니다 — 잡음이 있거나
   여러 화자가 겹치는 음성, 감정 표현이 강한 음성에서의 성능은 검증되지 않았습니다.
4. 타깃 자체가 전문가 FACS 라벨이 아니라 **OpenFace pseudo-label**입니다.
5. 강도(intensity, 0~5 연속값)만 예측하며, 발생 여부(presence)는 별도로 계산하지
   않습니다.

## 폴더 구성

```
config.json               -- 전체 스펙(피처 추출 파라미터, 체크포인트 경로, 성능, 한계) 기계 판독 가능한 형태
infer.py                  -- wav -> AU25/AU26 추론 스크립트
audio_feature_extractor.py -- wav -> MFCC+WavLM 781차원 피처 추출
model/                    -- TCN, Mamba 모델 정의 (predictor 구조 코드)
checkpoints/              -- mamba_combined.pt (AU25용), tcn_combined.pt (AU26용)
requirements.txt          -- 필요 파이썬 패키지
```

이 폴더 하나만 있으면 실행 가능한 self-contained 패키지입니다 (외부 프로젝트 경로에 의존하지 않음).

```bash
pip install -r requirements.txt
python infer.py --wav path/to/speech.wav --out predictions.npz
```
