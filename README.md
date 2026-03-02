# 리뷰 속성별 감성분석 파인튜닝 (ABSA)

리뷰 텍스트의 다양한 속성(가격/향/용량/발림성 등)에 대해 **속성별 긍/부정 감성**을 분류하는 ABSA 프로젝트입니다.  
한국어 사전학습모델(`klue/bert-base`, `klue/roberta-base`)을 활용해 입력 구조를 비교 실험했습니다.

- 입력 구조 비교
  - `"[ASPECT] [SEP] TEXT"`
  - `"ASPECT: TEXT"`

원본 실험은 `notebooks/`에, (선택) 재현 스크립트는 `scripts/`에 정리했습니다.

## 내 역할

- **발표자**: 프로젝트 흐름 구조화, 결과/인사이트 정리 및 QnA 준비

## 폴더 구조

```
.
├─ notebooks/
├─ assets/
├─ reports/
├─ src/absa/
└─ scripts/
```

## 데이터 준비

노트북에서 `data.csv`를 로드하므로 아래 위치에 두는 것을 기준으로 구성했습니다.

- `data/raw/data.csv`

권장 컬럼(노트북 기준):
- `text`: 리뷰 문장
- `label`: 정답(0/1 또는 스코어)
- `aspect`: 속성명(예: 보습/가격/품질 등)

## 실행 방법 (선택)

딥러닝 파인튜닝을 포함하므로 환경이 무거울 수 있습니다.

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt

python scripts/train_step1.py --model-name klue/bert-base
python scripts/train_step2.py --model-name klue/bert-base
python scripts/train_step3.py --model-name klue/bert-base --format case1
```

출력 디렉토리:
- `artifacts/step1/`, `artifacts/step2/`, `artifacts/step3/`

## 결과/성과 (요약)

- BERT/RoBERTa 모델에서 입력 구조 2종 비교 및 속성별 Precision/Recall/F1 시각화
- 최고 F1-score **0.89** (BERT, `[ASPECT]+[SEP]` 구조 / 팀 보고서 기준)
- 스페셜 토큰 `[ASPECT]` 추가 시 일부 속성에서 성능 향상 확인

![결과1](assets/결과%201.png)
![결과2](assets/결과%202.png)

## 트러블슈팅 (요약)

- `[ASPECT]` 스페셜 토큰 처리: `tokenizer.add_special_tokens()` + `resize_token_embeddings()` 적용
- 데이터 구조 정의 혼동(멀티라벨 vs 이진): Step별 입력/라벨 구조 명확화
- 평가 단계 인덱스 불일치: pandas 변환 후 병합 검증
- 하이퍼파라미터 튜닝: Optuna 기반 탐색 도입

## 링크

- PPT: `https://www.notion.so/PPT-2a6c8182fdea8013a318ee40b016e983?pvs=21`
- 보고서: `https://www.notion.so/2a6c8182fdea80b895f1fb3924f52d99?pvs=21`
