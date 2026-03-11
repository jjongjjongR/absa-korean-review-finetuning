# ABSA Korean Review Finetuning

리뷰 문장(`text`)과 속성(`aspect`)별 감성(`label`)을 다루는 ABSA 프로젝트입니다.

## 데이터 위치

- 실제 데이터: `data/raw/data.csv`

필수 컬럼:

- `text`
- `aspect`
- `label`

## 모델 위치

- `models/klue-bert-base`
- `models/klue-roberta-base`

현재 환경에서는 Hugging Face direct download가 차단되어 있어, 로컬 모델 폴더를 사용하도록 검증했습니다.

## 실행

Step1:

```bash
python scripts/train_step1.py --model-name models/klue-bert-base --epochs 1 --batch-size 8 --sample-per-class 50 --out-dir verify_step1
```

Step2:

```bash
python scripts/train_step2.py --model-name models/klue-bert-base --epochs 1 --batch-size 8 --sample-size 200 --out-dir verify_step2
```

Step3:

```bash
python scripts/train_step3.py --model-name models/klue-bert-base models/klue-roberta-base --format case1 case2 --epochs 1 --batch-size 8 --sample-size 200 --out-dir verify_step3
```

## 이번 검증 결과

- 검증 일시: 2026-03-11
- 실제 `data/raw/data.csv` 사용
- 로컬 모델 폴더 사용

Step1:
- 실행 성공
- 결과:
  - `eval_accuracy = 0.7500`

Step2:
- 실행 성공
- 결과:
  - `eval_exact_match = 0.1167`

Step3:
- 실행 성공
- 조합별 결과:
  - `klue-bert-base + case1 = 0.8167`
  - `klue-bert-base + case2 = 0.7833`
  - `klue-roberta-base + case1 = 0.6167`
  - `klue-roberta-base + case2 = 0.6167`

## 산출물

- `verify_step1/`
- `verify_step2/`
- `verify_step3/`

## 구현 요약

- `scripts/train_step1.py`
  - notebook 기준 binary sentiment fine-tuning
  - 평가 결과와 예측 저장
- `scripts/train_step2.py`
  - notebook 기준 multi-label aspect detection
  - `get_dummies + groupby max` 방식 반영
- `scripts/train_step3.py`
  - `case1`, `case2` 입력 포맷 모두 지원
  - `bert`, `roberta` 비교 실행 지원
