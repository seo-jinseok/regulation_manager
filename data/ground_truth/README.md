# RAG Ground Truth Dataset

regulation_manager RAG 시스템 평가를 위한 Ground Truth 질문-정답 쌍 데이터셋입니다.

## 데이터셋 개요

- **데이터셋 ID**: `rag_gt_v1.0`
- **총 질문-정답 쌍**: 500개
- **생성 방식**:
  - Flip-the-RAG 자동 생성: 300개 (60%)
  - Expert 템플릿: 200개 (40%)
- **데이터 분할**:
  - Train: 350개 (70%)
  - Validation: 75개 (15%)
  - Test: 75개 (15%)

## 데이터 구조

### 질문-정답 쌍 형식

```json
{
  "id": "gt_001",
  "query": "졸업 요건은 어떻게 되나요?",
  "answer": "졸업 요건은 다음과 같습니다: 총 140학점 이수, 전공 60학점, 교양 30학점",
  "context": ["교육과정규정 제25조"],
  "category": "졸업",
  "difficulty": "중급",
  "query_type": "자연어 질문",
  "metadata": {
    "source": "교육과정규정",
    "article": "제25조",
    "keywords": ["졸업", "요건", "학점"]
  }
}
```

### 카테고리

- 졸업
- 휴학
- 복학
- 장학금
- 등록
- 성적
- 교과과정
- 교환학생
- 규정해석
- 신청절차
- 기간
- 서류
- 자격

### 난이도

- **초급**: 단순 사실 확인
- **중급**: 이해 필요
- **고급**: 종합적 사고 필요

### 질문 유형

- 정확한 쿼리
- 구어체 쿼리
- 모호한 쿼리
- 오타 포함 쿼리
- 영문 혼용 쿼리
- 복합 질문
- 문맥 의존 질문

## 데이터셋 생성

### 자동 생성 (Flip-the-RAG)

규정 문서에서 답변을 추출하고 LLM으로 질문을 생성합니다.

```python
from rag.data_generation import FlipTheRAGGenerator

generator = FlipTheRAGGenerator()
pairs = generator.generate(
    regulation_dir=Path("data/processed/regulations"),
    target_pairs=300
)
```

### 전문가 템플릿

다양한 시나리오와 질문 유형을 커버하는 템플릿을 제공합니다.

```python
from rag.data_generation import ExpertTemplateGenerator

template_gen = ExpertTemplateGenerator()
templates = template_gen.generate_templates(target_count=200)
```

## 데이터 검증

데이터셋 품질 검증 시스템을 제공합니다.

### 검증 항목

1. **답변 품질** (Relevance, Accuracy, Sufficiency, Clarity)
2. **질문 다양성** (중복 검증, 길이 분포)
3. **카테고리 균형** (분포 분석)
4. **난이도 분포** (초급/중급/고급 비율)
5. **질문 유형 분포** (7가지 유형 커버)
6. **답변 완전성** (빈 답변, 짧은 답변 검증)

### 검증 실행

```python
from rag.data_generation import DataValidator

validator = DataValidator()
results = validator.validate_dataset(pairs)
validator.print_summary(results)
```

## 사용 방법

### CLI로 데이터셋 생성

```bash
# 기본: 500개 질문-정답 쌍 생성
python scripts/generate_ground_truth.py --count 500

# 사용자 정의 설정
python scripts/generate_ground_truth.py \
    --count 1000 \
    --flip-ratio 0.7 \
    --output data/ground_truth

# 검증만 실행
python scripts/generate_ground_truth.py \
    --validate-only \
    --input data/ground_truth
```

### 데이터셋 로드

```python
import json

# Train 데이터 로드
with open("data/ground_truth/train/train.jsonl", "r") as f:
    train_pairs = [json.loads(line) for line in f]

# Validation 데이터 로드
with open("data/ground_truth/val/val.jsonl", "r") as f:
    val_pairs = [json.loads(line) for line in f]

# Test 데이터 로드
with open("data/ground_truth/test/test.jsonl", "r") as f:
    test_pairs = [json.loads(line) for line in f]
```

## RAG 평가

### RAGAS 메트릭 계산

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# RAG 파이프라인 실행
results = []
for pair in test_pairs:
    retrieved_contexts = retriever.retrieve(pair["query"])
    generated_answer = generator.generate(pair["query"], retrieved_contexts)

    results.append({
        "question": pair["query"],
        "answer": generated_answer,
        "contexts": retrieved_contexts,
        "ground_truth": pair["answer"],
    })

# RAGAS 평가
evaluation_results = evaluate(
    results,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]
)

print(evaluation_results)
```

## 파일 구조

```
data/ground_truth/
├── README.md                    # 이 문서
├── metadata.json                # 데이터셋 메타데이터
├── quality_report.html          # 품질 보고서 (HTML)
├── QUALITY_REPORT.md            # 품질 보고서 (Markdown)
├── validation_report.json       # 검증 결과 상세
├── train/
│   └── train.jsonl             # Train 데이터 (350개)
├── val/
│   └── val.jsonl               # Validation 데이터 (75개)
└── test/
    └── test.jsonl              # Test 데이터 (75개)
```

## 품질 보고서

데이터셋 생성 시 자동으로 품질 보고서가 생성됩니다:

- `quality_report.html`: 시각적 HTML 보고서
- `QUALITY_REPORT.md`: Markdown 형식 보고서
- `validation_report.json`: 원시 검증 데이터

### 품질 기준

- **우수 (EXCELLENT)**: 0.8 ~ 1.0
- **양호 (GOOD)**: 0.6 ~ 0.8
- **보통 (FAIR)**: 0.4 ~ 0.6
- **미흡 (POOR)**: 0.0 ~ 0.4

## 모듈 구조

```python
src/rag/data_generation/
├── __init__.py                  # 모듈 초기화
├── flip_the_rag_generator.py   # Flip-the-RAG Generator
├── templates.py                 # Expert Template Generator
├── validator.py                 # Data Validator
├── quality_reporter.py          # Quality Reporter
└── build_dataset.py            # Dataset Builder (Main)
```

## 참고사항

### Flip-the-RAG 방식

Flip-the-RAG는 기존 RAG 방식을 반전시킨 접근법입니다:
1. **기존 RAG**: 질문 → 문서 검색 → 답변 생성
2. **Flip-the-RAG**: 문서 → 답변 추출 → 질문 생성

이 방식은 다음과 같은 장점이 있습니다:
- 규정 문서에 기반한 실질적인 질문 생성
- 다양한 질문 유형과 난이도 커버
- 고품질의 정답 보장

### 데이터셋 관리

데이터셋을 업데이트할 때:
1. `metadata.json`의 `dataset_id`를 증가시킵니다 (예: `v1.0` → `v1.1`)
2. `created_at` 타임스탬프를 업데이트합니다.
3. 품질 보고서를 재생성합니다.

## 라이선스

이 데이터셋은 regulation_manager 프로젝트의 일부로, 프로젝트의 라이선스를 따릅니다.

## 연락처

데이터셋 관련 문의사항은 프로젝트 이슈 트래커를 이용해 주세요.
