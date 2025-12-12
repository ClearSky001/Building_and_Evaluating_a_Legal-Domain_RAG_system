# 리트리버 실험 가이드

## 개요
RAG_with_Retriever.py는 다양한 리트리버를 실험할 수 있는 시스템입니다.

## 지원하는 리트리버 종류

### 1. naive_cosine_e5 (기본값)
- Embedding 기반 dense retrieval
- multilingual-e5-large-instruct 모델 사용
- Cosine similarity로 유사도 계산

### 2. bm25
- 키워드 기반 sparse retrieval
- TF-IDF 스타일의 확률적 랭킹
- 파라미터: k1=1.5, b=0.75 (조정 가능)

### 3. tfidf
- TF-IDF 벡터화 기반 retrieval
- Unigram + Bigram 사용
- Scikit-learn 기반 구현

### 4. hybrid_rrf
- Dense + Sparse 하이브리드
- Reciprocal Rank Fusion으로 결합
- 가중치: [0.5, 0.5] (조정 가능)

## 사용법

### 기본 실행
```bash
python RAG_with_Retriever.py --retriever_id naive_cosine_e5
```

### BM25 실행
```bash
python RAG_with_Retriever.py --retriever_id bm25 --bm25_k1 1.5 --bm25_b 0.75
```

### 하이브리드 실행
```bash
python RAG_with_Retriever.py --retriever_id hybrid_rrf --hybrid_weights 0.6 0.4
```

### 전체 비교 실험
```bash
python test_retrievers.py
```

## 주요 파라미터

- `--k_ctx`: 생성에 사용할 문서 수 (기본값: 5)
- `--k_in`: 리랭커용 후보군 수 (기본값: 50)
- `--k_dbg`: 디버깅용 출력 수 (기본값: 10)
- `--seed`: 랜덤 시드 (기본값: 42)
- `--exp_name`: 실험 이름

## 출력 파일

모든 결과는 `exp_outputs/` 디렉토리에 저장됩니다:

1. **{exp_name}_config.json**: 실험 설정
2. **retriever_report.csv**: 실험 결과 요약
3. **cands_{retriever_id}_{index_version}.jsonl**: 후보군 덤프

## 실험 메타데이터

각 실험마다 자동으로 기록되는 정보:
- timestamp
- retriever_id
- embedding_model
- k 값들
- latency
- top1_score
- seed

## 예시: 리트리버별 특성

| 리트리버 | 장점 | 단점 | 사용 시나리오 |
|----------|------|------|---------------|
| naive_cosine_e5 | 의미적 유사도 높음 | 키워드 매칭 약함 | 일반 질문 |
| bm25 | 정확한 키워드 매칭 | 의미 이해 부족 | 법조문 검색 |
| tfidf | 빠른 속도 | 의미 이해 부족 | 대량 문서 |
| hybrid_rrf | 균형잡힌 성능 | 복잡도 높음 | 종합적 검색 |

