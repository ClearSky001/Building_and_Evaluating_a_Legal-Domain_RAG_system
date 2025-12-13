# 🤖 Legal-Domain RAG System: Building & Evaluating

A Legal-Domain Retrieval-Augmented Generation (RAG) System with Comprehensive Retriever & Reranker Evaluation for Enhanced Legal Information Retrieval

## Highlights

- **법률 도메인 전용 RAG 시스템 구축**
  - 법률 문서 검색 및 생성(RAG) 파이프라인 전반 구현
- **다양한 Retriever & Reranker 비교 평가**
  - BM25, Dense, TF-IDF 등 세 종류의 retriever 비교
  - BM25, Cohere, Hybrid, LLM(GPT-4o 기반), Rule reranker 등 reranker들의 성능 분석
-  **정량 평가 자동화 + 시각화 제공**
   - Top-k 성능 지표(Retrieval / Rerank) 자동 측정
   - 시각적 성능 비교 도표 포함
-  **연구/재현 가능 코드 + 실험 스크립트 포함**
   - 환경 설정, 데이터 로딩, 모델 실행까지 재현 가능
-  **학술적 기여**
   - 법률 RAG 분야 실험적 비교 및 분석
   - [관련 학회/저널 발표(논문)](https://koreascience.kr/article/CFKO202532532432735.page)

---

## 프로젝트 소개 & 주요 성과 요약
법무·세무 자문에 필요한 **부동산세 판례 문서 RAG 시스템**을 처음부터 구축하고, 다양한 리트리버·리랭커를 실험하며 정량적으로 비교한 프로젝트입니다. `Case Data Crawling`에서 직접 수집한 HTML 판례를 정제해 `output_chunks_with_embeddings.json`으로 임베딩을 만들고, `Naive_RAG` → `Retriever_Experiment` → `RAG_Retriever_Reranker_Experiment` 단계로 RAG 파이프라인을 설계하였습니다.  
- **데이터 파이프라인**: Selenium/BeautifulSoup 크롤러(`[한이음]크롤러.ipynb`)와 `Law_Data_Collecting_Process.ipynb`를 통해 판례 목록·원문·QA 페어를 구축했습니다.  
- **법무 도메인 RAG 시스템**: `Improved_Basic_RAG.py`와 `RAG_with_Retriever.py`는 LangChain, `multilingual-e5-large-instruct`, GPT-4o 등을 엮은 인터랙티브 RAG 레퍼런스입니다.  
- **Retriever/Reranker 풀 파이프라인**: Dense, BM25, TF-IDF, Hybrid RRF, Cross-Encoder, Embedding, LLM, Rule 기반 등 20여 종 리랭커를 스크립트(`comprehensive_test.py`, `test_final_rerankers.py`) 하나로 검증할 수 있게 했습니다.  
- **정량 평가 자동화**: `Q-A Data for Ragas Evaluation/real_estate_tax_QA.json`과 RAGAS 노트북/스크립트를 이용해 context precision·faithfulness 등을 반복 측정 후 시각화(`ragas_bar_charts.png` 등)했습니다.  
- **논문화·발표**: 최종 실험 결과는 `Accepted Paper & Presentation/`에 수록된 논문·발표 자료로 제출되어 대회·학회 공유까지 연결되었습니다.

---

## 프로젝트 배경
- 부동산 공시가격, 증여·양도세 등 복잡한 법령·지침은 최신 판례와 함께 조회되어야 하지만, 공개 말뭉치가 적고 보안상 외부 서비스 사용이 제한적입니다.  
- 법무사·세무사가 실제로 묻는 질의 형태(설명형, 계산형, 절차형)를 반영한 **자체 QA 데이터**를 만들어야 했고, 한국어 기반 고품질 검색/생성 체인을 설계해야 했습니다.  
- LLM 응답 신뢰성을 확보하기 위해 단순 BLEU나 EM 대신 **RAGAS**와 같은 맥락 기반 지표가 필요했습니다.

---

## 기술 스택
- **언어**: Python 3.10+
- **LLM & 오케스트레이션**: LangChain, LangChain Core/Community, LangChain OpenAI, LangSmith, dotenv
- **임베딩 & 검색**: SentenceTransformers (`intfloat/multilingual-e5-large-instruct`, GTE, MPNet, Jina, Cohere 등), BM25 (`rank-bm25`, `langchain_community.retrievers.BM25Retriever`), TF-IDF(`scikit-learn`), Hybrid RRF, EnsembleRetriever
- **리랭커**: SentenceTransformer CrossEncoder, Cohere Rerank API, LLM 기반 Pairwise/Listwise reranker, 규칙 기반 LegalRuleBoost 등 (`RAG_with_Various_Rerankers/`)
- **데이터/시각화**: Selenium, BeautifulSoup4, pandas, numpy, matplotlib, ragas, datasets(HF), Plotly/Seaborn(notebooks)
- **기타**: python-dotenv, tqdm, argparse, csv/json 유틸, chromedriver

---

## 데이터 수집 & 전처리
1. **판례 수집 (`Case Data Crawling/`)**  
   - `판례_목록.csv`, `merged_cases.html`: 국세청/대법원 등 공개 포털에서 수집한 판례 메타·본문.  
   - `판례_크롤링_결과/BeautifulSoup_결과`, `.../Selenium_결과`: 정적·동적 페이지 모두 커버하도록 두 파이프라인으로 나눴습니다.  
   - `chromedriver.exe`와 `[한이음]크롤러.ipynb`가 전체 절차(검색 → 상세 진입 → HTML 저장)를 자동화합니다.
2. **데이터 이슈 & 클리닝 노트 (`Data Collecting Process/`)**  
   - `Law_Data_Collecting_Process.ipynb`와 `법무자문 프로젝트 데이터 수집에서의 이슈/`는 빠진 필드, 중복 판례, HTML 캐릭터 이슈 등을 정리합니다.
3. **QA 세트 구성 (`Q-A Data for Ragas Evaluation/real_estate_tax_QA.json`)**  
   - `question`, `ground_truth`, `ground_truth_contexts`, `metadata(case_number, topic)` 필드를 갖춘 40개 이상의 레퍼런스 페어.  
   - RAGAS, 리랭커 실험, 문서 요약 테스트에 재사용됩니다.
4. **임베딩 빌드 (`Naive_RAG/output_chunks_with_embeddings.json`, 약 34MB)**  
   - 판례 HTML을 청크화 후 `SentenceTransformerEmbeddings` 클래스로 임베딩 → 문서와 chunk_index, filename, embedding을 모두 저장.  
   - `Basic_RAG_with_HTML_JSON_File_Based.ipynb`에서 추출 파라미터와 품질 로그를 확인할 수 있습니다.

---

## 파이프라인 & 실험 시나리오
1. **Naive RAG 베이스 (`Naive_RAG/`)**  
   - `Improved_Basic_RAG.py`: 순수 Python 모듈에서 `.json` 임베딩을 읽고, 맞춤형 `NaiveVectorStore` + `ChatOpenAI(gpt-4o-mini)` 체인을 구성합니다.  
   - `RAG_with_Retriever.py`: `ExperimentConfig` dataclass로 retriever 타입, k 값, 로그 경로, LangSmith 옵션을 CLI에서 제어할 수 있습니다.  
   - `test_retrievers.py`: `naive_cosine_e5`, `bm25`, `tfidf`, `hybrid_rrf` 등을 순차로 실행해 `exp_outputs/`에 결과를 모읍니다.
2. **Retriever 비교 (`Retriever_Experiment/`)**  
   - `Retriever eval.py`: HF `datasets`, `ragas` 기반으로 사용자 정의 retriever 콜백을 평가합니다.  
   - `Retriever.ipynb`: 실험 노트북 버전으로, Stratified 샘플과 지표(precision/recall)를 시각화합니다.
3. **Reranker & 압축리트리버 실험 (`RAG_Retriever_Reranker_Experiment/`)**  
   - `RAG_with_Various_Rerankers/`: `fixed_base_v2.py`, `BaseReranker`, `SimpleCompressionRetriever`를 베이스로 하여 BM25, CrossEncoder, Embedding, Hybrid, LLM, Rule 계열 서브 디렉터리로 분기.  
   - `comprehensive_test.py`, `test_final_rerankers.py`, `final_test.py`: 모든/파이널 리랭커 로더 자동 검증.  
   - `RAGAS_Full_Evaluation.ipynb`, `RAGAS_Reranker_Performance_Comparison.ipynb`, `Visualize_RAGAS_results.ipynb`: GPU 버전 평가, 결과 대시보드, heatmap/line/radar chart 이미지를 생성합니다.
4. **정량 평가 & 리포트 (`Reranker_RAGAS_result/`, `ragas_*.png`)**  
   - 리랭커별 `*_ragas_evaluation_*.csv` + 통합 `Reranker_RAGAS_Comparison.csv`(xlsx 포맷) + 순위표(`RAGAS_Rankings_*.csv`).  
   - `RAGAS_Final_Results_20250917_173542.csv`는 Cohere Rerank가 overall_score 0.9644로 최고임을 보여줍니다.
5. **결과 공유 (`Accepted Paper & Presentation/`)**  
   - `Building and Evaluating a Legal-Domain RAG system...pdf`, `Paper_Presentation.pdf`에서 연구 배경, 파이프라인 다이어그램, RAGAS 결과를 확인할 수 있습니다.

---

## 사용법 (빠른 시작)
1. **환경 준비**
   ```bash
   cd Building_and_Evaluating_a_Legal-Domain_RAG_system
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -U langchain langchain-openai langchain-community sentence-transformers rank-bm25 ragas datasets pandas numpy scikit-learn python-dotenv matplotlib seaborn plotly selenium beautifulsoup4 tqdm
   ```
2. **환경 변수 설정 (`.env`)**
   ```ini
   OPENAI_API_KEY=sk-...
   LANGSMITH_ENDPOINT=https://api.smith.langchain.com
   LANGSMITH_API_KEY=lsv2-...
   ```
3. **(선택) 임베딩 갱신**  
   - `Naive_RAG/Basic_RAG_with_HTML_JSON_File_Based.ipynb` or `Improved_Basic_RAG.py`에서 `output_chunks_with_embeddings.json`을 재생성합니다.
4. **베이스라인 RAG 실행**
   ```bash
   cd Naive_RAG
   python RAG_with_Retriever.py --retriever_id naive_cosine_e5 --k_ctx 6
   # BM25
   python RAG_with_Retriever.py --retriever_id bm25 --bm25_k1 1.2 --bm25_b 0.7
   # Hybrid RRF
   python RAG_with_Retriever.py --retriever_id hybrid_rrf --hybrid_weights 0.6 0.4
   # 일괄 실험
   python test_retrievers.py
   ```
5. **리랭커 벤치마크**
   ```bash
   cd ..\RAG_Retriever_Reranker_Experiment
   python comprehensive_test.py          # 모든 모델 import/초기화 테스트
   python test_final_rerankers.py        # FINAL 모델 (BM25, CrossEncoder, Embedding)
   python final_test.py                  # 경량 smoke test
   python simple_test.py                 # 단일 쿼리 디버그
   python test_imports.py                # 패키지 의존성 확인
   ```
6. **RAGAS 평가 & 시각화**
   ```bash
   # retriever callable을 평가 (예: Retriever_Experiment/Retriever eval.py)
   python Retriever_Experiment/"Retriever eval.py"
   # 또는 노트북 실행
   jupyter notebook RAG_Retriever_Reranker_Experiment/RAGAS_Full_Evaluation.ipynb
   ```
7. **결과 확인**  
   - `RAG_Retriever_Reranker_Experiment/ragas_*_charts.png`, `RAGAS_Rankings_*.csv`, `exp_outputs/`.  
   - 논문/발표는 `Accepted Paper & Presentation/` 참고.

---

## 실험 결과 요약
`RAGAS_Final_Results_20250917_173542.csv` 기준 상위 리랭커 성능은 다음과 같습니다.

| Reranker                | context_precision | context_recall | faithfulness | answer_relevancy | overall_score |
|------------------------|------------------:|---------------:|-------------:|-----------------:|--------------:|
| Cohere Rerank          | **0.9750**        | 1.0000         | **1.0000**   | 0.8827           | **0.9644**    |
| LLM(gpt-4o Reranker)   | 0.6428            | 1.0000         | **1.0000**   | 0.8827           | 0.8814        |
| Legal Rule Boost       | 0.5566            | 1.0000         | **1.0000**   | 0.8827           | 0.8598        |
| Hybrid CombSum         | **0.9765**        | 1.0000         | 0.0000       | 0.8827           | 0.7148        |
| BM25                   | 0.6211            | 1.0000         | 0.3333       | **0.8827**       | 0.7093        |

- **Context Precision 1위**는 Hybrid CombSum이지만 Faithfulness 0으로, Cohere Rerank가 가장 안정적인 균형을 보여줍니다.  
- `ragas_radar_charts.png`, `ragas_heatmaps.png`, `ragas_line_charts.png`은 지표 변화를 한눈에 보여줍니다.  
- `Reranker_RAGAS_Comparison.csv`에는 reranker/metric별 순위와 raw score가 모두 포함됩니다.

---

## 디렉터리 가이드
- `Case Data Crawling/`: 크롤러 노트북, chromedriver, 판례 HTML/CSV.  
- `Data Collecting Process/`: 수집 이슈 정리 노트와 보완 자료.  
- `Naive_RAG/`: 기본 RAG 코드, 실험 출력(`exp_outputs/`), retriever 가이드/테스트 스크립트.  
- `Retriever_Experiment/`: retriever 평가 노트북 & `Retriever eval.py`.  
- `Q-A Data for Ragas Evaluation/`: 평가용 QA 레퍼런스 JSON.  
- `RAG_Retriever_Reranker_Experiment/`: 리랭커 모듈, 테스트 스크립트, RAGAS 노트북·시각화, 결과 CSV/PNG.  
- `Accepted Paper & Presentation/`: 제출된 논문 PDF와 발표 자료.  
- `Retriever_Experiment/Retriever.ipynb`, `RAG_Retriever_Reranker_Experiment/Reranker.ipynb`: 실험용 Jupyter 기반 워크플로.

---

## 향후 계획
- 판례·행정해석을 연도/세목 기준으로 자동 태깅하여 **다중 벡터store + 필터링** 전략 실험.  
- `output_chunks_with_embeddings.json`을 LangChain `FAISS` 또는 `Chroma` 저장소로 변환해 온라인 업데이트를 지원.  
- RAGAS 외에 **Answer Similarity**, **Human Preference Logging**을 도입하여 리랭커 튜닝.  
-법령 개정 추적을 위해 `Case Data Crawling` 크론 잡/CI로 자동화.

---

## 개인 기여 & 배운 점
- **데이터 엔지니어링**: 크롤링부터 QA 태깅, 임베딩 생성까지 전처리 파이프라인을 직접 설계·자동화.  
- **LLM RAG 아키텍처링**: LangChain 추상화를 활용해 retriever/retrieval-augmented generation 모듈을 독립적으로 교체 가능하도록 구조화.  
- **실험 자동화**: CLI 스크립트(예: `test_retrievers.py`, `comprehensive_test.py`)로 수십 가지 설정을 반복 실행하고, 결과 JSONL·CSV를 자동 수집하도록 구축.  
- **정량 평가**: RAGAS/시각화 코드를 직접 작성해 reranker 선택 의사결정에 활용했습니다.

---

## 참고 자료
- `Accepted Paper & Presentation/Building and Evaluating a Legal-Domain RAG system_ A Comparative Study of Retrievers and Rerankers in the Real Estate Tax field.pdf`
- `Accepted Paper & Presentation/Paper_Presentation.pdf`
- `RAG_Retriever_Reranker_Experiment/RAGAS_Full_Evaluation.ipynb`
- `RAG_Retriever_Reranker_Experiment/ragas_bar_charts.png` 등 시각화 자산
