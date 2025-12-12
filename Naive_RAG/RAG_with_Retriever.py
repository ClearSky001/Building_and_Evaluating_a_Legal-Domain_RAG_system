# -*- coding: utf-8 -*-
import os
import sys
import json
import time
import random
import argparse
import csv
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np
from dotenv import load_dotenv

# LangChain
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings

# SentenceTransformer
from sentence_transformers import SentenceTransformer

# Additional imports for retriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# 0) Experiment Config & Utils
# -----------------------------

@dataclass
class ExperimentConfig:
    # data/index
    embeddings_file: str = "output_chunks_with_embeddings.json"
    index_version: str = "v2025-08-16"
    retriever_id: str = "naive_cosine_e5"   # "naive_cosine_e5", "bm25", "tfidf", "hybrid_rrf"
    distance_metric: str = "cosine"
    embedding_model: str = "intfloat/multilingual-e5-large-instruct"
    
    # BM25 parameters
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    
    # Hybrid retriever weights
    hybrid_weights: List[float] = field(default_factory=lambda: [0.5, 0.5])  # Equal weights for dense/sparse

    # k-values
    k_ctx: int = 5         # generation에 들어갈 컨텍스트 수
    k_in: int = 50         # 리랭커용 후보군 덤프 수
    k_dbg: int = 10        # 디버깅/프린트용 조회 수

    # LLM
    llm_model: str = "gpt-4o-mini"
    temperature: float = 0.0

    # seeds
    seed: int = 42

    # logging
    out_dir: str = "exp_outputs"
    exp_name: str = "retriever_baseline"

    # langsmith (optional)
    tracing_v2: str = "true"
    langsmith_endpoint: str = ""
    langsmith_api_key: str = ""


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True  # (옵션) 성능 저하 가능
        # torch.backends.cudnn.benchmark = False     # (옵션)
    except Exception:
        pass


def now_kst_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# -----------------------------
# 1) Embeddings Wrapper
# -----------------------------

class SentenceTransformerEmbeddings(Embeddings):
    """SentenceTransformer를 LangChain Embeddings 인터페이스로 래핑"""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-large-instruct", seed: int = 42):
        # deterministic을 최대한 유지하려고 seed를 앞서 설정
        # SentenceTransformer 자체는 추론 시 비결정성이 거의 없지만, 안전하게 시드 고정
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=False, convert_to_numpy=True).tolist()
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        # E5 모델은 쿼리에 "query: " 접두사 권장
        query_text = f"query: {text}"
        embedding = self.model.encode([query_text], normalize_embeddings=False, convert_to_numpy=True)[0].tolist()
        return embedding


# -----------------------------
# 2) Naive VectorStore (cosine)
# -----------------------------

class NaiveVectorStore(VectorStore):
    """Naive VectorStore - LangChain 호환 버전 (cosine 유사도)"""

    def __init__(self, documents: List[Document], embeddings: List[List[float]], embedding_function: Embeddings):
        self.documents = documents
        self.embedding_function = embedding_function
        mat = np.array(embeddings, dtype=np.float32)

        # 0 division 방지
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        self._embeddings_matrix = mat / norms

    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None, **kwargs) -> List[str]:
        raise NotImplementedError("add_texts는 현재 구현되지 않았습니다.")

    def similarity_search_by_vector(self, embedding: List[float], k: int = 4, **kwargs) -> List[Document]:
        query_vector = np.array(embedding, dtype=np.float32)
        qnorm = np.linalg.norm(query_vector)
        if qnorm == 0:
            qnorm = 1e-12
        query_norm = query_vector / qnorm

        sims = np.dot(self._embeddings_matrix, query_norm)
        # 상위 k개
        top_k_idx = sims.argsort()[::-1][:k]
        return [self.documents[i] for i in top_k_idx]

    def similarity_search(self, query: str, k: int = 4, **kwargs) -> List[Document]:
        qemb = self.embedding_function.embed_query(query)
        return self.similarity_search_by_vector(qemb, k, **kwargs)

    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs) -> List[tuple]:
        qemb = self.embedding_function.embed_query(query)
        query_vector = np.array(qemb, dtype=np.float32)
        qnorm = np.linalg.norm(query_vector)
        if qnorm == 0:
            qnorm = 1e-12
        query_norm = query_vector / qnorm

        sims = np.dot(self._embeddings_matrix, query_norm)
        top_k_idx = sims.argsort()[::-1][:k]

        results = []
        for idx in top_k_idx:
            results.append((self.documents[idx], float(sims[idx])))
        return results

    @classmethod
    def from_texts(cls, texts: List[str], embedding: Embeddings, metadatas: Optional[List[dict]] = None, **kwargs):
        raise NotImplementedError("from_texts는 현재 구현되지 않았습니다.")


# -----------------------------
# 2.5) TF-IDF VectorStore
# -----------------------------

class TFIDFVectorStore(VectorStore):
    """TF-IDF 기반 VectorStore"""
    
    def __init__(self, documents: List[Document], embedding_function: Embeddings = None):
        self.documents = documents
        self.embedding_function = embedding_function  # TF-IDF는 사용하지 않지만 인터페이스 호환성을 위해
        
        # TF-IDF 벡터화
        texts = [doc.page_content for doc in documents]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),  # unigram + bigram
            max_features=10000,
            min_df=2,
            max_df=0.95
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None, **kwargs) -> List[str]:
        raise NotImplementedError("add_texts는 현재 구현되지 않았습니다.")
    
    def similarity_search_by_vector(self, embedding: List[float], k: int = 4, **kwargs) -> List[Document]:
        raise NotImplementedError("TF-IDF는 벡터 검색을 지원하지 않습니다.")
    
    def similarity_search(self, query: str, k: int = 4, **kwargs) -> List[Document]:
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_k_idx = similarities.argsort()[::-1][:k]
        return [self.documents[i] for i in top_k_idx]
    
    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs) -> List[tuple]:
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_k_idx = similarities.argsort()[::-1][:k]
        
        results = []
        for idx in top_k_idx:
            results.append((self.documents[idx], float(similarities[idx])))
        return results
    
    @classmethod
    def from_texts(cls, texts: List[str], embedding: Embeddings, metadatas: Optional[List[dict]] = None, **kwargs):
        raise NotImplementedError("from_texts는 현재 구현되지 않았습니다.")


# -----------------------------
# 3) RAG System
# -----------------------------

class LegalRAGSystem:
    """법률 문서 RAG 시스템"""

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.documents: List[Document] = []
        self.vectorstore: Optional[NaiveVectorStore] = None
        self.retriever = None
        self.rag_chain = None
        self.llm = None
        self.embedding_model: Optional[SentenceTransformerEmbeddings] = None

        # 환경
        self._setup_environment()
        # seed
        set_seed(self.cfg.seed)
        # 초기화
        self._initialize_system()

    # --- env & IO ---

    def _setup_environment(self):
        load_dotenv()
        os.environ["LANGCHAIN_TRACING_V2"] = self.cfg.tracing_v2
        if self.cfg.langsmith_endpoint:
            os.environ["LANGCHAIN_ENDPOINT"] = self.cfg.langsmith_endpoint
        if self.cfg.langsmith_api_key:
            os.environ["LANGCHAIN_API_KEY"] = self.cfg.langsmith_api_key

        ensure_dir(self.cfg.out_dir)
        # 실험 메타 저장 - 타임스탬프와 retriever_id 포함
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        meta_filename = f"{self.cfg.exp_name}_{self.cfg.retriever_id}_{timestamp}_config.json"
        meta_path = os.path.join(self.cfg.out_dir, meta_filename)
        save_json(asdict(self.cfg), meta_path)
        print(f"✅ 환경/설정 준비 완료, config 저장: {meta_path}")

    def _load_embeddings_data(self) -> tuple:
        print(f"📂 임베딩 데이터 로드: {self.cfg.embeddings_file}")
        with open(self.cfg.embeddings_file, "r", encoding="utf-8") as f:
            chunk_data = json.load(f)

        documents, embeddings_array = [], []
        for item in chunk_data:
            doc = Document(
                page_content=item["text"],
                metadata={
                    "filename": item["filename"],
                    "chunk_index": item["chunk_index"],
                    "source": f"{item['filename']}_chunk_{item['chunk_index']}"
                }
            )
            documents.append(doc)
            embeddings_array.append(item["embedding"])

        print(f"✅ 문서 청크: {len(documents)}개")
        if documents:
            print(f"📄 첫 청크: {documents[0].page_content[:100]}...")
        return documents, embeddings_array

    def _create_vectorstore(self, documents: List[Document], embeddings_array: List[List[float]]):
        print(f"🔧 VectorStore 생성 (type: {self.cfg.retriever_id})...")
        self.embedding_model = SentenceTransformerEmbeddings(model_name=self.cfg.embedding_model, seed=self.cfg.seed)
        
        if self.cfg.retriever_id in ["naive_cosine_e5", "hybrid_rrf"]:
            vs = NaiveVectorStore(documents=documents, embeddings=embeddings_array, embedding_function=self.embedding_model)
        elif self.cfg.retriever_id == "tfidf":
            vs = TFIDFVectorStore(documents=documents, embedding_function=self.embedding_model)
        else:
            # BM25는 vectorstore가 아니므로 None 반환
            vs = None
            
        if vs:
            print(f"✅ VectorStore 준비(문서 수: {len(documents)})")
        return vs

    # --- RAG ---
    
    def _setup_retriever(self):
        """retriever_id에 따라 적절한 리트리버 설정"""
        print(f"🔧 Retriever 설정 중... (type: {self.cfg.retriever_id})")
        
        if self.cfg.retriever_id == "naive_cosine_e5":
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.cfg.k_ctx})
            
        elif self.cfg.retriever_id == "bm25":
            # BM25 Retriever
            texts = [doc.page_content for doc in self.documents]
            self.retriever = BM25Retriever.from_texts(
                texts=texts,
                metadatas=[doc.metadata for doc in self.documents],
                k=self.cfg.k_ctx,
                bm25_params={"k1": self.cfg.bm25_k1, "b": self.cfg.bm25_b}
            )
            
        elif self.cfg.retriever_id == "tfidf":
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.cfg.k_ctx})
            
        elif self.cfg.retriever_id == "hybrid_rrf":
            # Hybrid retriever using Reciprocal Rank Fusion
            # Dense retriever (embedding-based)
            dense_retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.cfg.k_ctx * 2})
            
            # Sparse retriever (BM25)
            texts = [doc.page_content for doc in self.documents]
            sparse_retriever = BM25Retriever.from_texts(
                texts=texts,
                metadatas=[doc.metadata for doc in self.documents],
                k=self.cfg.k_ctx * 2,
                bm25_params={"k1": self.cfg.bm25_k1, "b": self.cfg.bm25_b}
            )
            
            # Ensemble with RRF
            weights = self.cfg.hybrid_weights or [0.5, 0.5]
            self.retriever = EnsembleRetriever(
                retrievers=[dense_retriever, sparse_retriever],
                weights=weights,
                search_kwargs={"k": self.cfg.k_ctx}
            )
        else:
            raise ValueError(f"Unknown retriever_id: {self.cfg.retriever_id}")
            
        print(f"✅ Retriever 준비 완료: {self.cfg.retriever_id}")

    def _setup_rag_components(self):
        print("⚙️ RAG 컴포넌트 구성...")
        # Retriever는 이미 _setup_retriever()에서 설정됨

        legal_prompt = PromptTemplate.from_template(
            """당신은 부동산세법 전문가입니다. 다음 법률 문서를 참고하여 질문에 정확하고 상세하게 답변해주세요.

📋 **참고 문서:**
{context}

📝 **답변 지침:**
1. 반드시 제공된 문서 내용에 근거하여 답변하세요
2. 관련 법령 조문이나 조항을 명시해주세요
3. 법률적 근거를 구체적으로 제시해주세요
4. 문서에서 명확한 답을 찾을 수 없다면 "제공된 문서에서 해당 정보를 찾을 수 없습니다"라고 말하세요
5. 답변은 한국어로 작성하세요

❓ **질문:** {question}

💡 **답변:**"""
        )

        self.llm = ChatOpenAI(
            model=self.cfg.llm_model,
            temperature=self.cfg.temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

        def format_docs(docs):
            formatted = []
            for i, doc in enumerate(docs, 1):
                src = doc.metadata.get("source", "unknown")
                content = doc.page_content.strip()
                formatted.append(f"📄 **문서 {i}** ({src})\n{content}")
            return "\n\n" + "\n\n".join(formatted) + "\n\n"

        self.rag_chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | legal_prompt
            | self.llm
            | StrOutputParser()
        )
        print("✅ RAG 체인 구성 완료")

    def _initialize_system(self):
        print("🚀 시스템 초기화 시작")
        try:
            self.documents, emb = self._load_embeddings_data()
            self.vectorstore = self._create_vectorstore(self.documents, emb)
            self._setup_retriever()
            self._setup_rag_components()
            print("🎉 초기화 완료")
        except Exception as e:
            print(f"❌ 초기화 오류: {e}")
            raise

    # -----------------------------
    # 4) Utilities (dump/report)
    # -----------------------------

    def dump_candidates(self, query: str, k_in: Optional[int] = None, filename: Optional[str] = None):
        """리랭커 팀 공정비교용 후보군 덤프(JSONL)"""
        k = k_in if k_in is not None else self.cfg.k_in
        out_name = filename or f"cands_{self.cfg.retriever_id}_{self.cfg.index_version}.jsonl"
        path = os.path.join(self.cfg.out_dir, out_name)

        t0 = time.time()
        
        # retriever_id에 따라 다른 방식으로 후보군 생성
        if self.cfg.retriever_id == "bm25":
            # BM25는 score를 제공하지 않으므로 순위를 score로 사용
            docs = self.retriever.get_relevant_documents(query)[:k]
            results = [(doc, 1.0 / (i + 1)) for i, doc in enumerate(docs)]
        elif self.cfg.retriever_id == "hybrid_rrf":
            # Hybrid도 마찬가지로 순위 기반 스코어
            docs = self.retriever.get_relevant_documents(query)[:k]
            results = [(doc, 1.0 / (i + 1)) for i, doc in enumerate(docs)]
        else:
            # vectorstore 기반 리트리버들
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
        dt = time.time() - t0

        with open(path, "w", encoding="utf-8") as f:
            for rank, (doc, score) in enumerate(results, start=1):
                rec = {
                    "timestamp": now_kst_iso(),
                    "query": query,
                    "rank": rank,
                    "score": score,
                    "text": doc.page_content,
                    "source": doc.metadata.get("source"),
                    "retriever_id": self.cfg.retriever_id,
                    "index_version": self.cfg.index_version,
                    "distance_metric": self.cfg.distance_metric,
                    "embedding_model": self.cfg.embedding_model,
                    "k_in": k,
                    "seed": self.cfg.seed,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"🗂️ 후보군 {k}개 저장: {path} (latency: {dt:.3f}s)")
        return path, dt, results[0][1] if results else None

    def write_report_row(self, csv_name: str, row: Dict[str, Any]):
        path = os.path.join(self.cfg.out_dir, csv_name)
        new_file = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new_file:
                w.writeheader()
            w.writerow(row)
        return path

    # -----------------------------
    # 5) Interactive & Tests
    # -----------------------------

    def test_similarity_search(self, query: str = "종합부동산세의 목적", k: Optional[int] = None):
        k = k or self.cfg.k_dbg
        print(f"\n🔍 유사도 검색 테스트 | k={k} | query='{query}' | retriever={self.cfg.retriever_id}")
        
        if self.cfg.retriever_id == "bm25" or self.cfg.retriever_id == "hybrid_rrf":
            docs = self.retriever.get_relevant_documents(query)[:k]
        else:
            if self.vectorstore is None:
                print("❌ VectorStore 미초기화")
                return []
            docs = self.vectorstore.similarity_search(query, k=k)
            
        for i, d in enumerate(docs, 1):
            print(f"[{i}] {d.metadata.get('source')} :: {d.page_content[:150]}...")
        return docs

    def test_similarity_search_with_score(self, query: str = "종합부동산세의 목적", k: Optional[int] = None):
        k = k or self.cfg.k_dbg
        print(f"\n🔍 점수 포함 유사도 검색 | k={k} | query='{query}' | retriever={self.cfg.retriever_id}")
        
        if self.cfg.retriever_id == "bm25" or self.cfg.retriever_id == "hybrid_rrf":
            docs = self.retriever.get_relevant_documents(query)[:k]
            results = [(doc, 1.0 / (i + 1)) for i, doc in enumerate(docs)]
        else:
            if self.vectorstore is None:
                print("❌ VectorStore 미초기화")
                return []
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
        for i, (d, s) in enumerate(results, 1):
            print(f"[{i}] score={s:.4f} | {d.metadata.get('source')} :: {d.page_content[:150]}...")
        return results

    def ask_question(self, question: str, show_sources: bool = True) -> str:
        print(f"\n🤖 질문: {question}")
        try:
            if show_sources:
                docs = self.retriever.invoke(question)
                print("📚 참고 문서:")
                for i, d in enumerate(docs, 1):
                    print(f"  {i}. {d.metadata.get('source')}")
            resp = self.rag_chain.invoke(question)
            print("\n💡 답변:\n" + resp)
            return resp
        except Exception as e:
            msg = f"❌ 답변 오류: {e}"
            print(msg)
            return msg

    def run_test_questions(self):
        tests = [
            "종합부동산세법의 목적을 법령 조문을 근거로 설명해 주세요.",
            "종합부동산세 납세의무자는 누구인가요?",
            "종합부동산세 과세대상은 무엇인가요?",
        ]
        print("\n" + "="*60)
        print("📋 테스트 질문 실행")
        print("="*60)
        for i, q in enumerate(tests, 1):
            print("\n" + "="*60)
            print(f"테스트 {i}: {q}")
            print("="*60)
            self.ask_question(q)

# -----------------------------
# 6) CLI / Main
# -----------------------------

def build_config_from_args() -> ExperimentConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings_file", type=str, default="output_chunks_with_embeddings.json")
    p.add_argument("--index_version", type=str, default="v2025-08-16")
    p.add_argument("--retriever_id", type=str, default="naive_cosine_e5")
    p.add_argument("--embedding_model", type=str, default="intfloat/multilingual-e5-large-instruct")
    p.add_argument("--k_ctx", type=int, default=5)
    p.add_argument("--k_in", type=int, default=50)
    p.add_argument("--k_dbg", type=int, default=10)
    p.add_argument("--llm_model", type=str, default="gpt-4o-mini")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="exp_outputs")
    p.add_argument("--exp_name", type=str, default="retriever_baseline")
    p.add_argument("--bm25_k1", type=float, default=1.5)
    p.add_argument("--bm25_b", type=float, default=0.75)
    p.add_argument("--hybrid_weights", type=float, nargs="*", default=None)
    args = p.parse_args()

    cfg = ExperimentConfig(
        embeddings_file=args.embeddings_file,
        index_version=args.index_version,
        retriever_id=args.retriever_id,
        embedding_model=args.embedding_model,
        k_ctx=args.k_ctx,
        k_in=args.k_in,
        k_dbg=args.k_dbg,
        llm_model=args.llm_model,
        temperature=args.temperature,
        seed=args.seed,
        out_dir=args.out_dir,
        exp_name=args.exp_name,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
        hybrid_weights=args.hybrid_weights,
    )
    return cfg


def main():
    print("🚀 법률 문서 RAG 베이스라인 시작")
    cfg = build_config_from_args()
    print(f"▶ Config: {cfg}")

    try:
        rag = LegalRAGSystem(cfg)

        # 1) 디버깅용 Retrieval 확인 (k_dbg)
        rag.test_similarity_search()
        rag.test_similarity_search_with_score()

        # 2) 후보군 덤프 (k_in) — 리랭커팀 공통 입력으로 사용
        q = "종합부동산세 과세표준 및 세율에 대해 설명해 주세요."
        path, latency, top1 = rag.dump_candidates(q, k_in=cfg.k_in)
        # 간단 리포트 CSV
        row = {
            "timestamp": now_kst_iso(),
            "exp_name": cfg.exp_name,
            "retriever_id": cfg.retriever_id,
            "index_version": cfg.index_version,
            "embedding_model": cfg.embedding_model,
            "k_in": cfg.k_in,
            "k_ctx": cfg.k_ctx,
            "query": q,
            "latency_s": round(latency, 4),
            "top1_score": round(top1, 6) if top1 is not None else None,
            "seed": cfg.seed,
        }
        csv_path = rag.write_report_row("retriever_report.csv", row)
        print(f"🧾 리포트 업데이트: {csv_path}")

        # 3) 스모크 질문 실행 (k_ctx)
        rag.run_test_questions()

        # 4) 선택: 인터랙티브
        try:
            user_input = input("\n🤔 대화형 모드를 시작하시겠습니까? (y/n): ").strip().lower()
        except EOFError:
            user_input = "n"
        if user_input in ("y", "yes", "예", "ㅇ"):
            print("\n=== 대화형 모드 ===")
            while True:
                text = input("질문(종료:q): ").strip()
                if text.lower() in ("q", "quit", "exit", "종료"):
                    break
                rag.ask_question(text)

        print("\n✅ 종료")
    except FileNotFoundError:
        print("❌ 'output_chunks_with_embeddings.json' 파일을 찾을 수 없습니다. 경로를 확인하세요.")
    except Exception as e:
        print(f"❌ 예외 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
