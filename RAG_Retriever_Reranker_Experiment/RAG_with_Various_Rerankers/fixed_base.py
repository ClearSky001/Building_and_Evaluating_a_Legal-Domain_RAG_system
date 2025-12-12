"""
모든 리랭커를 위한 완전히 수정된 기본 클래스들
LangChain 호환성 문제를 완전히 해결한 독립적인 구현
"""
import os
import json
import numpy as np
from typing import List, Optional, Any
import re
from abc import ABC, abstractmethod
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


class SentenceTransformerEmbeddings(Embeddings):
    """공통 임베딩 클래스"""
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large-instruct"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        query_text = f"query: {text}"
        embedding = self.model.encode(query_text)
        return embedding.tolist()


class NaiveVectorStore(VectorStore):
    """완전히 수정된 NaiveVectorStore"""
    def __init__(self, documents: List[Document], embeddings: List[List[float]], embedding_function: Embeddings):
        self.documents = documents
        self._embeddings_matrix = np.array(embeddings, dtype=np.float32)
        self.embedding_function = embedding_function
        self._embeddings_matrix = self._embeddings_matrix / np.linalg.norm(self._embeddings_matrix, axis=1, keepdims=True)

    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None, **kwargs) -> List[str]:
        """추상 메서드 구현"""
        raise NotImplementedError("add_texts는 현재 구현되지 않았습니다.")

    def similarity_search_by_vector(self, embedding: List[float], k: int = 4, **kwargs) -> List[Document]:
        query_vector = np.array(embedding, dtype=np.float32)
        query_norm = query_vector / np.linalg.norm(query_vector)
        similarities = np.dot(self._embeddings_matrix, query_norm)
        top_k_indices = similarities.argsort()[::-1][:k]
        return [self.documents[i] for i in top_k_indices]

    def similarity_search(self, query: str, k: int = 4, **kwargs) -> List[Document]:
        query_embedding = self.embedding_function.embed_query(query)
        return self.similarity_search_by_vector(query_embedding, k, **kwargs)

    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs):
        query_embedding = self.embedding_function.embed_query(query)
        query_vector = np.array(query_embedding, dtype=np.float32)
        query_norm = query_vector / np.linalg.norm(query_vector)
        similarities = np.dot(self._embeddings_matrix, query_norm)
        top_k_indices = similarities.argsort()[::-1][:k]
        results = []
        for idx in top_k_indices:
            doc = self.documents[idx]
            score = float(similarities[idx])
            results.append((doc, score))
        return results

    def as_retriever(self, search_kwargs: Optional[dict] = None, **kwargs):
        """VectorStore를 Retriever로 변환 - 완전히 독립적인 구현"""
        search_kwargs = search_kwargs or {}
        
        class SimpleRetriever:
            """간단한 Retriever 구현"""
            def __init__(self, vectorstore, search_kwargs):
                self.vectorstore = vectorstore
                self.search_kwargs = search_kwargs
            
            def _get_relevant_documents(self, query: str) -> List[Document]:
                k = self.search_kwargs.get("k", 4)
                return self.vectorstore.similarity_search(query, k=k)
            
            def invoke(self, query: str) -> List[Document]:
                return self._get_relevant_documents(query)
        
        return SimpleRetriever(self, search_kwargs)

    @classmethod
    def from_texts(cls, texts: List[str], embedding: Embeddings, metadatas: Optional[List[dict]] = None, **kwargs):
        """추상 메서드 구현"""
        raise NotImplementedError("from_texts는 현재 구현되지 않았습니다.")


class BaseDocumentCompressor(ABC):
    """완전히 독립적인 BaseDocumentCompressor"""
    
    def __init__(self):
        pass
    
    @abstractmethod
    def compress_documents(
        self,
        documents: List[Document],
        query: str,
        callbacks: Optional[Any] = None,
    ) -> List[Document]:
        """문서 압축/리랭킹 메서드"""
        pass


class SimpleCompressionRetriever:
    """간단한 압축 리트리버 - ContextualCompressionRetriever 완전 대체"""
    
    def __init__(self, base_retriever, compressor):
        self.base_retriever = base_retriever
        self.compressor = compressor
    
    def invoke(self, query: str) -> List[Document]:
        """문서 검색 및 리랭킹"""
        # 기본 검색
        documents = self.base_retriever._get_relevant_documents(query)
        
        # 리랭킹
        if self.compressor and documents:
            documents = self.compressor.compress_documents(documents, query)
        
        return documents


# 공통 유틸리티 함수들
def _tokenize_ko(text: str) -> List[str]:
    """간단한 한국어 토크나이저"""
    return re.findall(r"[\w가-힣]+", text.lower())


def load_embeddings_data(embeddings_file: str) -> tuple:
    """임베딩 데이터 로드 공통 함수"""
    print(f"📂 임베딩 데이터 로드 중: {embeddings_file}")
    with open(embeddings_file, "r", encoding="utf-8") as f:
        chunk_data = json.load(f)
    
    documents = []
    embeddings_array = []
    
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
    
    print(f"✅ {len(documents)}개의 문서 청크 로드 완료")
    print(f"📄 첫 번째 청크 미리보기: {documents[0].page_content[:100]}...")
    
    return documents, embeddings_array


def setup_environment():
    """환경 설정 공통 함수"""
    load_dotenv()
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "")
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
    print("✅ 환경 설정 완료")


def create_legal_prompt():
    """법률 전문 프롬프트 생성"""
    return PromptTemplate.from_template(
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


def format_docs(docs):
    """문서 포맷팅 공통 함수"""
    formatted_docs = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', '알 수 없음')
        content = doc.page_content.strip()
        formatted_docs.append(f"📄 **문서 {i}** ({source})\n{content}")
    return "\n\n" + "\n\n".join(formatted_docs) + "\n\n"


def get_embeddings_file_path(current_file_path: str, embeddings_file: str = "output_chunks_with_embeddings.json") -> str:
    """임베딩 파일 경로를 올바르게 설정"""
    if os.path.isabs(embeddings_file):
        return embeddings_file
    
    # 현재 파일의 디렉토리에서 상위로 2단계 올라가서 임베딩 파일 찾기
    script_dir = os.path.dirname(os.path.abspath(current_file_path))
    embeddings_path = os.path.join(script_dir, "..", "..", "output_chunks_with_embeddings.json")
    embeddings_path = os.path.normpath(embeddings_path)
    
    return embeddings_path


# BM25 관련 클래스들
class BM25Reranker(BaseDocumentCompressor):
    """독립적인 BM25 리랭커"""
    def __init__(self, top_n: int = 5):
        super().__init__()
        self.top_n = top_n

    def compress_documents(self, documents: List[Document], query: str, callbacks=None) -> List[Document]:
        if not documents:
            return []
        
        corpus_tokens = [_tokenize_ko(doc.page_content) for doc in documents]
        bm25 = BM25Okapi(corpus_tokens)
        query_tokens = _tokenize_ko(query)
        scores = bm25.get_scores(query_tokens)
        ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:self.top_n]]


# CrossEncoder 관련 클래스들
class SentenceTransformerRerank(BaseDocumentCompressor):
    """독립적인 SentenceTransformerRerank"""
    
    def __init__(self, model_name: str, top_n: int = 10):
        super().__init__()
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
        except ImportError:
            raise ImportError("sentence_transformers가 필요합니다: pip install sentence-transformers")
        self.top_n = top_n
    
    def compress_documents(
        self,
        documents: List[Document],
        query: str,
        callbacks: Optional[Any] = None,
    ) -> List[Document]:
        if not documents:
            return []
        
        # 쿼리-문서 쌍으로 점수 계산
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs)
        
        # 점수 순으로 정렬
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 N개 반환
        return [doc for doc, score in doc_scores[:self.top_n]]


# Embedding 관련 클래스들
class EmbeddingCosineCompressor(BaseDocumentCompressor):
    """독립적인 EmbeddingCosineCompressor"""
    def __init__(self, top_n: int = 5, embed_model_name: str = "intfloat/multilingual-e5-large-instruct"):
        super().__init__()
        self.top_n = top_n
        self.embed = SentenceTransformerEmbeddings(embed_model_name)

    def compress_documents(self, documents: List[Document], query: str, callbacks=None) -> List[Document]:
        if not documents:
            return []
        q = np.array(self.embed.embed_query(query), dtype=np.float32)
        ds = np.array(self.embed.embed_documents([d.page_content for d in documents]), dtype=np.float32)
        sims = np.dot(ds, q)
        order = sims.argsort()[::-1][:self.top_n]
        return [documents[i] for i in order]
