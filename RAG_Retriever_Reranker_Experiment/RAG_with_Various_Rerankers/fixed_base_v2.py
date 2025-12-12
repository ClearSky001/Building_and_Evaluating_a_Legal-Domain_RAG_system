"""
새로운 인터페이스에 맞는 리랭커 기본 클래스들
표준화된 파라미터와 반환 형식 사용
"""
import os
import json
import numpy as np
from typing import List, Optional, Any, Dict
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


# 호환성을 위한 기존 클래스들
class BaseDocumentCompressor(ABC):
    """기존 호환성을 위한 BaseDocumentCompressor"""
    
    @abstractmethod
    def compress_documents(
        self,
        documents: List[Document],
        query: str,
        callbacks: Optional[Any] = None,
    ) -> List[Document]:
        """기존 호환성을 위한 메서드"""
        pass


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
        """VectorStore를 Retriever로 변환"""
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
            
            def get_candidate_documents(self, query: str) -> List[dict]:
                """새로운 인터페이스를 위한 후보 문서 반환"""
                docs = self._get_relevant_documents(query)
                candidate_docs = []
                for doc in docs:
                    candidate_docs.append({
                        'doc_id': doc.metadata.get('source', 'unknown'),
                        'chunk_index': doc.metadata.get('chunk_index', 0),
                        'filename': doc.metadata.get('filename', 'unknown'),
                        'text': doc.page_content,
                        'score': 1.0  # 기본 점수
                    })
                return candidate_docs
        
        return SimpleRetriever(self, search_kwargs)

    @classmethod
    def from_texts(cls, texts: List[str], embedding: Embeddings, metadatas: Optional[List[dict]] = None, **kwargs):
        """추상 메서드 구현"""
        raise NotImplementedError("from_texts는 현재 구현되지 않았습니다.")


class BaseReranker(ABC):
    """새로운 표준 리랭커 인터페이스"""
    
    def __init__(self, top_n: int = 10):
        self.top_n = top_n
    
    @abstractmethod
    def rerank_documents(
        self,
        query: str,
        candidate_documents: Optional[List[dict]] = None,
        **kwargs
    ) -> dict:
        """
        문서 리랭킹 메서드 - 표준 인터페이스
        
        Args:
            query (str): 사용자 질문/검색어
            candidate_documents (Optional[List[dict]]): 후보 문서들
                각 문서는 {'doc_id': str, 'chunk_index': int, 'filename': str, 'text': str, 'score': float} 형식
            
        Returns:
            dict: {'retrieved_docs': [{'doc_id': str, 'chunk_index': int, 'score': float, 'filename': str, 'text': str}, ...]}
        """
        pass


class SimpleCompressionRetriever:
    """새로운 인터페이스를 지원하는 압축 리트리버"""
    
    def __init__(self, base_retriever, reranker: BaseReranker):
        self.base_retriever = base_retriever
        self.reranker = reranker
    
    def invoke(self, query: str) -> List[Document]:
        """기존 호환성을 위한 메서드"""
        # 새로운 인터페이스 사용
        result = self.search_and_rerank(query)
        
        # Document 형식으로 변환
        documents = []
        for doc_info in result['retrieved_docs']:
            doc = Document(
                page_content=doc_info['text'],
                metadata={
                    'source': doc_info['doc_id'],
                    'chunk_index': doc_info['chunk_index'],
                    'filename': doc_info['filename']
                }
            )
            documents.append(doc)
        
        return documents
    
    def search_and_rerank(self, query: str) -> dict:
        """새로운 표준 인터페이스"""
        # 기본 검색으로 후보 문서 획득
        candidate_docs = self.base_retriever.get_candidate_documents(query)
        
        # 리랭킹 수행
        result = self.reranker.rerank_documents(query, candidate_docs)
        
        return result


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
class BM25Reranker(BaseReranker, BaseDocumentCompressor):
    """새로운 인터페이스를 사용하는 BM25 리랭커"""
    
    def __init__(self, top_n: int = 12):
        super().__init__(top_n)
    
    def rerank_documents(
        self,
        query: str,
        candidate_documents: Optional[List[dict]] = None,
        **kwargs
    ) -> dict:
        if not candidate_documents:
            return {'retrieved_docs': []}
        
        # BM25 점수 계산
        corpus_tokens = [_tokenize_ko(doc['text']) for doc in candidate_documents]
        bm25 = BM25Okapi(corpus_tokens)
        query_tokens = _tokenize_ko(query)
        scores = bm25.get_scores(query_tokens)
        
        # 점수와 문서를 함께 정렬
        scored_docs = []
        for i, doc in enumerate(candidate_documents):
            scored_docs.append({
                'doc_id': doc['doc_id'],
                'chunk_index': doc['chunk_index'],
                'score': float(scores[i]),
                'filename': doc['filename'],
                'text': doc['text']
            })
        
        # 점수 순으로 정렬하고 상위 N개 선택
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        
        return {'retrieved_docs': scored_docs[:self.top_n]}
    
    def compress_documents(self, documents: List[Document], query: str, callbacks=None) -> List[Document]:
        """LangChain BaseDocumentCompressor 인터페이스 호환"""
        if not documents:
            return []
        
        # Document를 dict 형태로 변환
        candidate_docs = []
        for i, doc in enumerate(documents):
            candidate_docs.append({
                'doc_id': f'doc_{i}',
                'chunk_index': i,
                'filename': doc.metadata.get('source', 'unknown'),
                'text': doc.page_content
            })
        
        # BM25 리랭킹 수행
        result = self.rerank_documents(query, candidate_docs)
        
        # 결과를 Document로 변환
        reranked_docs = []
        for doc_info in result['retrieved_docs']:
            # 원본 문서에서 해당하는 Document 찾기
            for doc in documents:
                if doc.page_content == doc_info['text']:
                    reranked_docs.append(doc)
                    break
        
        return reranked_docs[:self.top_n]


# CrossEncoder 관련 클래스들
class SentenceTransformerRerank(BaseReranker, BaseDocumentCompressor):
    """새로운 인터페이스를 사용하는 CrossEncoder 리랭커"""
    
    def __init__(self, model_name: str, top_n: int = 10):
        super().__init__(top_n)
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
        except ImportError:
            raise ImportError("sentence_transformers가 필요합니다: pip install sentence-transformers")
    
    def rerank_documents(
        self,
        query: str,
        candidate_documents: Optional[List[dict]] = None,
        **kwargs
    ) -> dict:
        if not candidate_documents:
            return {'retrieved_docs': []}
        
        # 쿼리-문서 쌍으로 점수 계산
        pairs = [[query, doc['text']] for doc in candidate_documents]
        scores = self.model.predict(pairs)
        
        # 점수와 문서를 함께 정렬
        scored_docs = []
        for i, doc in enumerate(candidate_documents):
            scored_docs.append({
                'doc_id': doc['doc_id'],
                'chunk_index': doc['chunk_index'],
                'score': float(scores[i]),
                'filename': doc['filename'],
                'text': doc['text']
            })
        
        # 점수 순으로 정렬하고 상위 N개 선택
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        
        return {'retrieved_docs': scored_docs[:self.top_n]}
    
    def compress_documents(self, documents: List[Document], query: str, callbacks=None) -> List[Document]:
        """LangChain BaseDocumentCompressor 인터페이스 호환"""
        if not documents:
            return []
        
        # Document를 dict 형태로 변환
        candidate_docs = []
        for i, doc in enumerate(documents):
            candidate_docs.append({
                'doc_id': f'doc_{i}',
                'chunk_index': i,
                'filename': doc.metadata.get('source', 'unknown'),
                'text': doc.page_content
            })
        
        # CrossEncoder 리랭킹 수행
        result = self.rerank_documents(query, candidate_docs)
        
        # 결과를 Document로 변환
        reranked_docs = []
        for doc_info in result['retrieved_docs']:
            # 원본 문서에서 해당하는 Document 찾기
            for doc in documents:
                if doc.page_content == doc_info['text']:
                    reranked_docs.append(doc)
                    break
        
        return reranked_docs[:self.top_n]


# Embedding 관련 클래스들
class EmbeddingCosineReranker(BaseReranker):
    """새로운 인터페이스를 사용하는 임베딩 코사인 유사도 리랭커"""
    
    def __init__(self, top_n: int = 10, embed_model_name: str = "intfloat/multilingual-e5-large-instruct"):
        super().__init__(top_n)
        self.embed = SentenceTransformerEmbeddings(embed_model_name)
    
    def rerank_documents(
        self,
        query: str,
        candidate_documents: Optional[List[dict]] = None,
        **kwargs
    ) -> dict:
        if not candidate_documents:
            return {'retrieved_docs': []}
        
        # 임베딩 유사도 계산
        q = np.array(self.embed.embed_query(query), dtype=np.float32)
        ds = np.array(self.embed.embed_documents([doc['text'] for doc in candidate_documents]), dtype=np.float32)
        sims = np.dot(ds, q)
        
        # 점수와 문서를 함께 정렬
        scored_docs = []
        for i, doc in enumerate(candidate_documents):
            scored_docs.append({
                'doc_id': doc['doc_id'],
                'chunk_index': doc['chunk_index'],
                'score': float(sims[i]),
                'filename': doc['filename'],
                'text': doc['text']
            })
        
        # 점수 순으로 정렬하고 상위 N개 선택
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        
        return {'retrieved_docs': scored_docs[:self.top_n]}


# Hybrid 관련 클래스들
class CombSumReranker(BaseReranker):
    """CombSum 방식의 하이브리드 리랭커"""
    
    def __init__(self, top_n: int = 12):
        super().__init__(top_n)
        self.embed_model = SentenceTransformerEmbeddings()
    
    def rerank_documents(
        self,
        query: str,
        candidate_documents: Optional[List[dict]] = None,
        **kwargs
    ) -> dict:
        if not candidate_documents:
            return {'retrieved_docs': []}
        
        # BM25 점수 계산
        corpus_tokens = [_tokenize_ko(doc['text']) for doc in candidate_documents]
        bm25 = BM25Okapi(corpus_tokens)
        query_tokens = _tokenize_ko(query)
        bm25_scores = bm25.get_scores(query_tokens)
        
        # 임베딩 유사도 점수 계산
        q_emb = np.array(self.embed_model.embed_query(query), dtype=np.float32)
        doc_embs = np.array(self.embed_model.embed_documents([d['text'] for d in candidate_documents]), dtype=np.float32)
        embed_scores = np.dot(doc_embs, q_emb)
        
        # 점수 정규화 (0-1 범위)
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
        embed_scores = (embed_scores - embed_scores.min()) / (embed_scores.max() - embed_scores.min() + 1e-8)
        
        # CombSum: 두 점수를 합산
        combined_scores = bm25_scores + embed_scores
        
        # 점수와 문서를 함께 정렬
        scored_docs = []
        for i, doc in enumerate(candidate_documents):
            scored_docs.append({
                'doc_id': doc['doc_id'],
                'chunk_index': doc['chunk_index'],
                'score': float(combined_scores[i]),
                'filename': doc['filename'],
                'text': doc['text']
            })
        
        # 점수 순으로 정렬하고 상위 N개 선택
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        
        return {'retrieved_docs': scored_docs[:self.top_n]}


# LLM 관련 클래스들
class LLMReranker(BaseReranker):
    """LLM 기반 리랭커"""
    
    def __init__(self, top_n: int = 10, llm_model: str = "gpt-4o-mini"):
        super().__init__(top_n)
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
    
    def rerank_documents(
        self,
        query: str,
        candidate_documents: Optional[List[dict]] = None,
        **kwargs
    ) -> dict:
        if not candidate_documents or not os.getenv("OPENAI_API_KEY"):
            # API 키가 없으면 원본 순서 유지
            return {'retrieved_docs': candidate_documents[:self.top_n] if candidate_documents else []}
        
        # LLM을 사용한 관련성 평가
        scored_docs = []
        for doc in candidate_documents:
            prompt = f"""
다음 질문과 문서의 관련성을 0-10 점수로 평가해주세요.

질문: {query}

문서: {doc['text'][:1000]}

점수만 숫자로 답하세요 (0-10):"""
            
            try:
                response = self.llm.invoke(prompt).content.strip()
                score = float(response) if response.replace('.', '').isdigit() else 5.0
            except:
                score = 5.0  # 기본 점수
            
            scored_docs.append({
                'doc_id': doc['doc_id'],
                'chunk_index': doc['chunk_index'],
                'score': score,
                'filename': doc['filename'],
                'text': doc['text']
            })
        
        # 점수 순으로 정렬하고 상위 N개 선택
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        
        return {'retrieved_docs': scored_docs[:self.top_n]}


# Rules 관련 클래스들
class LegalRuleBoostReranker(BaseReranker):
    """법률 규칙 기반 부스트 리랭커"""
    
    def __init__(self, top_n: int = 12):
        super().__init__(top_n)
        self.embed_model = SentenceTransformerEmbeddings()
        
        # 법률 조문 패턴
        self._ARTICLE_RE = re.compile(r"제\s?(\d+)\s?조")
        self._PARA_RE = re.compile(r"제\s?(\d+)\s?항")
        self._ITEM_RE = re.compile(r"제\s?(\d+)\s?호")
    
    def _calculate_legal_boost(self, text: str, query: str) -> float:
        """법률 문서의 중요도 부스트 계산"""
        boost = 0.0
        
        # 조문 언급 부스트
        if self._ARTICLE_RE.search(text):
            boost += 0.3
        
        # 항 언급 부스트  
        if self._PARA_RE.search(text):
            boost += 0.2
            
        # 호 언급 부스트
        if self._ITEM_RE.search(text):
            boost += 0.1
        
        # 특정 키워드 부스트
        legal_keywords = ["법률", "조문", "규정", "조항", "부동산", "세법", "종합부동산세"]
        for keyword in legal_keywords:
            if keyword in text:
                boost += 0.1
                
        # 질문 키워드와의 매칭 부스트
        query_words = query.split()
        for word in query_words:
            if len(word) > 1 and word in text:
                boost += 0.05
        
        return min(boost, 1.0)  # 최대 1.0으로 제한
    
    def rerank_documents(
        self,
        query: str,
        candidate_documents: Optional[List[dict]] = None,
        **kwargs
    ) -> dict:
        if not candidate_documents:
            return {'retrieved_docs': []}
        
        # 기본 임베딩 유사도 계산
        q_emb = np.array(self.embed_model.embed_query(query), dtype=np.float32)
        doc_embs = np.array(self.embed_model.embed_documents([d['text'] for d in candidate_documents]), dtype=np.float32)
        base_scores = np.dot(doc_embs, q_emb)
        
        # 법률 규칙 부스트 적용
        scored_docs = []
        for i, doc in enumerate(candidate_documents):
            boost = self._calculate_legal_boost(doc['text'], query)
            boosted_score = base_scores[i] * (1 + boost)
            
            scored_docs.append({
                'doc_id': doc['doc_id'],
                'chunk_index': doc['chunk_index'],
                'score': float(boosted_score),
                'filename': doc['filename'],
                'text': doc['text']
            })
        
        # 점수 순으로 정렬하고 상위 N개 선택
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        
        return {'retrieved_docs': scored_docs[:self.top_n]}


class EmbeddingCosineCompressor(BaseDocumentCompressor):
    """임베딩 코사인 유사도 기반 리랭커"""
    
    def __init__(self, top_n: int = 10, model_name: str = "intfloat/multilingual-e5-large-instruct"):
        super().__init__()
        self.top_n = top_n
        self.embed_model = SentenceTransformerEmbeddings(model_name)
    
    def compress_documents(self, documents: List[Document], query: str, callbacks=None) -> List[Document]:
        if not documents:
            return []
        
        # 쿼리 임베딩
        query_embedding = np.array(self.embed_model.embed_query(query))
        
        # 문서별 코사인 유사도 계산
        doc_scores = []
        for doc in documents:
            doc_embedding = np.array(self.embed_model.embed_documents([doc.page_content])[0])
            cosine_sim = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            doc_scores.append((doc, cosine_sim))
        
        # 점수 순으로 정렬
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in doc_scores[:self.top_n]]


# 다른 임베딩 모델들을 사용하는 리랭커들
class EmbeddingCosineGTECompressor(BaseDocumentCompressor):
    """GTE 임베딩 모델을 사용하는 리랭커"""
    
    def __init__(self, top_n: int = 10):
        super().__init__()
        self.top_n = top_n
        self.embed_model = SentenceTransformerEmbeddings("sentence-transformers/gte-large")
    
    def compress_documents(self, documents: List[Document], query: str, callbacks=None) -> List[Document]:
        if not documents:
            return []
        
        # 쿼리 임베딩
        query_embedding = np.array(self.embed_model.embed_query(query))
        
        # 문서별 코사인 유사도 계산
        doc_scores = []
        for doc in documents:
            doc_embedding = np.array(self.embed_model.embed_documents([doc.page_content])[0])
            cosine_sim = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            doc_scores.append((doc, cosine_sim))
        
        # 점수 순으로 정렬
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in doc_scores[:self.top_n]]


class EmbeddingCosineMPNetCompressor(BaseDocumentCompressor):
    """MPNet 임베딩 모델을 사용하는 리랭커"""
    
    def __init__(self, top_n: int = 10):
        super().__init__()
        self.top_n = top_n
        self.embed_model = SentenceTransformerEmbeddings("sentence-transformers/all-mpnet-base-v2")
    
    def compress_documents(self, documents: List[Document], query: str, callbacks=None) -> List[Document]:
        if not documents:
            return []
        
        # 쿼리 임베딩
        query_embedding = np.array(self.embed_model.embed_query(query))
        
        # 문서별 코사인 유사도 계산
        doc_scores = []
        for doc in documents:
            doc_embedding = np.array(self.embed_model.embed_documents([doc.page_content])[0])
            cosine_sim = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            doc_scores.append((doc, cosine_sim))
        
        # 점수 순으로 정렬
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in doc_scores[:self.top_n]]


class EmbeddingCosineParaphraseCompressor(BaseDocumentCompressor):
    """Paraphrase 임베딩 모델을 사용하는 리랭커"""
    
    def __init__(self, top_n: int = 10):
        super().__init__()
        self.top_n = top_n
        self.embed_model = SentenceTransformerEmbeddings("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    def compress_documents(self, documents: List[Document], query: str, callbacks=None) -> List[Document]:
        if not documents:
            return []
        
        # 쿼리 임베딩
        query_embedding = np.array(self.embed_model.embed_query(query))
        
        # 문서별 코사인 유사도 계산
        doc_scores = []
        for doc in documents:
            doc_embedding = np.array(self.embed_model.embed_documents([doc.page_content])[0])
            cosine_sim = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            doc_scores.append((doc, cosine_sim))
        
        # 점수 순으로 정렬
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in doc_scores[:self.top_n]]


class EmbeddingCosineStellaCompressor(BaseDocumentCompressor):
    """Stella 임베딩 모델을 사용하는 리랭커"""
    
    def __init__(self, top_n: int = 10):
        super().__init__()
        self.top_n = top_n
        self.embed_model = SentenceTransformerEmbeddings("infgrad/stella-base-ko-v2")
    
    def compress_documents(self, documents: List[Document], query: str, callbacks=None) -> List[Document]:
        if not documents:
            return []
        
        # 쿼리 임베딩
        query_embedding = np.array(self.embed_model.embed_query(query))
        
        # 문서별 코사인 유사도 계산
        doc_scores = []
        for doc in documents:
            doc_embedding = np.array(self.embed_model.embed_documents([doc.page_content])[0])
            cosine_sim = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            doc_scores.append((doc, cosine_sim))
        
        # 점수 순으로 정렬
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in doc_scores[:self.top_n]]


def _tokenize_ko(text: str) -> List[str]:
    """한국어 토큰화 함수"""
    import re
    # 간단한 한국어 토큰화 (공백, 구두점 기준)
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens