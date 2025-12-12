"""
모든 리랭커에서 사용할 공통 기본 클래스들
LangChain 버전 호환성 문제를 해결하기 위한 사용자 정의 클래스들
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Any
from langchain_core.documents import Document

class BaseDocumentCompressor(ABC):
    """사용자 정의 BaseDocumentCompressor - LangChain 호환성을 위해"""
    
    @abstractmethod
    def compress_documents(
        self,
        documents: List[Document],
        query: str,
        callbacks: Optional[Any] = None,
    ) -> List[Document]:
        """문서 압축/리랭킹 메서드"""
        pass

class SentenceTransformerRerank(BaseDocumentCompressor):
    """사용자 정의 SentenceTransformerRerank - CrossEncoder 기반"""
    
    def __init__(self, model_name: str, top_n: int = 10):
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
