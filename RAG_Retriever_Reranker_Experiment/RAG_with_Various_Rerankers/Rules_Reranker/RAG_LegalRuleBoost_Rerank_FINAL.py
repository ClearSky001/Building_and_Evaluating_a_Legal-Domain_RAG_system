import sys
import os
from pathlib import Path
import re

# 상위 디렉토리의 fixed_base 모듈 import
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from fixed_base_v2 import (
    SentenceTransformerEmbeddings,
    NaiveVectorStore,
    BaseDocumentCompressor,
    SimpleCompressionRetriever,
    load_embeddings_data,
    setup_environment,
    create_legal_prompt,
    format_docs,
    get_embeddings_file_path
)

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import List, Optional
from langchain_core.documents import Document
import numpy as np


# 법률 조문 패턴 정의
_ARTICLE_RE = re.compile(r"제\s?(\d+)\s?조")
_PARA_RE = re.compile(r"제\s?(\d+)\s?항")
_ITEM_RE = re.compile(r"제\s?(\d+)\s?호")


class LegalRuleBoostCompressor(BaseDocumentCompressor):
    """법률 규칙 기반 부스트 리랭커"""
    def __init__(self, top_n: int = 10):
        super().__init__()
        self.top_n = top_n
        self.embed_model = SentenceTransformerEmbeddings()

    def _calculate_legal_boost(self, text: str, query: str) -> float:
        """법률 문서의 중요도 부스트 계산"""
        boost = 0.0
        
        # 조문 언급 부스트
        if _ARTICLE_RE.search(text):
            boost += 0.3
        
        # 항 언급 부스트  
        if _PARA_RE.search(text):
            boost += 0.2
            
        # 호 언급 부스트
        if _ITEM_RE.search(text):
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

    def compress_documents(self, documents: list[Document], query: str, callbacks=None) -> list[Document]:
        if not documents:
            return []
        
        # 기본 임베딩 유사도 계산
        q_emb = np.array(self.embed_model.embed_query(query), dtype=np.float32)
        doc_embs = np.array(self.embed_model.embed_documents([d.page_content for d in documents]), dtype=np.float32)
        base_scores = np.dot(doc_embs, q_emb)
        
        # 법률 규칙 부스트 적용
        boosted_scores = []
        for i, doc in enumerate(documents):
            boost = self._calculate_legal_boost(doc.page_content, query)
            boosted_score = base_scores[i] * (1 + boost)
            boosted_scores.append(boosted_score)
        
        # 상위 N개 선택
        order = np.argsort(boosted_scores)[::-1][:self.top_n]
        return [documents[i] for i in order]


class LegalRAGSystemLegalRuleBoost:
    def __init__(self, embeddings_file: str = "output_chunks_with_embeddings.json", base_k: int = 80, rerank_top_n: int = 12):
        # 임베딩 파일 경로 올바르게 설정
        self.embeddings_file = get_embeddings_file_path(__file__, embeddings_file)
        self.base_k = base_k
        self.rerank_top_n = rerank_top_n
        
        self.documents = []
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        self.llm = None
        self.embedding_model = None

        self._setup_environment()
        self._initialize_system()

    def _setup_environment(self):
        setup_environment()

    def _load_embeddings_data(self) -> tuple:
        return load_embeddings_data(self.embeddings_file)

    def _create_vectorstore(self, documents, embeddings_array):
        print("🔧 Naive VectorStore 생성 중...")
        self.embedding_model = SentenceTransformerEmbeddings()
        vectorstore = NaiveVectorStore(
            documents=documents,
            embeddings=embeddings_array,
            embedding_function=self.embedding_model
        )
        print(f"✅ Naive VectorStore 생성 완료 (문서 수: {len(documents)})")
        return vectorstore

    def _setup_rag_components(self):
        print("⚙️ RAG 컴포넌트 설정 중 (Legal Rule Boost Re-ranker) ...")
        
        # 기본 리트리버 생성
        base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.base_k})
        
        # Legal Rule Boost 압축기 생성
        compressor = LegalRuleBoostCompressor(top_n=self.rerank_top_n)
        
        # 압축 리트리버 생성
        self.retriever = SimpleCompressionRetriever(base_retriever, compressor)

        # 프롬프트 및 LLM 설정
        legal_prompt = create_legal_prompt()
        
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

        # RAG 체인 설정
        def get_context(question):
            docs = self.retriever.invoke(question)
            return format_docs(docs)
        
        self.rag_chain = (
            {
                "context": get_context,
                "question": RunnablePassthrough(),
            }
            | legal_prompt
            | self.llm
            | StrOutputParser()
        )
        print("✅ RAG 컴포넌트 설정 완료")

    def _initialize_system(self):
        print("🚀 RAG 시스템 초기화 시작...")
        try:
            self.documents, embeddings_array = self._load_embeddings_data()
            self.vectorstore = self._create_vectorstore(self.documents, embeddings_array)
            self._setup_rag_components()
            print("🎉 RAG 시스템 초기화 완료!")
        except Exception as e:
            print(f"❌ 시스템 초기화 중 오류 발생: {e}")
            raise

    def rerank_documents(
        self,
        query: str,
        candidate_documents: Optional[List[dict]] = None,
        model: str = "Legal-Rule-Boost"
    ) -> dict:
        """
        새로운 표준 인터페이스 - 리랭킹만 수행
        
        Args:
            query (str): 사용자 질문/검색어
            candidate_documents (Optional[List[dict]]): 후보 문서들
            model (str): 리랭커 모델/방식 (테스트용)
            
        Returns:
            dict: {'retrieved_docs': [{'doc_id': str, 'chunk_index': int, 'score': float, 'filename': str, 'text': str}, ...]}
        """
        if candidate_documents is None:
            # 후보 문서가 없으면 기본 검색 수행
            base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.base_k})
            candidate_docs = base_retriever.invoke(query)
            # Document 객체를 dict 형태로 변환
            candidate_documents = []
            for i, doc in enumerate(candidate_docs):
                doc_dict = {
                    'doc_id': f"doc_{i}",
                    'chunk_index': i,
                    'score': 1.0,
                    'filename': doc.metadata.get('source', 'unknown'),
                    'text': doc.page_content
                }
                candidate_documents.append(doc_dict)
        
        # Legal Rule Boost 리랭킹 수행
        docs_to_rerank = [Document(page_content=doc['text'], metadata={'source': doc['filename']}) for doc in candidate_documents]
        reranked_docs = self.reranker.compress_documents(docs_to_rerank, query)
        
        # 결과를 원래 형식으로 변환
        result = {
            'retrieved_docs': []
        }
        
        for i, doc in enumerate(reranked_docs):
            doc_dict = {
                'doc_id': f"doc_{i}",
                'chunk_index': i,
                'score': 1.0 - (i * 0.1),  # 순서에 따른 점수
                'filename': doc.metadata.get('source', 'unknown'),
                'text': doc.page_content
            }
            result['retrieved_docs'].append(doc_dict)
        
        return result

    def search(self, query: str, k: int = 10):
        """검색 + 리랭킹을 함께 수행하는 메서드"""
        try:
            # 1. 기본 검색으로 더 많은 문서를 가져옴
            base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.base_k})
            initial_docs = base_retriever.invoke(query)
            
            if not initial_docs:
                return []
            
            # 2. Legal Rule Boost로 리랭킹 수행 (실제로 순서를 바꿈)
            # 초기 검색된 문서들을 Legal Rule Boost로 재정렬
            reranked_docs = self.retriever.reranker.compress_documents(initial_docs, query)
            
            return reranked_docs[:k] if reranked_docs else []
        except Exception as e:
            print(f"❌ 검색 중 오류 발생: {e}")
            return []

    def ask_question(self, question: str, show_sources: bool = True) -> str:
        print(f"\n🤖 질문 처리 중: {question}")
        print("-" * 50)
        try:
            if show_sources:
                relevant_docs = self.retriever.invoke(question)
                print("📚 **참고한 문서:**")
                for i, doc in enumerate(relevant_docs, 1):
                    source = doc.metadata.get('source', '알 수 없음')
                    print(f"  {i}. {source}")
                print()
                
            # OpenAI API 키가 없으면 검색만 수행
            if not os.getenv("OPENAI_API_KEY"):
                return "OpenAI API 키가 설정되지 않아 검색만 수행했습니다."
            
            response = self.rag_chain.invoke(question)
            print("💡 **답변:**")
            print(response)
            return response
        except Exception as e:
            error_msg = f"❌ 답변 생성 중 오류가 발생했습니다: {e}"
            print(error_msg)
            return error_msg


def main():
    print("🚀 법률 문서 RAG 시스템 (Legal Rule Boost Re-ranker) 시작")
    print("=" * 60)
    try:
        rag_system = LegalRAGSystemLegalRuleBoost()
        rag_system.ask_question("종합부동산세법의 목적을 법령 조문을 근거로 하여 설명해주세요.")
        print("\n✅ 프로그램이 정상적으로 종료되었습니다.")
    except Exception as e:
        print(f"❌ 예상치 못한 오류가 발생했습니다: {e}")


if __name__ == "__main__":
    main()
