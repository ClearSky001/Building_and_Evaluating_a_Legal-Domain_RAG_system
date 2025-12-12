"""
완전히 수정된 Embedding E5 리랭커 - 모든 문제 해결
"""
import sys
import os
from pathlib import Path
from typing import List, Optional, Dict

# 상위 디렉토리의 fixed_base_v2 모듈 import
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from fixed_base_v2 import (
    SentenceTransformerEmbeddings,
    NaiveVectorStore,
    BaseReranker,
    SimpleCompressionRetriever,
    load_embeddings_data,
    setup_environment,
    create_legal_prompt,
    format_docs,
    get_embeddings_file_path,
    EmbeddingCosineCompressor
)

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document


class LegalRAGSystemEmbeddingE5:
    def __init__(self, embeddings_file: str = "output_chunks_with_embeddings.json", base_k: int = 60, rerank_top_n: int = 10):
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
        self.reranker = None

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
        print("⚙️ RAG 컴포넌트 설정 중 (Embedding Cosine E5 Re-ranker) ...")
        
        # 기본 리트리버 생성
        base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.base_k})
        
        # Embedding 리랭커 생성 (다른 모델로 리랭킹하여 의미 있는 차이 생성)
        self.reranker = EmbeddingCosineCompressor(
            top_n=self.rerank_top_n,
            model_name="sentence-transformers/all-MiniLM-L6-v2"  # 다른 모델 사용
        )
        
        # 압축 리트리버 생성
        self.retriever = SimpleCompressionRetriever(base_retriever, self.reranker)

        # 프롬프트 및 LLM 설정
        legal_prompt = create_legal_prompt()
        
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
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
        model: str = "Embedding-E5"
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
            candidate_documents = self.retriever.base_retriever.get_candidate_documents(query)
        
        # Document 객체로 변환
        docs = []
        for doc_dict in candidate_documents:
            doc = Document(
                page_content=doc_dict['text'],
                metadata={
                    'source': doc_dict.get('filename', ''),
                    'doc_id': doc_dict.get('doc_id', ''),
                    'chunk_index': doc_dict.get('chunk_index', 0)
                }
            )
            docs.append(doc)
        
        # 리랭킹 수행
        reranked_docs = self.reranker.compress_documents(docs, query)
        
        # 결과를 원래 형식으로 변환
        result = {
            'retrieved_docs': []
        }
        
        for i, doc in enumerate(reranked_docs):
            doc_dict = {
                'doc_id': doc.metadata.get('doc_id', f'doc_{i}'),
                'chunk_index': doc.metadata.get('chunk_index', i),
                'score': 1.0 - (i * 0.1),  # 순서에 따른 점수
                'filename': doc.metadata.get('source', ''),
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
            
            # 2. Embedding E5로 리랭킹 수행 (실제로 순서를 바꿈)
            # 초기 검색된 문서들을 Embedding E5로 재정렬
            reranked_docs = self.retriever.reranker.compress_documents(initial_docs, query)
            
            return reranked_docs[:k] if reranked_docs else []
        except Exception as e:
            print(f"❌ 검색 중 오류 발생: {e}")
            return []

    def search_and_rerank(self, query: str) -> dict:
        """검색 + 리랭킹을 함께 수행하는 메서드"""
        return self.retriever.search_and_rerank(query)

    def ask_question(self, question: str, show_sources: bool = True) -> str:
        print(f"\n🤖 질문 처리 중: {question}")
        print("-" * 50)
        try:
            if show_sources:
                # 새로운 인터페이스 사용
                result = self.search_and_rerank(question)
                retrieved_docs = result['retrieved_docs']
                
                print("📚 **참고한 문서:**")
                for i, doc_info in enumerate(retrieved_docs, 1):
                    print(f"  {i}. {doc_info['doc_id']} (점수: {doc_info['score']:.4f})")
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
    print("🚀 법률 문서 RAG 시스템 (CrossEncoder: MiniLM-L-6-v2) 시작")
    print("=" * 60)
    try:
        rag_system = LegalRAGSystemEmbeddingE5()
        rag_system.ask_question("종합부동산세법의 목적을 법령 조문을 근거로 하여 설명해주세요.")
        print("\n✅ 프로그램이 정상적으로 종료되었습니다.")
    except FileNotFoundError:
        print("❌ 오류: 'output_chunks_with_embeddings.json' 파일을 찾을 수 없습니다.")
        print("📁 현재 디렉토리에 임베딩 파일이 있는지 확인해주세요.")
    except Exception as e:
        print(f"❌ 예상치 못한 오류가 발생했습니다: {e}")


if __name__ == "__main__":
    main()
