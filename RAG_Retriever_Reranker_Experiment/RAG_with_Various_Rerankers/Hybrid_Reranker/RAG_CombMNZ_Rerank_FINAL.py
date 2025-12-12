"""
완전히 수정된 CombMNZ Hybrid 리랭커 - 모든 문제 해결
"""
import sys
import os
from pathlib import Path

# 상위 디렉토리의 fixed_base 모듈 import
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from fixed_base import (
    SentenceTransformerEmbeddings,
    NaiveVectorStore,
    BaseDocumentCompressor,
    SimpleCompressionRetriever,
    load_embeddings_data,
    setup_environment,
    create_legal_prompt,
    format_docs,
    get_embeddings_file_path,
    _tokenize_ko
)

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import numpy as np


class CombMNZCompressor(BaseDocumentCompressor):
    """CombMNZ 방식의 하이브리드 리랭커"""
    def __init__(self, top_n: int = 10):
        super().__init__()
        self.top_n = top_n
        self.embed_model = SentenceTransformerEmbeddings()

    def compress_documents(self, documents: list[Document], query: str, callbacks=None) -> list[Document]:
        if not documents:
            return []
        
        # BM25 점수 계산
        corpus_tokens = [_tokenize_ko(doc.page_content) for doc in documents]
        bm25 = BM25Okapi(corpus_tokens)
        query_tokens = _tokenize_ko(query)
        bm25_scores = bm25.get_scores(query_tokens)
        
        # 임베딩 유사도 점수 계산
        q_emb = np.array(self.embed_model.embed_query(query), dtype=np.float32)
        doc_embs = np.array(self.embed_model.embed_documents([d.page_content for d in documents]), dtype=np.float32)
        embed_scores = np.dot(doc_embs, q_emb)
        
        # 점수 정규화 (0-1 범위)
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
        embed_scores = (embed_scores - embed_scores.min()) / (embed_scores.max() - embed_scores.min() + 1e-8)
        
        # CombMNZ: 점수 합 * 검색된 시스템 수
        # 여기서는 2개 시스템(BM25, Embedding)이므로 항상 2를 곱함
        combined_scores = (bm25_scores + embed_scores) * 2
        
        # 상위 N개 선택
        order = combined_scores.argsort()[::-1][:self.top_n]
        return [documents[i] for i in order]


class LegalRAGSystemCombMNZ:
    def __init__(self, embeddings_file: str = "output_chunks_with_embeddings.json", base_k: int = 100, rerank_top_n: int = 12):
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
        print("⚙️ RAG 컴포넌트 설정 중 (CombMNZ Hybrid Re-ranker) ...")
        
        # 기본 리트리버 생성
        base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.base_k})
        
        # CombMNZ 압축기 생성
        compressor = CombMNZCompressor(top_n=self.rerank_top_n)
        
        # 압축 리트리버 생성
        self.retriever = SimpleCompressionRetriever(base_retriever, compressor)

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
    print("🚀 법률 문서 RAG 시스템 (CombMNZ Hybrid Re-ranker) 시작")
    print("=" * 60)
    try:
        rag_system = LegalRAGSystemCombMNZ()
        rag_system.ask_question("종합부동산세법의 목적을 법령 조문을 근거로 하여 설명해주세요.")
        print("\n✅ 프로그램이 정상적으로 종료되었습니다.")
    except Exception as e:
        print(f"❌ 예상치 못한 오류가 발생했습니다: {e}")


if __name__ == "__main__":
    main()
