"""
완전히 수정된 BGE Base 리랭커 - 모든 문제 해결
"""
import sys
import os
from pathlib import Path

# 상위 디렉토리의 fixed_base 모듈 import
current_dir = Path(__file__).parent
parent_dir = current_dir.parent.parent
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
    get_embeddings_file_path,
    SentenceTransformerRerank
)

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


class LegalRAGSystemBGEBase:
    def __init__(self, embeddings_file: str = "output_chunks_with_embeddings.json", base_k: int = 80, rerank_top_n: int = 10):
        # 임베딩 파일 경로 올바르게 설정 (BGE 계열은 2단계 상위)
        if not os.path.isabs(embeddings_file):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.embeddings_file = os.path.join(script_dir, "..", "..", "..", "output_chunks_with_embeddings.json")
            self.embeddings_file = os.path.normpath(self.embeddings_file)
        else:
            self.embeddings_file = embeddings_file
            
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
        print("⚙️ RAG 컴포넌트 설정 중 (BGE Base Reranker) ...")
        
        # 기본 리트리버 생성
        base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.base_k})
        
        # BGE Base 압축기 생성
        compressor = SentenceTransformerRerank(
            model_name="BAAI/bge-reranker-base",
            top_n=self.rerank_top_n,
        )
        
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

    def search(self, query: str, k: int = 10):
        """검색 + 리랭킹을 함께 수행하는 메서드"""
        try:
            # SimpleCompressionRetriever의 invoke 메서드 사용
            reranked_docs = self.retriever.invoke(query)
            
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
    print("🚀 법률 문서 RAG 시스템 (Embedding Cosine GTE Re-ranker) 시작")
    print("=" * 60)
    try:
        rag_system = LegalRAGSystemBGEBase()
        rag_system.ask_question("종합부동산세법의 목적을 법령 조문을 근거로 하여 설명해주세요.")
        print("\n✅ 프로그램이 정상적으로 종료되었습니다.")
    except Exception as e:
        print(f"❌ 예상치 못한 오류가 발생했습니다: {e}")


if __name__ == "__main__":
    main()
