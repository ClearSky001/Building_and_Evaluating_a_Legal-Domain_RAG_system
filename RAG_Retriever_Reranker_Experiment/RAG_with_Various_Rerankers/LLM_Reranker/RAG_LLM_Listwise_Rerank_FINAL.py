"""
완전히 수정된 LLM Listwise 리랭커 - 모든 문제 해결
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
    get_embeddings_file_path
)

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document


class LLMListwiseReranker(BaseDocumentCompressor):
    """LLM 기반 Listwise 리랭킹"""
    def __init__(self, top_n: int = 10, llm_model: str = "gpt-4o-mini"):
        super().__init__()
        self.top_n = top_n
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

    def compress_documents(self, documents: list[Document], query: str, callbacks=None) -> list[Document]:
        if not documents or not os.getenv("OPENAI_API_KEY"):
            # API 키가 없으면 원본 순서 유지
            return documents[:self.top_n]
        
        # 문서들을 번호와 함께 제시
        doc_list = []
        for i, doc in enumerate(documents[:20], 1):  # 최대 20개까지만 처리
            content = doc.page_content[:500]  # 내용 제한
            doc_list.append(f"{i}. {content}")
        
        doc_text = "\n\n".join(doc_list)
        
        prompt = f"""
다음 질문과 관련하여 아래 문서들을 관련성 순으로 정렬해주세요.

질문: {query}

문서들:
{doc_text}

가장 관련성이 높은 순서대로 문서 번호만 나열해주세요 (예: 3,7,1,5,2):
"""
        
        try:
            response = self.llm.invoke(prompt).content.strip()
            
            # 응답에서 번호 추출
            import re
            numbers = re.findall(r'\d+', response)
            
            # 순서대로 문서 재배열
            reordered_docs = []
            used_indices = set()
            
            for num_str in numbers:
                try:
                    idx = int(num_str) - 1  # 1-based to 0-based
                    if 0 <= idx < len(documents) and idx not in used_indices:
                        reordered_docs.append(documents[idx])
                        used_indices.add(idx)
                        if len(reordered_docs) >= self.top_n:
                            break
                except ValueError:
                    continue
            
            # 부족한 경우 나머지 문서들로 채움
            for i, doc in enumerate(documents):
                if i not in used_indices and len(reordered_docs) < self.top_n:
                    reordered_docs.append(doc)
            
            return reordered_docs[:self.top_n]
            
        except Exception:
            # 오류 시 원본 순서 유지
            return documents[:self.top_n]


class LegalRAGSystemLLMListwise:
    def __init__(self, embeddings_file: str = "output_chunks_with_embeddings.json", base_k: int = 50, rerank_top_n: int = 10):
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
        print("⚙️ RAG 컴포넌트 설정 중 (LLM Listwise Re-ranker) ...")
        
        # 기본 리트리버 생성
        base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.base_k})
        
        # LLM Listwise 압축기 생성
        compressor = LLMListwiseReranker(top_n=self.rerank_top_n)
        
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
    print("🚀 법률 문서 RAG 시스템 (LLM Listwise Re-ranker) 시작")
    print("=" * 60)
    try:
        rag_system = LegalRAGSystemLLMListwise()
        rag_system.ask_question("종합부동산세법의 목적을 법령 조문을 근거로 하여 설명해주세요.")
        print("\n✅ 프로그램이 정상적으로 종료되었습니다.")
    except Exception as e:
        print(f"❌ 예상치 못한 오류가 발생했습니다: {e}")


if __name__ == "__main__":
    main()
