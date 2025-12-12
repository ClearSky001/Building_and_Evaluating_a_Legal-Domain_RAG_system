import sys
import os
import time
from pathlib import Path

# 상위 디렉토리의 fixed_base 모듈 import
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from fixed_base_v2 import (
    SentenceTransformerEmbeddings,
    NaiveVectorStore,
    BaseDocumentCompressor,
    BaseReranker,
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


class CohereRerank(BaseReranker):
    """실제 Cohere API를 사용하는 리랭커"""
    def __init__(self, top_n: int = 10):
        super().__init__(top_n)
        # Rate limiting 변수들
        self.last_call_time = 0
        self.call_count = 0
        self.rate_limit_delay = 7  # 7초 대기 (Trial: 10 calls/minute)
        
        try:
            import cohere
            # .env 파일 로드
            from dotenv import load_dotenv
            load_dotenv()
            
            # Cohere API 키 확인
            cohere_api_key = os.getenv("COHERE_API_KEY")
            if not cohere_api_key:
                print("⚠️ COHERE_API_KEY가 설정되지 않았습니다. CrossEncoder를 대신 사용합니다.")
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                self.use_cohere = False
            else:
                print(f"✅ Cohere API 키 확인됨: {cohere_api_key[:10]}...")
                self.co = cohere.Client(cohere_api_key)
                self.use_cohere = True
                # Cohere 사용 시에도 model 속성 추가 (fallback용)
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except ImportError:
            print("⚠️ cohere 패키지가 설치되지 않았습니다. CrossEncoder를 대신 사용합니다.")
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            self.use_cohere = False

    def _rate_limit_check(self):
        """Rate limiting 체크 및 대기"""
        current_time = time.time()
        
        # 첫 번째 호출이거나 1분이 지났으면 카운터 리셋
        if current_time - self.last_call_time > 60:
            self.call_count = 0
        
        # 10번 호출했으면 7초 대기
        if self.call_count >= 10:
            wait_time = self.rate_limit_delay - (current_time - self.last_call_time)
            if wait_time > 0:
                print(f"⏳ Cohere API Rate Limit 도달. {wait_time:.1f}초 대기 중...")
                time.sleep(wait_time)
            self.call_count = 0
        
        self.last_call_time = current_time
        self.call_count += 1

    def rerank_documents(self, query: str, candidate_documents=None, **kwargs):
        """새로운 표준 인터페이스 - rerank_documents 메서드"""
        if not candidate_documents:
            return {'retrieved_docs': []}
        
        try:
            if self.use_cohere:
                # Rate limiting 체크
                self._rate_limit_check()
                
                # Cohere rerank API 사용
                texts = [doc['text'] for doc in candidate_documents]
                response = self.co.rerank(
                    model="rerank-multilingual-v3.0",
                    query=query,
                    documents=texts,
                    top_n=self.top_n
                )
                
                # Cohere 결과를 기반으로 문서 정렬
                reranked_docs = []
                for result in response.results:
                    original_doc = candidate_documents[result.index]
                    reranked_docs.append({
                        'doc_id': original_doc['doc_id'],
                        'chunk_index': original_doc['chunk_index'],
                        'filename': original_doc['filename'],
                        'text': original_doc['text'],
                        'score': result.relevance_score
                    })
                
                return {'retrieved_docs': reranked_docs}
            else:
                # CrossEncoder fallback
                pairs = [[query, doc['text']] for doc in candidate_documents]
                scores = self.model.predict(pairs)
                
                # 점수 순으로 정렬
                doc_scores = list(zip(candidate_documents, scores))
                doc_scores.sort(key=lambda x: x[1], reverse=True)
                
                reranked_docs = []
                for doc, score in doc_scores[:self.top_n]:
                    reranked_docs.append({
                        'doc_id': doc['doc_id'],
                        'chunk_index': doc['chunk_index'],
                        'filename': doc['filename'],
                        'text': doc['text'],
                        'score': float(score)
                    })
                
                return {'retrieved_docs': reranked_docs}
                
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate limit" in error_msg.lower():
                print(f"⚠️ Cohere API Rate Limit 도달. CrossEncoder로 대체합니다.")
                # CrossEncoder로 fallback
                pairs = [[query, doc['text']] for doc in candidate_documents]
                scores = self.model.predict(pairs)
                
                # 점수 순으로 정렬
                doc_scores = list(zip(candidate_documents, scores))
                doc_scores.sort(key=lambda x: x[1], reverse=True)
                
                reranked_docs = []
                for doc, score in doc_scores[:self.top_n]:
                    reranked_docs.append({
                        'doc_id': doc['doc_id'],
                        'chunk_index': doc['chunk_index'],
                        'filename': doc['filename'],
                        'text': doc['text'],
                        'score': float(score)
                    })
                
                return {'retrieved_docs': reranked_docs}
            else:
                print(f"⚠️ 리랭킹 중 오류 발생: {e}")
                return {'retrieved_docs': candidate_documents[:self.top_n]}

    def compress_documents(self, documents: list[Document], query: str, callbacks=None) -> list[Document]:
        """기존 호환성을 위한 compress_documents 메서드"""
        if not documents:
            return []
        
        # Document를 dict 형태로 변환
        candidate_docs = []
        for i, doc in enumerate(documents):
            candidate_docs.append({
                'doc_id': f'doc_{i}',
                'chunk_index': i,
                'filename': doc.metadata.get('source', 'unknown'),
                'text': doc.page_content,
                'score': 0.0
            })
        
        # rerank_documents 호출
        result = self.rerank_documents(query, candidate_docs)
        
        # Document 형식으로 변환
        reranked_docs = []
        for doc_info in result['retrieved_docs']:
            doc = Document(
                page_content=doc_info['text'],
                metadata={
                    'source': doc_info['filename'],
                    'chunk_index': doc_info['chunk_index'],
                    'score': doc_info['score']
                }
            )
            reranked_docs.append(doc)
        
        return reranked_docs


class LegalRAGSystemCohereRerank:
    def __init__(self, embeddings_file: str = "output_chunks_with_embeddings.json", base_k: int = 80, rerank_top_n: int = 10):
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
        # Cohere 리랭커는 기본 E5 모델 사용 (공정한 비교를 위해)
        self.embedding_model = SentenceTransformerEmbeddings()
        vectorstore = NaiveVectorStore(
            documents=documents,
            embeddings=embeddings_array,
            embedding_function=self.embedding_model
        )
        print(f"✅ Naive VectorStore 생성 완료 (문서 수: {len(documents)})")
        return vectorstore

    def _setup_rag_components(self):
        print("⚙️ RAG 컴포넌트 설정 중 (Cohere Reranker) ...")
        
        # 기본 리트리버 생성
        base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.base_k})
        
        # Cohere 압축기 생성
        compressor = CohereRerank(top_n=self.rerank_top_n)
        
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
            # 1. 기본 검색으로 더 많은 문서를 가져옴
            base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.base_k})
            initial_docs = base_retriever.invoke(query)
            
            if not initial_docs:
                return []
            
            # 2. Cohere로 리랭킹 수행 (실제로 순서를 바꿈)
            # 초기 검색된 문서들을 Cohere로 재정렬
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
    print("🚀 법률 문서 RAG 시스템 (Cohere Reranker) 시작")
    print("=" * 60)
    print("📝 Cohere API 사용을 위해서는 다음이 필요합니다:")
    print("   1. pip install cohere")
    print("   2. COHERE_API_KEY 환경변수 설정")
    print("   3. API 키가 없으면 CrossEncoder를 대신 사용합니다.")
    print("=" * 60)
    try:
        rag_system = LegalRAGSystemCohereRerank()
        rag_system.ask_question("종합부동산세법의 목적을 법령 조문을 근거로 하여 설명해주세요.")
        print("\n✅ 프로그램이 정상적으로 종료되었습니다.")
    except Exception as e:
        print(f"❌ 예상치 못한 오류가 발생했습니다: {e}")


if __name__ == "__main__":
    main()
