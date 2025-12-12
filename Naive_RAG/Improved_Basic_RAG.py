import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# LangChain 임포트 (최신 버전)
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings

# SentenceTransformer 임포트 (기존 임베딩과 동일한 모델 사용)
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddings(Embeddings):
    """SentenceTransformer를 LangChain Embeddings 인터페이스로 래핑"""
    
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large-instruct"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """문서들을 임베딩합니다."""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """쿼리를 임베딩합니다."""
        # E5 모델의 경우 쿼리에 "query: " 접두사 추가
        query_text = f"query: {text}"
        embedding = self.model.encode(query_text)
        return embedding.tolist()


class NaiveVectorStore(VectorStore):
    """Naive VectorStore - LangChain 호환 버전"""
    
    def __init__(self, documents: List[Document], embeddings: List[List[float]], embedding_function: Embeddings):
        self.documents = documents
        self._embeddings_matrix = np.array(embeddings, dtype=np.float32)
        self.embedding_function = embedding_function
        
        # 임베딩 정규화
        self._embeddings_matrix = self._embeddings_matrix / np.linalg.norm(self._embeddings_matrix, axis=1, keepdims=True)
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None, **kwargs) -> List[str]:
        """텍스트를 벡터스토어에 추가합니다."""
        raise NotImplementedError("add_texts는 현재 구현되지 않았습니다.")
    
    def similarity_search_by_vector(self, embedding: List[float], k: int = 4, **kwargs) -> List[Document]:
        """벡터를 사용하여 유사한 문서를 검색합니다."""
        query_vector = np.array(embedding, dtype=np.float32)
        query_norm = query_vector / np.linalg.norm(query_vector)
        
        # 코사인 유사도 계산
        similarities = np.dot(self._embeddings_matrix, query_norm)
        
        # 상위 k개 인덱스 추출
        top_k_indices = similarities.argsort()[::-1][:k]
        
        return [self.documents[i] for i in top_k_indices]
    
    def similarity_search(self, query: str, k: int = 4, **kwargs) -> List[Document]:
        """쿼리 텍스트를 사용하여 유사한 문서를 검색합니다."""
        # 쿼리를 임베딩으로 변환
        query_embedding = self.embedding_function.embed_query(query)
        return self.similarity_search_by_vector(query_embedding, k, **kwargs)
    
    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs) -> List[tuple]:
        """유사도 점수와 함께 문서를 검색합니다."""
        query_embedding = self.embedding_function.embed_query(query)
        query_vector = np.array(query_embedding, dtype=np.float32)
        query_norm = query_vector / np.linalg.norm(query_vector)
        
        # 코사인 유사도 계산
        similarities = np.dot(self._embeddings_matrix, query_norm)
        
        # 상위 k개 인덱스와 점수 추출
        top_k_indices = similarities.argsort()[::-1][:k]
        
        results = []
        for idx in top_k_indices:
            doc = self.documents[idx]
            score = float(similarities[idx])
            results.append((doc, score))
        
        return results
    
    @classmethod
    def from_texts(cls, texts: List[str], embedding: Embeddings, metadatas: Optional[List[dict]] = None, **kwargs):
        """텍스트로부터 벡터스토어를 생성합니다."""
        raise NotImplementedError("from_texts는 현재 구현되지 않았습니다.")


class LegalRAGSystem:
    """법률 문서 RAG 시스템 클래스"""
    
    def __init__(self, embeddings_file: str = "output_chunks_with_embeddings.json"):
        """
        RAG 시스템을 초기화합니다.
        
        Args:
            embeddings_file (str): 임베딩된 청크 데이터 파일 경로
        """
        # 파일 경로를 절대경로로 변환 (현재 스크립트 디렉토리 기준)
        if not os.path.isabs(embeddings_file):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.embeddings_file = os.path.join(script_dir, embeddings_file)
        else:
            self.embeddings_file = embeddings_file
        self.documents = []
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        self.llm = None
        self.embedding_model = None
        
        # 환경 설정
        self._setup_environment()
        
        # 시스템 초기화
        self._initialize_system()
    
    def _setup_environment(self):
        """환경변수 및 설정을 로드합니다."""
        load_dotenv()
        
        # LangSmith 설정 (선택적)
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "")
        os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
        
        print("✅ 환경 설정 완료")
    
    def _load_embeddings_data(self) -> tuple:
        """임베딩된 데이터를 로드하고 Document 객체를 생성합니다."""
        print(f"📂 임베딩 데이터 로드 중: {self.embeddings_file}")
        
        with open(self.embeddings_file, "r", encoding="utf-8") as f:
            chunk_data = json.load(f)
        
        documents = []
        embeddings_array = []
        
        for item in chunk_data:
            # LangChain Document 객체 생성
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
    
    def _create_vectorstore(self, documents: List[Document], embeddings_array: List[List[float]]):
        """Naive VectorStore를 생성합니다."""
        print("🔧 Naive VectorStore 생성 중...")
        
        # 기존 임베딩과 동일한 모델 사용
        self.embedding_model = SentenceTransformerEmbeddings()
        
        # Naive VectorStore 생성
        vectorstore = NaiveVectorStore(
            documents=documents,
            embeddings=embeddings_array,
            embedding_function=self.embedding_model
        )
        
        print(f"✅ Naive VectorStore 생성 완료 (문서 수: {len(documents)})")
        return vectorstore
    
    def _setup_rag_components(self):
        """RAG 시스템 컴포넌트를 설정합니다."""
        print("⚙️ RAG 컴포넌트 설정 중...")
        
        # Retriever 생성(Naive Retriever)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # 법률 문서 특화 프롬프트 생성
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
        
        # LLM 생성
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,  # 일관된 답변을 위해 낮은 온도 설정
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Context 포맷팅 함수
        def format_docs(docs):
            """검색된 문서들을 포맷팅합니다."""
            formatted_docs = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source', '알 수 없음')
                content = doc.page_content.strip()
                formatted_docs.append(f"📄 **문서 {i}** ({source})\n{content}")
            return "\n\n" + "\n\n".join(formatted_docs) + "\n\n"
        
        # RAG 체인 생성
        self.rag_chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | legal_prompt
            | self.llm
            | StrOutputParser()
        )
        
        print("✅ RAG 컴포넌트 설정 완료")
    
    def _initialize_system(self):
        """전체 시스템을 초기화합니다."""
        print("🚀 RAG 시스템 초기화 시작...")
        
        try:
            # 데이터 로드
            self.documents, embeddings_array = self._load_embeddings_data()
            
            # 벡터스토어 생성
            self.vectorstore = self._create_vectorstore(self.documents, embeddings_array)
            
            # RAG 컴포넌트 설정
            self._setup_rag_components()
            
            print("🎉 RAG 시스템 초기화 완료!")
            
        except Exception as e:
            print(f"❌ 시스템 초기화 중 오류 발생: {e}")
            raise
    
    def test_similarity_search(self, query: str = "종합부동산세의 목적", k: int = 10):
        """유사도 검색을 테스트합니다."""
        print(f"\n🔍 유사도 검색 테스트")
        print(f"검색 쿼리: '{query}'")
        print(f"검색 결과 ({k}개):")
        print("-" * 50)
        
        if self.vectorstore is None:
            print("❌ 벡터스토어가 초기화되지 않았습니다.")
            return []
        
        try:
            similar_docs = self.vectorstore.similarity_search(query, k=k)
            
            for i, doc in enumerate(similar_docs, 1):
                print(f"[결과 {i}] {doc.metadata['source']}")
                print(f"내용: {doc.page_content[:150]}...")
                print("-" * 50)
            
            return similar_docs
            
        except Exception as e:
            print(f"❌ 유사도 검색 중 오류 발생: {e}")
            return []
    
    def test_similarity_search_with_score(self, query: str = "종합부동산세의 목적", k: int = 10):
        """점수와 함께 유사도 검색을 테스트합니다."""
        print(f"\n🔍 점수 포함 유사도 검색 테스트")
        print(f"검색 쿼리: '{query}'")
        print(f"검색 결과 ({k}개):")
        print("-" * 50)
        
        if self.vectorstore is None:
            print("❌ 벡터스토어가 초기화되지 않았습니다.")
            return []
        
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            for i, (doc, score) in enumerate(results, 1):
                print(f"[결과 {i}] 유사도: {score:.4f}")
                print(f"출처: {doc.metadata['source']}")
                print(f"내용: {doc.page_content[:150]}...")
                print("-" * 50)
            
            return results
            
        except Exception as e:
            print(f"❌ 유사도 검색 중 오류 발생: {e}")
            return []
    
    def ask_question(self, question: str, show_sources: bool = True) -> str:
        """
        법률 질문에 대한 답변을 생성합니다.
        
        Args:
            question (str): 법률 질문
            show_sources (bool): 참고 문서 출처 표시 여부
        
        Returns:
            str: 답변
        """
        print(f"\n🤖 질문 처리 중: {question}")
        print("-" * 50)
        
        try:
            # 관련 문서 검색 및 출처 표시
            if show_sources:
                relevant_docs = self.retriever.invoke(question)
                print("📚 **참고한 문서:**")
                for i, doc in enumerate(relevant_docs, 1):
                    source = doc.metadata.get('source', '알 수 없음')
                    print(f"  {i}. {source}")
                print()
            
            # RAG 답변 생성
            response = self.rag_chain.invoke(question)
            
            print("💡 **답변:**")
            print(response)
            return response
            
        except Exception as e:
            error_msg = f"❌ 답변 생성 중 오류가 발생했습니다: {e}"
            print(error_msg)
            return error_msg
    
    def run_test_questions(self):
        """미리 정의된 테스트 질문들을 실행합니다."""
        test_questions = [
            "종합부동산세법의 목적을 법령 조문을 근거로 하여 설명해주세요.",
            "종합부동산세 납세의무자는 누구인가요?",
            "종합부동산세 과세대상은 무엇인가요?"
        ]
        
        print(f"\n{'='*60}")
        print("📋 테스트 질문 실행")
        print(f"{'='*60}")
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{'='*60}")
            print(f"테스트 {i}: {question}")
            print(f"{'='*60}")
            
            try:
                self.ask_question(question)
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
            
            print(f"\n{'-'*60}")


def interactive_mode(rag_system: LegalRAGSystem):
    """대화형 모드를 실행합니다."""
    print(f"\n{'='*60}")
    print("🤖 법률 문서 질의응답 시스템 - 대화형 모드")
    print("종료하려면 'quit', 'exit', 또는 '종료'를 입력하세요.")
    print(f"{'='*60}")
    
    while True:
        try:
            question = input("\n❓ 질문을 입력하세요: ").strip()
            
            if question.lower() in ['quit', 'exit', '종료', 'q']:
                print("👋 질의응답 시스템을 종료합니다.")
                break
            
            if not question:
                print("⚠️ 질문을 입력해주세요.")
                continue
            
            rag_system.ask_question(question)
            
        except KeyboardInterrupt:
            print("\n\n👋 사용자가 중단했습니다. 시스템을 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 예상치 못한 오류가 발생했습니다: {e}")


def main():
    """메인 함수"""
    print("🚀 법률 문서 RAG 시스템 시작")
    print("=" * 60)
    
    try:
        # RAG 시스템 초기화
        rag_system = LegalRAGSystem()
        
        # 유사도 검색 테스트
        rag_system.test_similarity_search()
        
        # 점수 포함 유사도 검색 테스트
        rag_system.test_similarity_search_with_score()
        
        # 미리 정의된 테스트 질문 실행
        rag_system.run_test_questions()
        
        # 대화형 모드 실행 (선택적)
        user_input = input("\n🤔 대화형 모드를 시작하시겠습니까? (y/n): ").strip().lower()
        if user_input in ['y', 'yes', '예', 'ㅇ']:
            interactive_mode(rag_system)
        
        print("\n✅ 프로그램이 정상적으로 종료되었습니다.")
        
    except FileNotFoundError:
        print("❌ 오류: 'output_chunks_with_embeddings.json' 파일을 찾을 수 없습니다.")
        print("📁 현재 디렉토리에 임베딩 파일이 있는지 확인해주세요.")
    except Exception as e:
        print(f"❌ 예상치 못한 오류가 발생했습니다: {e}")


if __name__ == "__main__":
    main()