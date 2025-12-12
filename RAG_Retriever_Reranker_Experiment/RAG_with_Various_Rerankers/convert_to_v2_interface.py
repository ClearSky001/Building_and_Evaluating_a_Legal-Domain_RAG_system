#!/usr/bin/env python3
"""
모든 FINAL 파일들을 새로운 V2 인터페이스로 자동 변환하는 스크립트
"""
import os
import re
from pathlib import Path
from typing import Dict, List

def get_reranker_info(file_path: Path) -> Dict:
    """파일 경로에서 리랭커 정보 추출"""
    file_name = file_path.stem
    
    # 리랭커 타입별 정보 매핑
    reranker_mappings = {
        # BM25 계열
        "RAG_BM25_Rerank_FINAL": {
            "reranker_class": "BM25Reranker",
            "description": "BM25 기본",
            "model_name": "BM25"
        },
        "RAG_BM25_CharNgram_Rerank_FINAL": {
            "reranker_class": "BM25CharNgramReranker", 
            "description": "BM25 CharNgram",
            "model_name": "BM25-CharNgram"
        },
        "RAG_BM25_Kiwi_Rerank_FINAL": {
            "reranker_class": "BM25KiwiReranker",
            "description": "BM25 Kiwi", 
            "model_name": "BM25-Kiwi"
        },
        "RAG_BM25_Regex_Rerank_FINAL": {
            "reranker_class": "BM25RegexReranker",
            "description": "BM25 Regex",
            "model_name": "BM25-Regex"
        },
        "RAG_BM25_Stopword_Rerank_FINAL": {
            "reranker_class": "BM25StopwordReranker",
            "description": "BM25 Stopword",
            "model_name": "BM25-Stopword"
        },
        
        # CrossEncoder 계열
        "RAG_CE_MiniLM_L6_Rerank_FINAL": {
            "reranker_class": "SentenceTransformerRerank",
            "description": "CrossEncoder MiniLM L6",
            "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2"
        },
        "RAG_CE_MiniLM_L12_Rerank_FINAL": {
            "reranker_class": "SentenceTransformerRerank", 
            "description": "CrossEncoder MiniLM L12",
            "model_name": "cross-encoder/ms-marco-MiniLM-L-12-v2"
        },
        
        # BGE 계열
        "RAG_BGE_Base_Rerank_FINAL": {
            "reranker_class": "SentenceTransformerRerank",
            "description": "BGE Base",
            "model_name": "BAAI/bge-reranker-base"
        },
        
        # Embedding 계열
        "RAG_EmbeddingCosine_E5_Rerank_FINAL": {
            "reranker_class": "EmbeddingCosineReranker",
            "description": "Embedding E5",
            "model_name": "intfloat/multilingual-e5-large-instruct"
        },
        "RAG_EmbeddingCosine_GTE_Rerank_FINAL": {
            "reranker_class": "EmbeddingCosineReranker",
            "description": "Embedding GTE", 
            "model_name": "thenlper/gte-multilingual-base"
        },
        
        # Hybrid 계열
        "RAG_CombSum_Rerank_FINAL": {
            "reranker_class": "CombSumReranker",
            "description": "Hybrid CombSum",
            "model_name": "CombSum"
        },
        
        # LLM 계열
        "RAG_LLM_Rerank_FINAL": {
            "reranker_class": "LLMReranker",
            "description": "LLM 기본",
            "model_name": "gpt-4o-mini"
        },
        
        # Rules 계열
        "RAG_LegalRuleBoost_Rerank_FINAL": {
            "reranker_class": "LegalRuleBoostReranker",
            "description": "Legal Rule Boost",
            "model_name": "LegalRuleBoost"
        }
    }
    
    return reranker_mappings.get(file_name, {
        "reranker_class": "BM25Reranker",
        "description": "Unknown",
        "model_name": "Unknown"
    })

def generate_v2_content(file_path: Path, info: Dict) -> str:
    """V2 인터페이스에 맞는 파일 내용 생성"""
    
    # 클래스 이름 추출
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 기존 클래스 이름 찾기
    class_match = re.search(r'class (LegalRAGSystem\w+):', content)
    class_name = class_match.group(1) if class_match else "LegalRAGSystemUnknown"
    
    # BGE 계열인지 확인
    is_bge = "BGE 계열" in str(file_path)
    
    # 경로 설정
    if is_bge:
        path_setup = '''# 임베딩 파일 경로 올바르게 설정 (BGE 계열은 2단계 상위)
        if not os.path.isabs(embeddings_file):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.embeddings_file = os.path.join(script_dir, "..", "..", "..", "output_chunks_with_embeddings.json")
            self.embeddings_file = os.path.normpath(self.embeddings_file)
        else:
            self.embeddings_file = embeddings_file'''
    else:
        path_setup = '''# 임베딩 파일 경로 올바르게 설정
        self.embeddings_file = get_embeddings_file_path(__file__, embeddings_file)'''
    
    # 리랭커 설정
    reranker_class = info['reranker_class']
    model_name = info['model_name']
    
    if reranker_class == "SentenceTransformerRerank":
        reranker_init = f'''self.reranker = {reranker_class}(
            model_name="{model_name}",
            top_n=self.rerank_top_n,
        )'''
    elif reranker_class in ["EmbeddingCosineReranker"]:
        reranker_init = f'''self.reranker = {reranker_class}(
            top_n=self.rerank_top_n,
            embed_model_name="{model_name}"
        )'''
    else:
        reranker_init = f'''self.reranker = {reranker_class}(top_n=self.rerank_top_n)'''

    v2_content = f'''"""
새로운 표준 인터페이스를 사용하는 {info['description']} 리랭커 V2
"""
import sys
import os
from pathlib import Path
from typing import List, Optional, Dict

# 상위 디렉토리의 fixed_base_v2 모듈 import
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
{"parent_dir = parent_dir.parent  # BGE 계열은 한 단계 더" if is_bge else ""}
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
    {reranker_class}
)

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


class {class_name}:
    def __init__(self, embeddings_file: str = "output_chunks_with_embeddings.json", base_k: int = 80, rerank_top_n: int = 10):
        {path_setup}
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
        print(f"✅ Naive VectorStore 생성 완료 (문서 수: {{len(documents)}})")
        return vectorstore

    def _setup_rag_components(self):
        print("⚙️ RAG 컴포넌트 설정 중 ({info['description']} Re-ranker) ...")
        
        # 기본 리트리버 생성
        base_retriever = self.vectorstore.as_retriever(search_kwargs={{"k": self.base_k}})
        
        # 리랭커 생성
        {reranker_init}
        
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
            {{
                "context": get_context,
                "question": RunnablePassthrough(),
            }}
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
            print(f"❌ 시스템 초기화 중 오류 발생: {{e}}")
            raise

    def rerank_documents(
        self,
        query: str,
        candidate_documents: Optional[List[dict]] = None,
        model: str = "{info['model_name']}"
    ) -> dict:
        """
        새로운 표준 인터페이스 - 리랭킹만 수행
        
        Args:
            query (str): 사용자 질문/검색어
            candidate_documents (Optional[List[dict]]): 후보 문서들
            model (str): 리랭커 모델/방식 (테스트용)
            
        Returns:
            dict: {{'retrieved_docs': [{{'doc_id': str, 'chunk_index': int, 'score': float, 'filename': str, 'text': str}}, ...]}}
        """
        if candidate_documents is None:
            # 후보 문서가 없으면 기본 검색 수행
            candidate_documents = self.retriever.base_retriever.get_candidate_documents(query)
        
        return self.reranker.rerank_documents(query, candidate_documents)

    def search_and_rerank(self, query: str) -> dict:
        """검색 + 리랭킹을 함께 수행하는 메서드"""
        return self.retriever.search_and_rerank(query)

    def ask_question(self, question: str, show_sources: bool = True) -> str:
        print(f"\\n🤖 질문 처리 중: {{question}}")
        print("-" * 50)
        try:
            if show_sources:
                # 새로운 인터페이스 사용
                result = self.search_and_rerank(question)
                retrieved_docs = result['retrieved_docs']
                
                print("📚 **참고한 문서:**")
                for i, doc_info in enumerate(retrieved_docs, 1):
                    print(f"  {{i}}. {{doc_info['doc_id']}} (점수: {{doc_info['score']:.4f}})")
                print()
                
            # OpenAI API 키가 없으면 검색만 수행
            if not os.getenv("OPENAI_API_KEY"):
                return "OpenAI API 키가 설정되지 않아 검색만 수행했습니다."
            
            response = self.rag_chain.invoke(question)
            print("💡 **답변:**")
            print(response)
            return response
        except Exception as e:
            error_msg = f"❌ 답변 생성 중 오류가 발생했습니다: {{e}}"
            print(error_msg)
            return error_msg


def main():
    print("🚀 법률 문서 RAG 시스템 ({info['description']} Re-ranker V2) 시작")
    print("=" * 60)
    try:
        rag_system = {class_name}()
        
        # 새로운 인터페이스 테스트
        query = "종합부동산세법의 목적은 무엇인가요?"
        result = rag_system.search_and_rerank(query)
        
        print(f"검색 결과: {{len(result['retrieved_docs'])}}개 문서")
        for i, doc in enumerate(result['retrieved_docs'][:3], 1):
            print(f"{{i}}. {{doc['doc_id']}} (점수: {{doc['score']:.4f}})")
            print(f"   내용: {{doc['text'][:100]}}...")
        
        print("\\n✅ 프로그램이 정상적으로 종료되었습니다.")
    except Exception as e:
        print(f"❌ 예상치 못한 오류가 발생했습니다: {{e}}")


if __name__ == "__main__":
    main()'''
    
    return v2_content

def convert_all_final_files():
    """모든 FINAL 파일들을 V2로 변환"""
    base_dir = Path(__file__).parent
    
    # 변환할 폴더들
    folders = [
        "BM25_Reranker",
        "CrossEncoder", 
        "CrossEncoder/BGE 계열",
        "Embedding_Reranker",
        "Hybrid_Reranker",
        "LLM_Reranker",
        "Rules_Reranker"
    ]
    
    converted_count = 0
    total_count = 0
    
    for folder in folders:
        folder_path = base_dir / folder
        if not folder_path.exists():
            continue
            
        print(f"\n📁 {folder} 폴더 처리 중...")
        
        # FINAL 파일들 찾기
        final_files = list(folder_path.glob("*_FINAL.py"))
        
        for final_file in final_files:
            total_count += 1
            
            try:
                # 리랭커 정보 추출
                info = get_reranker_info(final_file)
                
                # V2 내용 생성
                v2_content = generate_v2_content(final_file, info)
                
                # V2 파일 경로 생성
                v2_file = final_file.parent / (final_file.stem.replace("_FINAL", "_V2") + ".py")
                
                # V2 파일 작성
                with open(v2_file, 'w', encoding='utf-8') as f:
                    f.write(v2_content)
                
                print(f"✅ 변환 완료: {final_file.name} → {v2_file.name}")
                converted_count += 1
                
            except Exception as e:
                print(f"❌ 변환 실패: {final_file.name} - {e}")
    
    print(f"\n📊 변환 결과: {converted_count}/{total_count} 파일 성공")
    return converted_count == total_count

if __name__ == "__main__":
    success = convert_all_final_files()
    if success:
        print("\n🎉 모든 FINAL 파일들이 V2 인터페이스로 성공적으로 변환되었습니다!")
    else:
        print("\n⚠️ 일부 파일 변환에 실패했습니다.")
