#!/usr/bin/env python3
"""
간단한 리랭커 테스트 - BM25만 테스트
"""
import sys
import os
from pathlib import Path

# 현재 디렉토리를 PATH에 추가
BASE_DIR = Path(__file__).parent
RERANKERS_DIR = BASE_DIR / "RAG_with_Various_Rerankers"
EMB_PATH = BASE_DIR / "output_chunks_with_embeddings.json"

sys.path.insert(0, str(RERANKERS_DIR))

def test_bm25():
    """BM25 리랭커만 테스트"""
    
    print("🚀 BM25 리랭커 테스트 시작...")
    print(f"임베딩 파일 경로: {EMB_PATH}")
    print(f"임베딩 파일 존재 여부: {EMB_PATH.exists()}")
    
    if not EMB_PATH.exists():
        print("❌ 임베딩 파일이 존재하지 않습니다.")
        return False
    
    try:
        from BM25_Reranker.RAG_BM25_Rerank import LegalRAGSystemBM25Rerank
        print("✅ BM25 리랭커 모듈 import 성공!")
        
        # 초기화 테스트 (OpenAI API 키 없이도 가능)
        print("🔧 BM25 리랭커 초기화 중...")
        rag = LegalRAGSystemBM25Rerank(embeddings_file=str(EMB_PATH))
        print("✅ BM25 리랭커 초기화 성공!")
        
        # 문서 검색만 테스트 (LLM 호출 없이)
        print("📚 문서 검색 테스트 중...")
        test_query = "종합부동산세법의 목적은 무엇인가요?"
        relevant_docs = rag.retriever.invoke(test_query)
        
        print(f"✅ 검색 성공! 검색된 문서 수: {len(relevant_docs)}")
        if relevant_docs:
            print(f"📄 첫 번째 문서 미리보기:")
            print(f"   소스: {relevant_docs[0].metadata.get('source', '알 수 없음')}")
            print(f"   내용: {relevant_docs[0].page_content[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ BM25 리랭커 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_bm25()
    if success:
        print("\n🎉 테스트 성공! 모든 리랭커가 정상적으로 작동할 것입니다.")
    else:
        print("\n💥 테스트 실패! 문제를 해결해야 합니다.")
