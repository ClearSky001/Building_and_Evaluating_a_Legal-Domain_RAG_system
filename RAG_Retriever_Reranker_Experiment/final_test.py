#!/usr/bin/env python3
"""
FINAL 리랭커들의 간단한 테스트
"""
import sys
import os
from pathlib import Path

# 현재 디렉토리를 PATH에 추가
BASE_DIR = Path(__file__).parent
RERANKERS_DIR = BASE_DIR / "RAG_with_Various_Rerankers"
EMB_PATH = BASE_DIR / "output_chunks_with_embeddings.json"

sys.path.insert(0, str(RERANKERS_DIR))

def main():
    print("🚀 FINAL 리랭커들 간단 테스트")
    print(f"임베딩 파일: {EMB_PATH.exists()}")
    print("="*50)
    
    # BM25 FINAL 테스트
    try:
        print("\n🔍 BM25 FINAL 테스트...")
        from BM25_Reranker.RAG_BM25_Rerank_FINAL import LegalRAGSystemBM25Rerank
        rag = LegalRAGSystemBM25Rerank(embeddings_file=str(EMB_PATH))
        docs = rag.retriever.invoke("종합부동산세법의 목적은?")
        print(f"✅ BM25 성공! 문서 수: {len(docs)}")
    except Exception as e:
        print(f"❌ BM25 실패: {e}")
    
    # CrossEncoder FINAL 테스트
    try:
        print("\n🔍 CrossEncoder FINAL 테스트...")
        from CrossEncoder.RAG_CE_MiniLM_L6_Rerank_FINAL import LegalRAGSystemMiniLML6
        rag = LegalRAGSystemMiniLML6(embeddings_file=str(EMB_PATH))
        docs = rag.retriever.invoke("부동산세 세율은?")
        print(f"✅ CrossEncoder 성공! 문서 수: {len(docs)}")
    except Exception as e:
        print(f"❌ CrossEncoder 실패: {e}")
    
    # Embedding FINAL 테스트
    try:
        print("\n🔍 Embedding FINAL 테스트...")
        from Embedding_Reranker.RAG_EmbeddingCosine_E5_Rerank_FINAL import LegalRAGSystemEmbeddingE5
        rag = LegalRAGSystemEmbeddingE5(embeddings_file=str(EMB_PATH))
        docs = rag.retriever.invoke("부동산 취득세는?")
        print(f"✅ Embedding 성공! 문서 수: {len(docs)}")
    except Exception as e:
        print(f"❌ Embedding 실패: {e}")
    
    print("\n🎉 테스트 완료!")

if __name__ == "__main__":
    main()

