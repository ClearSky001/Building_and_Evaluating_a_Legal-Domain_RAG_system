#!/usr/bin/env python3
"""
완전히 수정된 FINAL 리랭커들 테스트
"""
import sys
import os
from pathlib import Path
import importlib

# 현재 디렉토리를 PATH에 추가
BASE_DIR = Path(__file__).parent
RERANKERS_DIR = BASE_DIR / "RAG_with_Various_Rerankers"
EMB_PATH = BASE_DIR / "output_chunks_with_embeddings.json"

sys.path.insert(0, str(RERANKERS_DIR))

def test_final_rerankers():
    """FINAL 버전 리랭커들을 테스트"""
    
    print("🚀 FINAL 리랭커 테스트 시작...")
    print(f"임베딩 파일 경로: {EMB_PATH}")
    print(f"임베딩 파일 존재 여부: {EMB_PATH.exists()}")
    print("="*80)
    
    if not EMB_PATH.exists():
        print("❌ 임베딩 파일이 존재하지 않습니다.")
        return False
    
    # FINAL 버전 테스트 모듈들
    test_modules = [
        ("BM25_Reranker.RAG_BM25_Rerank_FINAL", "LegalRAGSystemBM25Rerank"),
        ("CrossEncoder.RAG_CE_MiniLM_L6_Rerank_FINAL", "LegalRAGSystemMiniLML6"),
        ("Embedding_Reranker.RAG_EmbeddingCosine_E5_Rerank_FINAL", "LegalRAGSystemEmbeddingE5"),
    ]
    
    results = []
    
    for module_name, expected_class_name in test_modules:
        try:
            print(f"\n🔍 테스트 중: {module_name}")
            
            # 모듈 import
            module = importlib.import_module(module_name)
            
            # 클래스 이름 찾기
            classes = [name for name in dir(module) if name.startswith("LegalRAGSystem")]
            if not classes:
                print(f"❌ {module_name}: LegalRAGSystem 클래스를 찾을 수 없음")
                results.append((module_name, False, "클래스 없음"))
                continue
                
            actual_class_name = classes[0]
            cls = getattr(module, actual_class_name)
            
            print(f"   찾은 클래스: {actual_class_name}")
            
            # 클래스 초기화 테스트
            try:
                rag = cls(embeddings_file=str(EMB_PATH))
                print(f"   ✅ 초기화 성공")
                
                # 문서 검색 테스트
                if hasattr(rag, 'retriever') and rag.retriever:
                    test_query = "종합부동산세법의 목적은 무엇인가요?"
                    relevant_docs = rag.retriever.invoke(test_query)
                    
                    print(f"   ✅ 검색 성공! 검색된 문서 수: {len(relevant_docs)}")
                    if relevant_docs:
                        print(f"   📄 첫 번째 문서 소스: {relevant_docs[0].metadata.get('source', '알 수 없음')}")
                        print(f"   📄 첫 번째 문서 내용: {relevant_docs[0].page_content[:100]}...")
                    
                    results.append((module_name, True, None))
                    print(f"   🎉 {module_name} 테스트 완료!")
                else:
                    print(f"   ⚠️ retriever가 없음")
                    results.append((module_name, False, "retriever 없음"))
                    
            except Exception as init_e:
                print(f"   ❌ 초기화 실패: {init_e}")
                results.append((module_name, False, f"초기화 실패: {str(init_e)[:100]}"))
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            print(f"   ❌ import 실패: {e}")
            results.append((module_name, False, f"import 실패: {str(e)[:100]}"))
    
    # 결과 요약
    print("\n" + "="*80)
    print("📊 FINAL 테스트 결과 요약:")
    print("="*80)
    
    success_count = 0
    for module_name, success, error in results:
        if success:
            print(f"✅ {module_name}")
            success_count += 1
        else:
            print(f"❌ {module_name}: {error}")
    
    print(f"\n📈 성공률: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    
    if success_count == len(results):
        print("\n🎉 모든 FINAL 리랭커가 정상적으로 작동합니다!")
        return True
    else:
        print(f"\n⚠️ {len(results) - success_count}개의 리랭커에서 문제가 발생했습니다.")
        return False

if __name__ == "__main__":
    success = test_final_rerankers()
    exit(0 if success else 1)
