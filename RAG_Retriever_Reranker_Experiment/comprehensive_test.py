#!/usr/bin/env python3
"""
모든 리랭커들의 포괄적인 테스트
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

def test_all_rerankers():
    """모든 리랭커들을 테스트"""
    
    print("🚀 전체 리랭커 테스트 시작...")
    print(f"임베딩 파일 경로: {EMB_PATH}")
    print(f"임베딩 파일 존재 여부: {EMB_PATH.exists()}")
    print("="*80)
    
    if not EMB_PATH.exists():
        print("❌ 임베딩 파일이 존재하지 않습니다.")
        return False
    
    # 테스트할 모듈들 정의
    test_modules = [
        # BM25 리랭커들
        ("BM25_Reranker.RAG_BM25_Rerank", "LegalRAGSystemBM25Rerank"),
        ("BM25_Reranker.RAG_BM25_CharNgram_Rerank", "LegalRAGSystemBM25CharNgram"),
        ("BM25_Reranker.RAG_BM25_Kiwi_Rerank", "LegalRAGSystemBM25Kiwi"),
        ("BM25_Reranker.RAG_BM25_Stopword_Rerank", "LegalRAGSystemBM25Stopword"),
        ("BM25_Reranker.RAG_BM25_Regex_Rerank", "LegalRAGSystemBM25Regex"),
        
        # CrossEncoder 리랭커들
        ("CrossEncoder.RAG_CE_MiniLM_L6_Rerank", "LegalRAGSystemMiniLML6"),
        ("CrossEncoder.RAG_CE_MiniLM_L12_Rerank", "LegalRAGSystemMiniLML12"),
        ("CrossEncoder.RAG_CE_Electra_Rerank", "LegalRAGSystemElectraCE"),
        ("CrossEncoder.RAG_CE_E5_Mistral_Rerank", "LegalRAGSystemE5Mistral"),
        
        # BGE 계열 (일부만 테스트)
        ("CrossEncoder.BGE 계열.RAG_BGE_Base_Rerank", "LegalRAGSystemBGEBase"),
        
        # Embedding 리랭커들
        ("Embedding_Reranker.RAG_EmbeddingCosine_E5_Rerank", "LegalRAGSystemEmbeddingE5"),
        ("Embedding_Reranker.RAG_EmbeddingCosine_GTE_Rerank", "LegalRAGSystemEmbeddingGTE"),
        
        # Hybrid 리랭커들 (일부만 테스트)
        ("Hybrid_Reranker.RAG_CombSum_Rerank", "LegalRAGSystemCombSum"),
        
        # LLM 리랭커들 (일부만 테스트)
        ("LLM_Reranker.RAG_LLM_Rerank", "LegalRAGSystemLLMRerank"),
        
        # Rules 리랭커들
        ("Rules_Reranker.RAG_LegalRuleBoost_Rerank", "LegalRAGSystemRuleBoost"),
    ]
    
    results = []
    
    for module_name, expected_class_name in test_modules:
        try:
            print(f"\n🔍 테스트 중: {module_name}")
            
            # 모듈 import
            module = importlib.import_module(module_name)
            
            # 클래스 이름이 정확하지 않을 수 있으므로 동적으로 찾기
            classes = [name for name in dir(module) if name.startswith("LegalRAGSystem")]
            if not classes:
                print(f"❌ {module_name}: LegalRAGSystem 클래스를 찾을 수 없음")
                results.append((module_name, False, "클래스 없음"))
                continue
                
            actual_class_name = classes[0]
            cls = getattr(module, actual_class_name)
            
            print(f"   찾은 클래스: {actual_class_name}")
            
            # 클래스 초기화 테스트 (OpenAI API 키 없이도 가능)
            try:
                rag = cls(embeddings_file=str(EMB_PATH))
                print(f"   ✅ 초기화 성공")
                
                # 문서 검색 테스트 (LLM 호출 없이)
                if hasattr(rag, 'retriever') and rag.retriever:
                    test_query = "종합부동산세법의 목적은 무엇인가요?"
                    relevant_docs = rag.retriever.invoke(test_query)
                    
                    print(f"   ✅ 검색 성공! 검색된 문서 수: {len(relevant_docs)}")
                    if relevant_docs:
                        print(f"   📄 첫 번째 문서 소스: {relevant_docs[0].metadata.get('source', '알 수 없음')}")
                    
                    results.append((module_name, True, None))
                    print(f"   🎉 {module_name} 테스트 완료!")
                else:
                    print(f"   ⚠️ retriever가 없음")
                    results.append((module_name, False, "retriever 없음"))
                    
            except Exception as init_e:
                print(f"   ❌ 초기화 실패: {init_e}")
                results.append((module_name, False, f"초기화 실패: {str(init_e)[:100]}"))
                
        except Exception as e:
            print(f"   ❌ import 실패: {e}")
            results.append((module_name, False, f"import 실패: {str(e)[:100]}"))
    
    # 결과 요약
    print("\n" + "="*80)
    print("📊 전체 테스트 결과 요약:")
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
        print("\n🎉 모든 리랭커가 정상적으로 작동합니다!")
        return True
    else:
        print(f"\n⚠️ {len(results) - success_count}개의 리랭커에서 문제가 발생했습니다.")
        return False

if __name__ == "__main__":
    success = test_all_rerankers()
    exit(0 if success else 1)
