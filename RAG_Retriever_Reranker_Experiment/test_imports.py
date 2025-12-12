#!/usr/bin/env python3
"""
수정된 리랭커들의 import 테스트
"""
import sys
import os
from pathlib import Path

# 현재 디렉토리를 PATH에 추가
BASE_DIR = Path(__file__).parent
RERANKERS_DIR = BASE_DIR / "RAG_with_Various_Rerankers"
sys.path.insert(0, str(RERANKERS_DIR))

def test_imports():
    """모든 리랭커 모듈들의 import 테스트"""
    
    test_modules = [
        # BM25 리랭커들
        ("BM25_Reranker.RAG_BM25_Rerank", "LegalRAGSystemBM25Rerank"),
        
        # CrossEncoder 리랭커들
        ("CrossEncoder.RAG_CE_MiniLM_L6_Rerank", "LegalRAGSystemMiniLML6"),
        ("CrossEncoder.RAG_CE_MiniLM_L12_Rerank", "LegalRAGSystemMiniLML12"),
        ("CrossEncoder.RAG_CE_Electra_Rerank", "LegalRAGSystemElectraCE"),
        ("CrossEncoder.RAG_CE_E5_Mistral_Rerank", "LegalRAGSystemE5Mistral"),
        
        # BGE 계열
        ("CrossEncoder.BGE 계열.RAG_BGE_Base_Rerank", "LegalRAGSystemBGEBase"),
        
        # Embedding 리랭커들
        ("Embedding_Reranker.RAG_EmbeddingCosine_E5_Rerank", "LegalRAGSystemEmbeddingE5"),
        
        # Hybrid 리랭커들
        ("Hybrid_Reranker.RAG_CombSum_Rerank", "LegalRAGSystemCombSum"),
        
        # LLM 리랭커들
        ("LLM_Reranker.RAG_LLM_Rerank", "LegalRAGSystemLLMRerank"),
        
        # Rules 리랭커들
        ("Rules_Reranker.RAG_LegalRuleBoost_Rerank", "LegalRAGSystemRuleBoost"),
    ]
    
    results = []
    
    for module_name, class_name in test_modules:
        try:
            print(f"테스트 중: {module_name}")
            module = __import__(module_name, fromlist=[class_name])
            
            # 클래스 이름이 정확하지 않을 수 있으므로 동적으로 찾기
            classes = [name for name in dir(module) if name.startswith("LegalRAGSystem")]
            if classes:
                actual_class_name = classes[0]
                cls = getattr(module, actual_class_name)
                print(f"✅ {module_name} ({actual_class_name}) import 성공!")
                results.append((module_name, True, None))
            else:
                print(f"❌ {module_name}: LegalRAGSystem 클래스를 찾을 수 없음")
                results.append((module_name, False, "클래스 없음"))
                
        except Exception as e:
            print(f"❌ {module_name} import 오류: {e}")
            results.append((module_name, False, str(e)))
    
    # 결과 요약
    print("\n" + "="*60)
    print("📊 테스트 결과 요약:")
    print("="*60)
    
    success_count = 0
    for module_name, success, error in results:
        if success:
            print(f"✅ {module_name}")
            success_count += 1
        else:
            print(f"❌ {module_name}: {error}")
    
    print(f"\n총 {len(results)}개 중 {success_count}개 성공 ({success_count/len(results)*100:.1f}%)")
    
    return results

if __name__ == "__main__":
    test_imports()
