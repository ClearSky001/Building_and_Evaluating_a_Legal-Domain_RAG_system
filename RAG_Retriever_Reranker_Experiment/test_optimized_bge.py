"""
최적화된 BGE Base 리랭커 테스트 스크립트
지연 초기화와 메모리 효율성 검증
"""

import sys
import os
import time
from pathlib import Path

# 프로젝트 경로 설정
BASE_DIR = Path.cwd()
RERANKERS_DIR = BASE_DIR / "RAG_with_Various_Rerankers"
sys.path.insert(0, str(RERANKERS_DIR))

def test_optimized_bge_reranker():
    """최적화된 BGE Base 리랭커 테스트"""
    print("🧪 최적화된 BGE Base 리랭커 테스트 시작")
    print("=" * 60)
    
    try:
        # 모듈 import (한글 경로 문제 해결)
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "bge_optimized", 
            RERANKERS_DIR / "CrossEncoder" / "BGE 계열" / "RAG_BGE_Base_Optimized_Rerank_FINAL.py"
        )
        bge_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bge_module)
        LegalRAGSystemBGEBase = bge_module.LegalRAGSystemBGEBase
        
        print("✅ 모듈 import 성공")
        
        # 시스템 생성 (지연 초기화)
        print("\n🔧 시스템 생성 중...")
        start_time = time.time()
        rag_system = LegalRAGSystemBGEBase()
        creation_time = time.time() - start_time
        print(f"✅ 시스템 생성 완료: {creation_time:.2f}초 (지연 초기화)")
        
        # 테스트 질문
        test_question = "부동산 취득세는 언제 내야 하나요?"
        
        print(f"\n🤖 테스트 질문: {test_question}")
        print("-" * 50)
        
        # 실제 사용 시 초기화 시작
        print("🚀 실제 사용 시작 - 시스템 초기화 중...")
        start_time = time.time()
        
        # 검색 테스트
        docs = rag_system.search(test_question, k=5)
        search_time = time.time() - start_time
        
        print(f"✅ 검색 완료: {len(docs)}개 문서, {search_time:.2f}초")
        
        # 문서 내용 확인
        if docs:
            print("\n📚 검색된 문서들:")
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source', '알 수 없음')
                content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                print(f"  {i}. {source}")
                print(f"     내용: {content_preview}")
                print()
        
        print("🎉 최적화된 BGE Base 리랭커 테스트 성공!")
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_memory_usage():
    """메모리 사용량 확인"""
    try:
        import torch
        print("\n📊 메모리 사용량:")
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                print(f"GPU {i}: {memory_allocated:.2f}GB 할당됨, {memory_reserved:.2f}GB 예약됨")
        else:
            print("GPU를 사용할 수 없습니다 (CPU 모드)")
            
    except ImportError:
        print("PyTorch가 설치되지 않았습니다.")

if __name__ == "__main__":
    print("🚀 최적화된 BGE Base 리랭커 테스트 시작")
    print(f"작업 디렉토리: {BASE_DIR}")
    print(f"리랭커 디렉토리: {RERANKERS_DIR}")
    
    # 메모리 사용량 확인 (테스트 전)
    check_memory_usage()
    
    # 테스트 실행
    success = test_optimized_bge_reranker()
    
    # 메모리 사용량 확인 (테스트 후)
    check_memory_usage()
    
    if success:
        print("\n✅ 모든 테스트 통과! 최적화된 BGE Base 리랭커 사용 준비 완료")
    else:
        print("\n❌ 테스트 실패. 문제를 해결한 후 다시 시도하세요.")
