#!/usr/bin/env python3
"""
모든 FINAL 파일들을 V2 인터페이스로 자동 업데이트하는 스크립트
"""
import os
import re
from pathlib import Path
from typing import List

def update_file_to_v2(file_path: Path) -> bool:
    """개별 파일을 V2 인터페이스로 업데이트"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 1. Import 수정
        content = content.replace('from fixed_base import (', 'from fixed_base_v2 import (')
        
        # 2. BaseDocumentCompressor → BaseReranker 변경
        content = content.replace('BaseDocumentCompressor,', 'BaseReranker,')
        
        # 3. self.reranker = None 추가 (없는 경우에만)
        if 'self.reranker = None' not in content:
            content = content.replace(
                'self.embedding_model = None\n\n        self._setup_environment()',
                'self.embedding_model = None\n        self.reranker = None\n\n        self._setup_environment()'
            )
        
        # 4. 리랭커 설정 부분 수정
        # compressor → self.reranker로 변경
        content = re.sub(
            r'compressor = (\w+)\((.*?)\)',
            r'self.reranker = \1(\2)',
            content
        )
        
        # 5. SimpleCompressionRetriever 인자 수정
        content = content.replace(
            'SimpleCompressionRetriever(base_retriever, compressor)',
            'SimpleCompressionRetriever(base_retriever, self.reranker)'
        )
        
        # 6. 새로운 인터페이스 메서드들 추가
        new_methods = '''
    def rerank_documents(
        self,
        query: str,
        candidate_documents: Optional[List[dict]] = None,
        model: str = "DEFAULT"
    ) -> dict:
        """
        새로운 표준 인터페이스 - 리랭킹만 수행
        
        Args:
            query (str): 사용자 질문/검색어
            candidate_documents (Optional[List[dict]]): 후보 문서들
            model (str): 리랭커 모델/방식 (테스트용)
            
        Returns:
            dict: {'retrieved_docs': [{'doc_id': str, 'chunk_index': int, 'score': float, 'filename': str, 'text': str}, ...]}
        """
        if candidate_documents is None:
            # 후보 문서가 없으면 기본 검색 수행
            candidate_documents = self.retriever.base_retriever.get_candidate_documents(query)
        
        return self.reranker.rerank_documents(query, candidate_documents)

    def search_and_rerank(self, query: str) -> dict:
        """검색 + 리랭킹을 함께 수행하는 메서드"""
        return self.retriever.search_and_rerank(query)
'''
        
        # ask_question 메서드 앞에 새로운 메서드들 삽입
        if 'def rerank_documents(' not in content:
            content = content.replace(
                '    def ask_question(self, question: str, show_sources: bool = True) -> str:',
                new_methods + '\n    def ask_question(self, question: str, show_sources: bool = True) -> str:'
            )
        
        # 7. ask_question 메서드의 show_sources 부분 수정
        old_show_sources = '''if show_sources:
                relevant_docs = self.retriever.invoke(question)
                print("📚 **참고한 문서:**")
                for i, doc in enumerate(relevant_docs, 1):
                    source = doc.metadata.get('source', '알 수 없음')
                    print(f"  {i}. {source}")
                print()'''
                
        new_show_sources = '''if show_sources:
                # 새로운 인터페이스 사용
                result = self.search_and_rerank(question)
                retrieved_docs = result['retrieved_docs']
                
                print("📚 **참고한 문서:**")
                for i, doc_info in enumerate(retrieved_docs, 1):
                    print(f"  {i}. {doc_info['doc_id']} (점수: {doc_info['score']:.4f})")
                print()'''
        
        content = content.replace(old_show_sources, new_show_sources)
        
        # 8. typing import 추가
        if 'from typing import List, Optional' not in content:
            content = content.replace(
                'from typing import List, Optional',
                'from typing import List, Optional, Dict'
            )
        
        # 파일 저장
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return True
        
    except Exception as e:
        print(f"❌ 업데이트 실패: {file_path.name} - {e}")
        return False

def main():
    """모든 FINAL 파일들을 V2로 업데이트"""
    base_dir = Path(__file__).parent
    
    # 업데이트할 폴더들
    folders = [
        "BM25_Reranker",
        "CrossEncoder", 
        "CrossEncoder/BGE 계열",
        "Embedding_Reranker",
        "Hybrid_Reranker",
        "LLM_Reranker",
        "Rules_Reranker"
    ]
    
    updated_count = 0
    total_count = 0
    
    for folder in folders:
        folder_path = base_dir / folder
        if not folder_path.exists():
            continue
            
        print(f"\n📁 {folder} 폴더 처리 중...")
        
        # FINAL 파일들 찾기
        final_files = list(folder_path.glob("*_FINAL.py"))
        
        for final_file in final_files:
            # 이미 업데이트된 파일 건너뛰기
            if final_file.name == "RAG_BM25_Rerank_FINAL.py":
                print(f"⏭️ 이미 업데이트됨: {final_file.name}")
                updated_count += 1
                total_count += 1
                continue
                
            total_count += 1
            
            if update_file_to_v2(final_file):
                print(f"✅ 업데이트 완료: {final_file.name}")
                updated_count += 1
            else:
                print(f"❌ 업데이트 실패: {final_file.name}")
    
    print(f"\n📊 업데이트 결과: {updated_count}/{total_count} 파일 성공")
    return updated_count == total_count

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 모든 FINAL 파일들이 V2 인터페이스로 성공적으로 업데이트되었습니다!")
    else:
        print("\n⚠️ 일부 파일 업데이트에 실패했습니다.")
