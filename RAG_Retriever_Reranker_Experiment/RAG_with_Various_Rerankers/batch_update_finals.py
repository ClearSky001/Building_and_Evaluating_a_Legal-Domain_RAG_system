#!/usr/bin/env python3
"""
모든 FINAL 파일들을 V2 인터페이스로 일괄 업데이트
"""
import os
import re
from pathlib import Path

def update_final_file(file_path: Path) -> bool:
    """FINAL 파일을 V2 인터페이스로 업데이트"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 이미 업데이트된 파일인지 확인
        if 'def rerank_documents(' in content and 'def search_and_rerank(' in content:
            print(f"⏭️ 이미 V2로 업데이트됨: {file_path.name}")
            return True
        
        # 1. Import 수정
        content = content.replace('from fixed_base import (', 'from fixed_base_v2 import (')
        
        # 2. typing import 추가
        if 'from typing import List, Optional, Dict' not in content:
            content = content.replace(
                'from pathlib import Path',
                'from pathlib import Path\nfrom typing import List, Optional, Dict'
            )
        
        # 3. BaseDocumentCompressor → BaseReranker
        content = content.replace('BaseDocumentCompressor,', 'BaseReranker,')
        
        # 4. self.reranker = None 추가
        if 'self.reranker = None' not in content:
            content = content.replace(
                'self.embedding_model = None\n\n        self._setup_environment()',
                'self.embedding_model = None\n        self.reranker = None\n\n        self._setup_environment()'
            )
        
        # 5. 압축기 → 리랭커 변경
        content = re.sub(
            r'(\s+)# (.+) 압축기 생성\n(\s+)compressor = (\w+)\((.*?)\)\n(\s+)\n(\s+)# 압축 리트리버 생성\n(\s+)self\.retriever = SimpleCompressionRetriever\(base_retriever, compressor\)',
            r'\1# \2 리랭커 생성\n\3self.reranker = \4(\5)\n\6\n\7# 압축 리트리버 생성\n\8self.retriever = SimpleCompressionRetriever(base_retriever, self.reranker)',
            content,
            flags=re.MULTILINE
        )
        
        # 6. 새로운 인터페이스 메서드들 추가
        if 'def rerank_documents(' not in content:
            # 모델명 추출
            model_name = "DEFAULT"
            if "BM25" in file_path.name:
                model_name = "BM25"
            elif "CrossEncoder" in file_path.name or "CE_" in file_path.name:
                model_name = "CrossEncoder"
            elif "BGE" in file_path.name:
                model_name = "BGE"
            elif "Embedding" in file_path.name:
                model_name = "Embedding"
            elif "Hybrid" in file_path.name or "Comb" in file_path.name or "RRF" in file_path.name:
                model_name = "Hybrid"
            elif "LLM" in file_path.name:
                model_name = "LLM"
            elif "Rule" in file_path.name:
                model_name = "Rules"
            
            new_methods = f'''
    def rerank_documents(
        self,
        query: str,
        candidate_documents: Optional[List[dict]] = None,
        model: str = "{model_name}"
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
'''
            
            # ask_question 메서드 앞에 삽입
            content = content.replace(
                '    def ask_question(self, question: str, show_sources: bool = True) -> str:',
                new_methods + '\n    def ask_question(self, question: str, show_sources: bool = True) -> str:'
            )
        
        # 7. show_sources 부분 수정
        old_pattern = r'if show_sources:\s*relevant_docs = self\.retriever\.invoke\(question\)\s*print\("📚 \*\*참고한 문서:\*\*"\)\s*for i, doc in enumerate\(relevant_docs, 1\):\s*source = doc\.metadata\.get\(\'source\', \'알 수 없음\'\)\s*print\(f"  \{i\}\. \{source\}"\)\s*print\(\)'
        
        new_show_sources = '''if show_sources:
                # 새로운 인터페이스 사용
                result = self.search_and_rerank(question)
                retrieved_docs = result['retrieved_docs']
                
                print("📚 **참고한 문서:**")
                for i, doc_info in enumerate(retrieved_docs, 1):
                    print(f"  {i}. {doc_info['doc_id']} (점수: {doc_info['score']:.4f})")
                print()'''
        
        content = re.sub(old_pattern, new_show_sources, content, flags=re.MULTILINE | re.DOTALL)
        
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
    
    # 모든 FINAL 파일들 찾기
    all_final_files = []
    for folder in ["BM25_Reranker", "CrossEncoder", "CrossEncoder/BGE 계열", "Embedding_Reranker", "Hybrid_Reranker", "LLM_Reranker", "Rules_Reranker"]:
        folder_path = base_dir / folder
        if folder_path.exists():
            final_files = list(folder_path.glob("*_FINAL.py"))
            all_final_files.extend(final_files)
    
    print(f"📂 찾은 FINAL 파일들: {len(all_final_files)}개")
    
    updated_count = 0
    
    for final_file in all_final_files:
        print(f"\n🔄 업데이트 중: {final_file.relative_to(base_dir)}")
        
        if update_final_file(final_file):
            updated_count += 1
            print(f"✅ 완료")
        else:
            print(f"❌ 실패")
    
    print(f"\n📊 업데이트 결과: {updated_count}/{len(all_final_files)} 파일 성공")
    
    if updated_count == len(all_final_files):
        print("🎉 모든 FINAL 파일들이 V2 인터페이스로 업데이트되었습니다!")
    else:
        print(f"⚠️ {len(all_final_files) - updated_count}개 파일 업데이트 실패")
    
    return updated_count == len(all_final_files)

if __name__ == "__main__":
    main()
