#!/usr/bin/env python3
"""
모든 리랭커 파일들의 import 문제를 일괄 수정하는 스크립트
"""
import os
import re
from pathlib import Path

def fix_file_imports(file_path: Path):
    """개별 파일의 import 문제를 수정"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 기존 잘못된 import들 제거
        content = re.sub(r'from langchain\.retrievers\.document_compressors import BaseDocumentCompressor\n', '', content)
        content = re.sub(r'from langchain_core\.retrievers\.document_compressors import BaseDocumentCompressor\n', '', content)
        content = re.sub(r'from langchain_community\.document_transformers import SentenceTransformerRerank\n', '', content)
        
        # 공통 base_classes import 추가 (중복 방지)
        if 'from base_classes import' not in content:
            # ContextualCompressionRetriever import 다음에 추가
            if 'from langchain.retrievers import ContextualCompressionRetriever' in content:
                content = content.replace(
                    'from langchain.retrievers import ContextualCompressionRetriever',
                    'from langchain.retrievers import ContextualCompressionRetriever\nfrom base_classes import BaseDocumentCompressor, SentenceTransformerRerank'
                )
            else:
                # 적절한 위치에 추가
                lines = content.split('\n')
                insert_idx = 0
                for i, line in enumerate(lines):
                    if line.startswith('from langchain') or line.startswith('import'):
                        insert_idx = i + 1
                lines.insert(insert_idx, 'from base_classes import BaseDocumentCompressor, SentenceTransformerRerank')
                content = '\n'.join(lines)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print(f"✅ 수정 완료: {file_path.name}")
        return True
        
    except Exception as e:
        print(f"❌ 수정 실패: {file_path.name} - {e}")
        return False

def main():
    """모든 리랭커 파일들을 수정"""
    base_dir = Path(__file__).parent
    
    # 수정할 폴더들
    folders = [
        "BM25_Reranker",
        "CrossEncoder",
        "CrossEncoder/BGE 계열",
        "Embedding_Reranker", 
        "Hybrid_Reranker",
        "LLM_Reranker",
        "Rules_Reranker"
    ]
    
    success_count = 0
    total_count = 0
    
    for folder in folders:
        folder_path = base_dir / folder
        if folder_path.exists():
            print(f"\n📁 {folder} 폴더 처리 중...")
            
            for py_file in folder_path.glob("*.py"):
                total_count += 1
                if fix_file_imports(py_file):
                    success_count += 1
    
    print(f"\n📊 수정 완료: {success_count}/{total_count} 파일")
    
if __name__ == "__main__":
    main()
