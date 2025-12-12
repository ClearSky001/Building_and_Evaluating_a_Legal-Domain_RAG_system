#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
리트리버 비교 실험 스크립트
여러 리트리버를 순차적으로 테스트하고 결과를 비교합니다.
"""

import subprocess
import sys
import os
import time

def run_retriever_experiment(retriever_id: str, exp_name: str):
    """특정 리트리버로 실험 실행"""
    print(f"\n{'='*60}")
    print(f"🔬 실험 시작: {retriever_id}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, "RAG_with_Retriever.py",
        "--retriever_id", retriever_id,
        "--exp_name", f"{exp_name}_{retriever_id}",
        "--k_ctx", "5",
        "--k_in", "50",
        "--k_dbg", "10",
    ]
    
    if retriever_id == "hybrid_rrf":
        cmd.extend(["--hybrid_weights", "0.5", "0.5"])
    
    try:
        # EOF 문제를 피하기 위해 stdin을 닫음
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        process.stdin.close()
        process.wait()
        
        if process.returncode == 0:
            print(f"✅ {retriever_id} 실험 완료")
        else:
            print(f"❌ {retriever_id} 실험 실패 (return code: {process.returncode})")
    except Exception as e:
        print(f"❌ {retriever_id} 실행 오류: {e}")
    
    time.sleep(2)  # 다음 실험 전 잠시 대기

def main():
    """메인 실행 함수"""
    print("🚀 리트리버 비교 실험 시작")
    print(f"현재 디렉토리: {os.getcwd()}")
    
    # 테스트할 리트리버 목록
    retrievers = [
        "naive_cosine_e5",   # 기본 코사인 유사도
        "bm25",              # BM25 (희소 벡터)
        "tfidf",             # TF-IDF
        "hybrid_rrf",        # 하이브리드 (Dense + Sparse)
    ]
    
    exp_name = f"retriever_comparison_{int(time.time())}"
    
    # 각 리트리버로 실험 실행
    for retriever_id in retrievers:
        run_retriever_experiment(retriever_id, exp_name)
    
    print(f"\n{'='*60}")
    print("🎉 모든 실험 완료!")
    print(f"📊 결과는 exp_outputs/ 디렉토리를 확인하세요")
    print(f"  - retriever_report.csv: 모든 실험의 요약")
    print(f"  - cands_*.jsonl: 각 리트리버의 후보군 덤프")
    print(f"  - {exp_name}_*_config.json: 각 실험의 설정")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()


