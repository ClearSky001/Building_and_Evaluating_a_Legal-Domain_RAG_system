'''
# eval_retriver.py
(리트리버 평가 함수)

사용법 : retriever_list에 구현한 리트리버 함수명을 넣어주세요.
이때, 각 함수들이 Optional[List[Dict]]형식의 응답을 반환해야 합니다.

동작 : 
1. 정답데이터(question/ground_truths)를 로드
2. 리트리버에 question 넣어 응답을 받음
3. RAGAS로 평가하여 context_precision, context_recall, score(context_precision, context_recall의 평균) 계산
4. 각 리트리버 함수에 대해 1-4수행
5. 결과를 DataFrame으로 출력
'''

import json
from typing import List, Dict, Optional, Callable
from datasets import Dataset as HFDataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall
import numpy as np
import os
import pandas as pd

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ.pop("LANGCHAIN_API_KEY", None)

# retriever: Callable[[str], Optional[List[Dict]]]
def eval_retriver(
    retriever: Callable[[str], Optional[List[Dict]]],
    qa_json_path: str = "data/real_estate_tax_QA.json"
) -> Dict:
    with open(qa_json_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    questions = [item['question'] for item in qa_data[:3]]
    references = [item['ground_truth'] for item in qa_data[:3]]
    contexts = []
    answers = []
    for q in questions:
        retrieved_docs = retriever(q)
        if not retrieved_docs or not isinstance(retrieved_docs, list):
            contexts.append([""])
            answers.append("")
            continue
        page_contents = [doc["page_content"] if isinstance(doc, dict) and "page_content" in doc else str(doc) for doc in retrieved_docs]
        contexts.append(page_contents)
        if "answer" in retrieved_docs[0]:
            answers.append(retrieved_docs[0]["answer"])
        else:
            answers.append(retrieved_docs[0]["page_content"] if "page_content" in retrieved_docs[0] else str(retrieved_docs[0]))
    hf_ds = HFDataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "reference": references
    })
    result = evaluate(
        hf_ds,
        metrics=[context_precision, context_recall]
    )
    cp = result["context_precision"]
    cr = result["context_recall"]
    score = np.mean([cp, cr])
    return {
        "retriever": retriever.__name__,
        "context_precision": cp,
        "context_recall": cr,
        "score": score
    }

# 여러 retriever 함수 평가 및 표 출력

def eval_retrivers(retrievers: List[Callable[[str], Optional[List[Dict]]]], qa_json_path: str = "data/real_estate_tax_QA.json"):
    results = []
    for retriever in retrievers:
        results.append(eval_retriver(retriever, qa_json_path))
    df = pd.DataFrame(results)
    display(df)
    return df

# 사용 예시: 더미 retriever 함수
def dummy_retriever1(question):
    return [
        {"page_content": "아무말아무말"},
        {"page_content": "asdfasdf."},
        {"page_content": "ㄱㄷㄴㅇㄹ;ㅣ나얼;미ㅏㄴ얼."}
    ]

def dummy_retriever2(question):
    return [
        {"page_content": "종합부동산세에서 공제할 재산세는 공시가격에서 기준금액을 뺀 금액에 재산세 공정시장가액비율과 종합부동산세 공정시장가액비율 가운데 작은 비율을 곱하고, 이에 재산세 세율을 적용해 계산해야 함"},
        {"page_content": "2012년에는 공시가격에서 공제금액을 뺀 후 공정시장가액비율과 재산세율을 곱해 산출합니다."},
        {"page_content": "공제 재산세 계산 시 두 공정시장가액비율 중 작은 비율을 적용하고 재산세율을 곱해야 합니다."}
    ]

# 여러 retriever 함수 평가
retriever_list = [dummy_retriever1, dummy_retriever2]
eval_retrivers(retriever_list, "data/real_estate_tax_QA.json")
