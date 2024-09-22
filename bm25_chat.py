import pandas as pd
import torch
import numpy as np

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from kiwipiepy import Kiwi

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.vectorstores import FAISS
from langchain.schema import Document


kiwi = Kiwi()

# 형태소 분석을 통해 문서를 토큰화하는 함수
def kiwi_tokenize(text):
    return [token.form for token in kiwi.tokenize(text)]

# 질문과 답변 리스트 생성
questions = ["우리나라 정부 형태가 뭐야?"]
int_sentences = ["우리나라 정부 형태에 대해 알려주세요."]
int_responses = ["우리나의 정부 형태는 의원내각제적 요소를 가미한 대통령제 입니다."]

# 문서와 쿼리를 토큰화
tokenized_questions = [kiwi_tokenize(question) for question in questions]

documents = [Document(page_content=question, metadata={'index' : i}) for i, question in enumerate(questions)]

# BM25 모델 설정
kiwi_bm25 = BM25Retriever.from_documents(documents, preprocess_func=kiwi_tokenize)

# FAISS 설정
hf_model_name = "intfloat/multilingual-e5-large" 
hf_embeddings = HuggingFaceEmbeddings(model_name=hf_model_name)
faiss = FAISS.from_documents(documents, hf_embeddings).as_retriever()

# EnsembleRetriever에서 검색 수행
retriever = EnsembleRetriever(
    retrievers=[kiwi_bm25, faiss],
    weights=[0.3, 0.7],
    search_type="mmr",
)

query = "우리나라 정부 형태가 뭐야?"

results = retriever.invoke(query)

# 중복되지 않은 int_sentence를 저장할 set
unique_sentences = set()
top_results = []

for result in results:
    index = result.metadata['index']
    if int_sentences[index] not in unique_sentences:
        unique_sentences.add(int_sentences[index])
        top_results.append(result)
    if len(top_results) == 3:
        break

print("사용자 질문:", query)
print("상의 3개 결과:")

for result in top_results:
    index = result.metadata['index']
    print(f"질문: {int_sentences[index]}")
    print(f"답변: {int_responses[index]}")
    print()
    
# 문서와 쿼리를 토큰화
tokenized_questions = [kiwi_tokenize(question) for question in questions]

documents = [Document(page_content=question, metadata={'index':i}) for i, question in enumerate(questions)]



print("test")