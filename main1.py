# pip install --upgrade langchain langchain-community langchain-text-splitters langchain-openai langchain-chroma pypdf python-dotenv
# pip install langchain 

import os
from turtle import st
from dotenv import load_dotenv
import streamlit as st
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

loader = PyPDFLoader("unsu.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,           # 하나의 청크가 가질 최대 글자 수
    chunk_overlap  = 20,        # 청크 간 문맥 연결을 위해 겹칠 글자 수
    length_function = len,      # 길이 측정 기준 (기본 문자열 길이)
    is_separator_regex = False, # 구분 기호의 정규표현식 해석 여부
)
texts = text_splitter.split_documents(pages)
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma.from_documents(texts, embeddings_model)

# 기존 main.py 이하 추가영역
# 멀티 쿼리 리트리버 생성 및 LLM 설정
# 모델 명시적으로 지정할 수 있다. 예: model="gpt-4o-mini" 모델이 가성비가 좋다
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 사용자의 질문을 다양한 각도에서 재해석하여 검색 확률을 높이는 MultiQueryRetriever를 생성합니다.
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=db.as_retriever(), 
    llm=llm
)

# RAG 체인 구성
# LLM에게 전달할 프롬프트(지시문)을 정의함
system_prompt = (
    "너는 질문-답변을 돕는 유능한 비서야. "
    "아래 제공된 맥락(context)만을 사용하여 질문에 답해줘. "
    "답변을 문서내용에 충실하게 이해해서 답변을 구성해줘. "
    "답을 모르면 모른다고 하고, 절대 답변을 지어내지 마.\n\n"

    "{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])
# system은 LLM에게 역할과 지시사항을 전달하는 부분입니다. human은 사용자의 질문이 들어가는 부분입니다.

# 검색된 문서들을 활용하여 질문에 답변하는 체인을 생성합니다.
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# CROMA 데이터베이스에서 검색된 문서들을 활용하여 질문에 답변하는 RAG 체인을 생성합니다.
# retriever = db.as_retriever() # 추가 question = "뺨을 몇번 때렸는지?" 추가 후 반영

# RAG 체인 생성: 검색과 질문-답변 체인을 결합하여 최종 RAG 체인을 만듦
rag_chain = create_retrieval_chain(retriever_from_llm, question_answer_chain)

# 질문 실행(question) 및 결과 출력(response)
# question = "아내가 먹고 싶어하는 음식은 무엇이야?"
# question = "아내가 같이 이야기한 음식은 무엇이야?"
# question = "뺨을 몇번 때렸는지?"
# question = "뺨을 때렸는지?"


# --- UI 설정 ---
st.set_page_config(page_title="운수좋은날 질문&답변 기능", page_icon="📖")
st.title("운수좋은날 질문&답변 기능")
st.markdown("운수좋은날 내용에 대해서 문의 해보세요")

# --- 메인 화면: URL 입력 ---
question = st.text_input("질문을 입력하세요.", placeholder="질문을 입력해 주세요...")

if st.button("확인요청"):
    if not question:
        st.warning("⚠️ 질문을 입력해 주세요.")
        st.stop()

    with st.spinner("질문에 답변하는 중입니다... 잠시만 기다려주세요! ⏳"):
        response = rag_chain.invoke({"input": question})
        st.markdown(response ["answer"])


# response = rag_chain.invoke({"input": question})

# 결과 출력
# st.success("✅ 정리 완료되었습니다!")
            # st.subheader(f"📺 영상 제목: {video_title}")
            # st.markdown("---")
# st.markdown(response) 삭제 예정
            

# # 결과 출력
# print(f"검색된 참조 문서 개수: {len(response.get('context', []))}")
# print(f"답변: {response['answer']}")

# print("--- [최종 답변] ---")
# print(response['answer'])