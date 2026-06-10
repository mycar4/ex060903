import os
import streamlit as st
from dotenv import load_dotenv

# 올바른 Langchain 모듈 임포트
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


from langchain_core.prompts import ChatPromptTemplate

# 환경변수 로드 (.env 파일에서 OPENAI_API_KEY 가져오기)
load_dotenv()

# --- 1. Streamlit 페이지 설정 (반드시 최상단에 위치) ---
st.set_page_config(page_title="운수 좋은 날 Q&A", page_icon="📖", layout="centered")

# --- 2. RAG 체인 초기화 및 캐싱 ---
# @st.cache_resource를 사용해 앱이 재실행되어도 DB 생성 작업은 최초 1회만 수행하도록 설정합니다.
@st.cache_resource(show_spinner="📄 문서를 분석하고 데이터베이스를 구축하는 중입니다...")
def initialize_rag_chain():
    # PDF 파일 존재 여부 확인
    if not os.path.exists("unsu.pdf"):
        st.error("오류: 현재 디렉토리에 'unsu.pdf' 파일이 없습니다. 파일을 업로드해주세요.")
        st.stop()

    # 1. 문서 로드 및 분할
    loader = PyPDFLoader("unsu.pdf")
    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,           # 하나의 청크가 가질 최대 글자 수
        chunk_overlap=20,         # 청크 간 문맥 연결을 위해 겹칠 글자 수
        length_function=len,      # 길이 측정 기준
        is_separator_regex=False, 
    )
    texts = text_splitter.split_documents(pages)

    # 2. 임베딩 모델 및 벡터 DB(Chroma) 생성
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma.from_documents(texts, embeddings_model)

    # 3. LLM 설정 (가성비 좋은 gpt-4o-mini 사용)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 4. MultiQueryRetriever 생성 (질문을 다각도로 재해석)
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=db.as_retriever(), 
        llm=llm
    )

    # 5. 프롬프트 설정
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

    # 6. RAG 체인 생성 및 반환
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever_from_llm, question_answer_chain)
    
    return rag_chain

# 체인 로드 실행
rag_chain = initialize_rag_chain()


# --- 3. UI 화면 구성 ---
st.title("📖 운수 좋은 날 질문&답변")
st.markdown("현진건의 소설 **'운수 좋은 날'** 내용에 대해서 무엇이든 문의해 보세요!")
st.divider()

# Streamlit Form을 사용하여 엔터키나 버튼 클릭 시 한 번에 실행되도록 구성
with st.form("qa_form"):
    question = st.text_input(
        "질문을 입력하세요.", 
        placeholder="예: 아내가 먹고 싶어하던 음식은 무엇이야?"
    )
    submitted = st.form_submit_button("확인요청 🚀")

# --- 4. 질문 실행 및 결과 출력 ---
if submitted:
    if not question.strip():
        st.warning("⚠️ 질문을 입력해 주세요.")
    else:
        with st.spinner("문서를 검색하여 답변을 생성하고 있습니다... 잠시만 기다려주세요! ⏳"):
            try:
                # LLM에 질문 전달 및 답변 생성
                response = rag_chain.invoke({"input": question})
                
                # 결과 출력
                st.success("✅ 답변 생성이 완료되었습니다!")
                
                # 답변을 눈에 띄게 박스 형태로 출력
                st.info(response["answer"])
                
                # (선택) 참조한 문서가 몇 개인지 디버깅 용도로 표시하고 싶다면 아래 주석을 해제하세요.
                # with st.expander("참조된 문서 청크 확인하기"):
                #     st.write(f"총 {len(response.get('context', []))}개의 문서 조각을 참고했습니다.")
                
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")