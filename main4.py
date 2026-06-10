import os
import streamlit as st
import tempfile
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

# --- 1. Streamlit 페이지 설정 ---
st.set_page_config(page_title="나만의 문서 Q&A 챗봇", page_icon="📂", layout="centered")

# --- 2. RAG 체인 초기화 및 캐싱 ---
# 파일의 바이트 데이터와 이름을 기반으로 캐싱합니다. 
# 새로운 파일이 업로드되면 캐시가 갱신되어 새로운 DB를 구축합니다.
@st.cache_resource(show_spinner="📄 업로드된 문서를 분석하고 데이터베이스를 구축하는 중입니다...")
def initialize_rag_chain(file_bytes, file_name):
    # Streamlit이 읽은 바이너리 데이터를 임시 파일로 디스크에 저장합니다. (PyPDFLoader 요구사항)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_file_path = tmp_file.name

    try:
        # 1. 문서 로드 및 분할
        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load_and_split()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,           
            chunk_overlap=20,         
            length_function=len,      
            is_separator_regex=False, 
        )
        texts = text_splitter.split_documents(pages)

        # 2. 임베딩 모델 및 벡터 DB(Chroma) 생성
        embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
        db = Chroma.from_documents(texts, embeddings_model)

        # 3. LLM 설정 (gpt-4o-mini)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # 4. MultiQueryRetriever 생성 (다각도 질문 재해석 검색기)
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

    finally:
        # DB 구축이 완료되면 생성했던 임시 파일은 안전하게 삭제합니다.
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)


# --- 3. UI 화면 구성 ---
st.title("📂 문서 기반 질문&답변 시스템")
st.markdown("원하는 PDF 문서를 업로드하고, 문서 내용에 대해 자유롭게 질문해 보세요!")
st.divider()

# 파일 업로더 컴포넌트 추가
uploaded_file = st.file_uploader(
    "질문하고 싶은 PDF 파일을 업로드 해주세요. (해당 문서를 바탕으로 답변합니다.)", 
    type=["pdf"]
)

# 파일이 업로드되었을 때만 질문창이 활성화되도록 제어
if uploaded_file is not None:
    # 업로드된 파일의 바이트 및 파일명 추출
    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name
    
    # RAG 체인 생성 및 로드 (동일 파일은 캐싱 처리됨)
    rag_chain = initialize_rag_chain(file_bytes, file_name)
    
    st.success(f"✅ '{file_name}' 문서 분석 완료! 이제 아래에서 질문을 입력하세요.")
    st.divider()

    # 검색 폼 구성
    with st.form("qa_form"):
        question = st.text_input(
            "질문을 입력하세요.", 
            placeholder="예: 이 문서의 핵심 요약이나 주요 내용을 알려줘."
        )
        submitted = st.form_submit_button("확인요청 🚀")

    # --- 4. 질문 실행 및 결과 출력 ---
    if submitted:
        if not question.strip():
            st.warning("⚠️ 질문을 입력해 주세요.")
        else:
            with st.spinner("문서를 검색하여 답변을 생성하고 있습니다... 잠시만 기다려주세요! ⏳"):
                try:
                    # RAG 체인 실행
                    response = rag_chain.invoke({"input": question})
                    
                    st.success("✅ 답변 생성이 완료되었습니다!")
                    
                    # 1. 최종 답변 출력 영역
                    st.subheader("🤖 AI 답변")
                    st.info(response["answer"])
                    
                    st.divider()
                    
                    # 2. 답변을 찾는 과정 및 문서 개수 출력 영역
                    context_docs = response.get('context', [])
                    
                    st.subheader("🔍 답변 생성 과정 및 참조 정보")
                    st.markdown(f"📚 **참고한 문서 조각(Chunk) 개수:** `{len(context_docs)}개`")
                    
                    with st.expander("📄 AI가 답변을 찾기 위해 분석한 본문 내용 확인하기"):
                        st.markdown("LLM이 아래의 본문 조각들을 바탕으로 문맥을 이해하고 답변을 재구성했습니다.")
                        
                        for i, doc in enumerate(context_docs):
                            # 메타데이터에서 페이지 번호 가져오기 (+1 처리)
                            page_num = doc.metadata.get('page', 0) + 1
                            
                            st.markdown(f"**[참조 내용 {i+1}] — 업로드 파일 {page_num}페이지에서 검색됨**")
                            st.info(doc.page_content)
                            st.caption("---")
                            
                except Exception as e:
                    st.error(f"오류가 발생했습니다: {e}")
else:
    # 파일을 아직 업로드하지 않았을 때의 가이드 메시지
    st.info("💡 서비스를 시작하려면 먼저 질문할 PDF 문서를 업로드해 주세요.")