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

# --- 1. Streamlit 페이지 설정 ---
st.set_page_config(page_title="운수 좋은 날 Q&A", page_icon="📖", layout="centered")

# --- 2. RAG 체인 초기화 및 캐싱 ---
@st.cache_resource(show_spinner="📄 문서를 분석하고 데이터베이스를 구축하는 중입니다...")
def initialize_rag_chain():
    if not os.path.exists("unsu.pdf"):
        st.error("오류: 현재 디렉토리에 'unsu.pdf' 파일이 없습니다. 파일을 업로드해주세요.")
        st.stop()

    # 1. 문서 로드 및 분할
    loader = PyPDFLoader("unsu.pdf")
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

# 체인 로드 실행
rag_chain = initialize_rag_chain()


# --- 3. UI 화면 구성 ---
st.title("📖 운수 좋은 날 질문&답변")
st.markdown("현진건의 소설 **'운수 좋은 날'** 내용에 대해서 무엇이든 문의해 보세요!")
st.divider()

# 검색 폼 구성
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
                # LLM Chain 실행 (결과에 answer와 context가 함께 포함됨)
                response = rag_chain.invoke({"input": question})
                
                st.success("✅ 답변 생성이 완료되었습니다!")
                
                # [추가] 1. 최종 답변 출력 영역
                st.subheader("🤖 AI 답변")
                st.info(response["answer"])
                
                st.divider()
                
                # [추가] 2. 답변을 찾는 과정 및 문서 개수 출력 영역
                context_docs = response.get('context', [])
                
                st.subheader("🔍 답변 생성 과정 및 참조 정보")
                # 참고한 문서 개수 표시
                st.markdown(f"📚 **참고한 문서 조각(Chunk) 개수:** `{len(context_docs)}개`")
                
                # 참고한 구체적인 문맥 내용을 Expander로 열어볼 수 있게 구성
                with st.expander("📄 AI가 답변을 찾기 위해 분석한 소설 본문 확인하기"):
                    st.markdown("LLM이 아래의 본문 조각들을 바탕으로 문맥을 이해하고 답변을 재구성했습니다.")
                    
                    for i, doc in enumerate(context_docs):
                        # PDF의 페이지 번호 추출 (기본이 0부터 시작하므로 +1 해줍니다)
                        page_num = doc.metadata.get('page', 0) + 1
                        
                        st.markdown(f"**[참조 내용 {i+1}] — PDF {page_num}페이지에서 검색됨**")
                        # 인용구 형태로 본문 텍스트 출력
                        st.info(doc.page_content)
                        st.caption("---")
                        
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")