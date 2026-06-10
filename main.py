# pip install --upgrade langchain langchain-community langchain-text-splitters langchain-openai langchain-chroma pypdf python-dotenv
# pip install --upgrade langchain langchain-community langchain-text-splitters langchain-openai langchain-chroma pypdf python-dotenv
# pip install python-dotenv
# pip install 

from dotenv import load_dotenv
# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

#문서 로드 및 텍스트 분할
loader = PyPDFLoader("unsu.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,           # 하나의 청크가 가질 최대 글자 수
    chunk_overlap = 20,         # 청크 간에 겹칠 글자 수 (문맥 단절 방지)
    length_function = len,      # 길이를 측정할 함수 (기본 문자열 길이)
    is_separator_regex = False, # 구분 기호(separator)를 정규표현식으로 해석할지 여부
)

texts = text_splitter.split_documents(pages)

# 임베딩 모델 생성
# OpenAIEmbeddings는 OpenAI의 임베딩 모델을 사용하여 텍스트를 벡터로 변환하는 클래스입니다.
# 필요에 따라 model="text-embedding-3-small" 등의 특정 모델을 지정할 수 있습니다. 
embeddings_model = OpenAIEmbeddings()

# 벡터 데이터베이스 생성
# 분할된 텍스트 청크들을 임베딩 모델을 통해 벡터로 변환하고, Chroma 데이터베이스에 저장합니다.
# from_documents는 임베딩과 저장 과정을 한 번에 처리합니다.
# 현재 코드는 메모리 상에 임시 저장하는 방식입니다.
db = Chroma.from_documents(texts, embeddings_model)