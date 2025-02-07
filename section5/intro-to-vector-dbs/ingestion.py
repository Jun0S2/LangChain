# Document loaders can even load notion db !
# https://python.langchain.com/v0.1/docs/integrations/document_loaders/

# 1. 텍스트 문서를 불러온 후, 1000자 단위로 분할
# 2. 각 분할된 텍스트에 대해 OpenAI 임베딩을 생성
# 3. 생성된 임베딩을 Pinecone 벡터 데이터베이스에 저장
import os
from dotenv import load_dotenv  # .env 파일에서 환경 변수를 불러오기 위해 사용
from langchain_community.document_loaders import TextLoader  # 텍스트 문서를 불러오는 모듈
from langchain_text_splitters import CharacterTextSplitter  # 문서를 분할하기 위한 모듈
from langchain_openai import OpenAIEmbeddings  # OpenAI의 임베딩 기능을 사용하기 위한 모듈
from langchain_pinecone import PineconeVectorStore  # Pinecone에 벡터를 저장하기 위한 모듈

# .env 파일에 저장된 환경 변수 불러오기 (예: OPENAI_API_KEY, INDEX_NAME 등)
load_dotenv()

# 메인 실행 부분
if __name__ == "__main__":
    print("Ingesting...")

    # 텍스트 파일을 불러옵니다.
    loader = TextLoader("/Users/edenmarco/Desktop/intro-to-vector-dbs/mediumblog1.txt")
    document = loader.load()  # 문서 로드

    print("splitting...")

    # 문서를 1000자 크기로 분할하며, 중복(overlap)은 없습니다.
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)  # 문서 분할
    print(f"created {len(texts)} chunks")  # 분할된 문서의 수 출력

    # OpenAI Embeddings 객체 생성 (환경 변수에서 API 키 불러오기)
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    print("ingesting...")

    # 분할된 문서들을 임베딩한 후, Pinecone 벡터 스토어에 저장
    PineconeVectorStore.from_documents(
        texts,  # 분할된 문서 리스트
        embeddings,  # 임베딩 객체
        index_name=os.environ["INDEX_NAME"]  # Pinecone 인덱스 이름 (환경 변수에서 불러옴)
    )

    print("finish")  # 작업 완료 메시지 출력
