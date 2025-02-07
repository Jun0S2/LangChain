# --------- 환경 설정 ---------
import os

# OpenAI API 키 설정 (환경 변수에 직접 입력)
os.environ["OPENAI_API_KEY"] = "xxx"

# --------- 필요한 라이브러리 불러오기 ---------
from langchain_community.document_loaders import PyPDFLoader  # PDF 파일 로딩 모듈
from langchain_text_splitters import CharacterTextSplitter  # 텍스트 분할 모듈
from langchain_openai import OpenAIEmbeddings, OpenAI  # OpenAI 임베딩 및 LLM 모델
from langchain_community.vectorstores import FAISS  # FAISS 벡터 스토어 (로컬 벡터 검색 엔진)
from langchain.chains.retrieval import create_retrieval_chain  # 검색 체인 생성 모듈
from langchain.chains.combine_documents import create_stuff_documents_chain  # 문서 결합 체인 생성 모듈
from langchain import hub  # LangChain 프롬프트 허브

# --------- 메인 실행 부분 ---------
if __name__ == "__main__":
    print("hi")

    # --------- 1. PDF 파일 불러오기 ---------
    pdf_path = "/Users/edenmarco/Desktop/tmp/react.pdf"  # 분석할 PDF 파일 경로
    loader = PyPDFLoader(file_path=pdf_path)  # PDF 파일 로드
    documents = loader.load()  # PDF 문서를 텍스트로 변환하여 로드

    # --------- 2. 문서 분할 ---------
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,      # 각 텍스트 조각의 크기를 1000자로 설정
        chunk_overlap=30,     # 문서 조각 간 30자 중복 허용
        separator="\n"        # 줄바꿈 기준으로 분할
    )
    docs = text_splitter.split_documents(documents=documents)  # 문서 분할 실행

    # --------- 3. 문서 임베딩 및 FAISS에 저장 ---------
    embeddings = OpenAIEmbeddings()  # OpenAI 임베딩 모델 초기화 (문서 → 벡터로 변환)
    vectorstore = FAISS.from_documents(docs, embeddings)  # 분할된 문서를 FAISS 벡터 스토어에 저장
    vectorstore.save_local("faiss_index_react")  # 생성된 벡터 인덱스를 로컬에 저장 (faiss_index_react 폴더)

    # --------- 4. 저장된 FAISS 인덱스 불러오기 ---------
    new_vectorstore = FAISS.load_local(
        "faiss_index_react",  # 저장된 인덱스 폴더 경로
        embeddings,           # 동일한 임베딩 모델 사용
        allow_dangerous_deserialization=True  # 비직렬화 경고 무시 (로컬 파일이므로 안전)
    )

    # --------- 5. LangChain 프롬프트 및 체인 설정 ---------
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")  
    # LangChain 허브에서 Retrieval QA 프롬프트 불러오기 (검색 기반 QA 프롬프트)

    combine_docs_chain = create_stuff_documents_chain(
        OpenAI(), retrieval_qa_chat_prompt
    )
    # 검색된 문서를 결합하고 OpenAI LLM을 통해 최종 응답 생성

    retrieval_chain = create_retrieval_chain(
        new_vectorstore.as_retriever(),  # FAISS 벡터 스토어를 검색기로 사용
        combine_docs_chain               # 검색된 문서를 결합하는 체인
    )

    # --------- 6. 질문하고 답변 받기 ---------
    res = retrieval_chain.invoke({"input": "Give me the gist of ReAct in 3 sentences"})
    # "ReAct(Reasoning and Acting)"에 대한 요약 요청

    print(res["answer"])  # 최종 답변 출력
