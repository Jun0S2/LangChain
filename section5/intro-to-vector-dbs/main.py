
# 저장된 벡터 데이터베이스에서 질문에 관련된 문서를 검색
# 검색된 문서를 결합한 뒤, OpenAI LLM을 통해 질문에 대한 응답을 생성.
import os
from dotenv import load_dotenv  # 환경 변수 로드

# LangChain 모듈들: 프롬프트 템플릿, 임베딩, LLM, 벡터 스토어
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

# 문서 검색 및 결합을 위한 LangChain 기능
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# .env 파일의 환경 변수 불러오기
load_dotenv()

# 메인 실행 부분
if __name__ == "__main__":
    print("Retrieving...")

    # OpenAI 임베딩 및 LLM 객체 초기화
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI()

    # 사용자가 질문할 쿼리 정의
    query = "what is Pinecone in machine learning?"

    # 쿼리를 프롬프트 템플릿으로 변환 후 LLM에 전달
    chain = PromptTemplate.from_template(template=query) | llm
    # 위 코드는 실제로 LLM을 호출하지 않으므로 주석 처리된 부분에서 직접 호출 가능
    # result = chain.invoke(input={})
    # print(result.content)

    # Pinecone 벡터 스토어 초기화 (환경 변수에서 인덱스 이름 불러오기)
    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"],
        embedding=embeddings
    )

    # LangChain Hub에서 사전 정의된 Retrieval QA 프롬프트 불러오기
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # 검색된 문서들을 결합하는 체인 생성
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

    # 벡터 스토어를 이용해 문서를 검색하고, 검색된 문서를 결합하는 체인 생성
    retrival_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(),  # 벡터 스토어를 검색기로 사용
        combine_docs_chain=combine_docs_chain  # 검색된 문서 결합
    )

    # 최종적으로 질문(query)에 대한 답변 생성
    result = retrival_chain.invoke(input={"input": query})

    # 결과 출력
    print(result)
