
# 저장된 벡터 데이터베이스에서 질문에 관련된 문서를 검색
# 검색된 문서를 결합한 뒤, OpenAI LLM을 통해 질문에 대한 응답을 생성.
# 필요한 라이브러리 및 모듈 불러오기
import os
from dotenv import load_dotenv  # 환경 변수를 로드하기 위한 모듈
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # OpenAI 임베딩 및 챗봇 모델
from langchain_pinecone import PineconeVectorStore  # Pinecone 벡터 스토어와 연동하기 위한 모듈
from langchain.agents import initialize_agent, Tool  # LangChain 에이전트 및 도구 초기화 모듈

# .env 파일에 저장된 환경 변수 불러오기 (예: OPENAI_API_KEY, INDEX_NAME 등)
load_dotenv()

# OpenAI의 임베딩 모델 초기화 (텍스트 데이터를 벡터로 변환)
embeddings = OpenAIEmbeddings()

# OpenAI의 챗봇 모델 초기화 (질문에 대한 자연어 응답 생성)
llm = ChatOpenAI()

# Pinecone 벡터 스토어 초기화
# 저장된 벡터 데이터베이스에서 문서 검색에 사용
vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"],  # 환경 변수에서 Pinecone 인덱스 이름 불러오기
    embedding=embeddings  # 임베딩 모델을 벡터 스토어에 연결
)

# -------- 벡터 검색 함수 정의 --------
def pinecone_vector_search(query):
    """
    Pinecone 벡터 스토어에서 입력된 쿼리와 가장 유사한 문서를 검색하는 함수.
    - query: 검색하고 싶은 질문(문자열)
    - return: 검색된 문서 내용 (최대 3개)
    """
    results = vectorstore.similarity_search(query, k=3)  # 유사한 상위 3개 문서 검색
    return "\n".join([doc.page_content for doc in results])  # 검색된 문서 내용을 문자열로 반환


# -------- 간단한 계산 함수 정의 --------
def simple_calculator(expression):
    """
    수학 수식을 받아서 결과를 반환하는 간단한 계산 함수.
    - expression: 계산할 수식 (예: "123 + 456")
    - return: 계산 결과 (문자열)
    """
    try:
        return str(eval(expression))  # eval()을 사용하여 수식 계산
    except Exception as e:
        return f"Error calculating expression: {str(e)}"  # 계산 중 오류 발생 시 에러 메시지 반환


# -------- LangChain 에이전트 도구 목록 정의 --------
tools = [
    Tool(
        name="Pinecone Vector Search",  # 도구 이름
        func=pinecone_vector_search,  # Pinecone 검색 함수 연결
        description="벡터 DB에서 문서를 검색합니다."  # 도구 설명
    ),
    Tool(
        name="Calculator",  # 도구 이름
        func=simple_calculator,  # 계산 함수 연결
        description="수학 계산을 수행합니다."  # 도구 설명
    )
]

# -------- LangChain 에이전트 초기화 --------
agent = initialize_agent(
    tools,  # 에이전트가 사용할 도구 목록
    llm,  # OpenAI 챗봇 모델
    agent_type="zero-shot-react-description"  # 에이전트 타입 설정 (지정된 도구를 상황에 맞게 사용)
)

# -------- 에이전트 실행 --------
# 에이전트에게 두 가지 작업을 지시:
# 1. Pinecone이 무엇인지 설명
# 2. 123 + 456 계산
query = "Pinecone이 뭔지 설명해줘, 그리고 123+456 계산해줘"

# 에이전트가 query에 따라 필요한 도구(Pinecone 검색, 계산기)를 사용하여 결과 생성
result = agent.run(query)

# -------- 결과 출력 --------
print(result)  # 에이전트가 반환한 최종 결과 출력
