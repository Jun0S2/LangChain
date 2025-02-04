# LinkedIn 프로필 검색 에이전트
from dotenv import load_dotenv  # .env 파일에서 API 키 같은 환경 변수를 로드
load_dotenv()  # 환경 변수 불러오기

# 📌 LangChain 관련 라이브러리
from langchain_openai import ChatOpenAI  # OpenAI GPT 모델 사용
from langchain.prompts.prompt import PromptTemplate  # 프롬프트 템플릿을 생성하는 클래스
from langchain_core.tools import Tool  # AI가 사용할 도구 (검색 기능 포함)
from langchain.agents import create_react_agent, AgentExecutor  # ReAct 기반 에이전트 생성 및 실행
from langchain import hub  # LangChain의 미리 정의된 프롬프트를 가져오는 라이브러리

# 📌 검색 도구 (Google에서 LinkedIn URL 찾기)
from tools.tools import get_profile_url_tavily  # Tavily 검색 API를 이용해 LinkedIn URL을 가져오는 함수

# 📌 LinkedIn 프로필 URL 검색 함수
def lookup(name: str) -> str:
    """
    입력된 이름(name)에 대해 Google 검색을 수행하여 LinkedIn 프로필 URL을 찾는 함수.
    """
    
    # 🔹 OpenAI 기반 언어 모델 (GPT-3.5 사용)
    llm = ChatOpenAI(
        temperature=0,  # 결과의 랜덤성을 줄이고 일관된 답변을 생성 (0이면 항상 동일한 답변 가능)
        model_name="gpt-3.5-turbo",  # 사용할 모델 지정
    )

    # 🔹 프롬프트 템플릿 (검색 요청을 위한 입력 문장 생성)
    template = """given the full name {name_of_person} I want you to get me a link to their Linkedin profile page.
                  Your answer should contain only a URL"""
    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )

    # 🔹 검색 도구 (Google 검색을 활용해 LinkedIn 프로필 URL 찾기)
    tools_for_agent = [
        Tool(
            name="Crawl Google 4 linkedin profile page",  # 도구 이름
            func=get_profile_url_tavily,  # Tavily API를 이용해 검색하는 함수
            description="Useful for when you need to get the LinkedIn Page URL",  # 설명
        )
    ]

    # 🔹 LangChain Hub에서 ReAct 프롬프트 템플릿 가져오기
    react_prompt = hub.pull("hwchase17/react")

    # 🔹 ReAct 기반 에이전트 생성 (검색 도구 활용 가능)
    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)

    # 🔹 에이전트 실행기 (실제 실행을 담당)
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)

    # 🔹 에이전트 실행 및 검색 수행
    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(name_of_person=name)}
    )

    # 🔹 검색된 LinkedIn 프로필 URL 반환
    linked_profile_url = result["output"]
    return linked_profile_url


# 🔹 직접 실행하는 경우 테스트용 코드 (예: "Eden Marco Udemy" 검색)
if __name__ == "__main__":
    print(lookup(name="Eden Marco Udemy"))
