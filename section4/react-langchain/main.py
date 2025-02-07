# 필수 라이브러리 임포트
from typing import Union, List
from dotenv import load_dotenv
from langchain.agents import tool
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import Tool
from langchain.tools.render import render_text_description

# load callbacks.py
from callbacks import AgentCallbackHandler


# .env 파일에서 환경 변수 로드 (API 키 등)
load_dotenv()

# 툴 정의: 문자열의 길이를 반환하는 함수
@tool
def get_text_length(text: str) -> int:
    """문자열의 길이를 반환하는 함수"""
    print(f"get_text_length enter with {text=}")
    
    # 문자열 앞뒤의 불필요한 문자를 제거 (따옴표, 개행 문자 등)
    text = text.strip("'\n").strip('"')
    
    return len(text)  # 문자열의 길이 반환

# 툴 이름으로 툴 객체를 찾는 함수
def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool wtih name {tool_name} not found")

# 메인 실행 부분
if __name__ == "__main__":
    print("Hello ReAct LangChain!")
    
    # 사용할 툴 목록 정의
    tools = [get_text_length]

    # 에이전트에 사용할 프롬프트 템플릿 정의
    # langchainsmith 에서 prompt 를 쓸 수 있음 (https://docs.smith.langchain.com/old/category/prompt-hub)
    # Example : hwchase17/react (https://smith.langchain.com/hub/hwchase17/react)
    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: {input}
    Thought: {agent_scratchpad}
    """
    
    # 프롬프트 객체 생성 및 툴 정보 추가
    # This prompt will be sent to LLM (also its chain of thought prompt!)
    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),  # 툴 설명 렌더링
        tool_names=", ".join([t.name for t in tools]),  # 툴 이름 목록 생성
    )
    
    # LLM (OpenAI Chat 모델) 초기화, Observation이 나오면 중단
    # stop 은 endcase 라 반드시 필요하다.
    llm = ChatOpenAI(
        temperature=0,
        model_kwargs={"stop": ["\nObservation", "Observation"]},
        callbacks=[AgentCallbackHandler()],
    )
    
    intermediate_steps = []  # 에이전트의 중간 수행 과정을 저장할 리스트

    # LangChain Expression Language (LCEL)을 사용한 에이전트 구성
    agent = (
        {
            "input": lambda x: x["input"],  # 입력 질문 처리
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),  # 에이전트의 사고 과정 포맷팅
        }
        | prompt  # 프롬프트 적용
        | llm  # LLM에 질문 전달
        | ReActSingleInputOutputParser()  # ReAct 패턴으로 출력 파싱
    )

   agent_step = None  # 초기화
    while not isinstance(agent_step, AgentFinish):
        # 첫 번째 에이전트 실행: "DOG"의 길이 질문
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {
                "input": "What is the length of the word: DOG",
                "agent_scratchpad": intermediate_steps,  # 이전 수행 과정 전달
            }
        )
        print(agent_step)  # 에이전트의 첫 번째 응답 출력

        # 에이전트가 액션을 요구한 경우 툴 실행
        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool  # 사용할 툴 이름
            tool_to_use = find_tool_by_name(tools, tool_name)  # 툴 객체 찾기
            tool_input = agent_step.tool_input  # 툴에 전달할 입력값
            observation = tool_to_use.func(str(tool_input))  # 툴 실행 및 결과 저장
            print(f"{observation=}")  # 툴 실행 결과 출력
            intermediate_steps.append((agent_step, str(observation)))  # 중간 결과 추가


    # 최종 답변인 경우 출력
    if isinstance(agent_step, AgentFinish):
        print("### AgentFinish ###")
        print(agent_step.return_values)  # 최종 답변 출력
