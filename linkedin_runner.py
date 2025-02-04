from dotenv import load_dotenv

# 📌 LangChain 관련 라이브러리
from langchain.prompts.prompt import PromptTemplate  # 프롬프트 템플릿을 생성하는 클래스
from langchain_openai import ChatOpenAI  # OpenAI 기반 LLM (GPT-4o-mini 사용)

# 📌 출력 형식 변환을 위한 Pydantic 기반 출력 파서
from output_parsers import summary_parser

# 📌 외부 API 또는 크롤링을 통해 데이터를 가져오는 함수
from third_parties.linkedin import scrape_linkedin_profile  # LinkedIn 프로필 정보 스크래핑
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent  # LinkedIn 프로필 검색
from agents.twitter_lookup_agent import lookup as twitter_lookup_agent  # Twitter 프로필 검색
from third_parties.twitter import scrape_user_tweets  # Twitter에서 최근 트윗 가져오기


# 📌 특정 인물에 대한 정보를 분석하고 Ice Breaker 생성
def ice_break_with(name: str) -> str:
    """
    - 주어진 인물(name)에 대해 LinkedIn 및 Twitter에서 정보를 가져온 후,
    - AI를 이용해 해당 인물의 요약 및 흥미로운 사실을 생성하는 함수.
    """

    # 🔹 LinkedIn 프로필 조회 및 데이터 수집
    linkedin_username = linkedin_lookup_agent(name=name)  # LinkedIn 프로필 URL 검색
    linkedin_data = scrape_linkedin_profile(
        linkedin_profile_url=linkedin_username, mock=True  # mock=True -> 테스트용 데이터 사용 가능
    )

    # 🔹 Twitter 프로필 조회 및 최근 트윗 가져오기
    twitter_username = twitter_lookup_agent(name=name)  # Twitter 사용자명 검색
    tweets = scrape_user_tweets(username=twitter_username, mock=True)  # 최근 트윗 가져오기

    # 🔹 AI 모델이 사용할 프롬프트 템플릿 정의
    summary_template = """
    Given the information about a person from LinkedIn {information},
    and their latest Twitter posts {twitter_posts}, I want you to create:
    1. A short summary
    2. Two interesting facts about them 

    Use both information from Twitter and LinkedIn
    \n{format_instructions}
    """
    
    # 🔹 프롬프트 템플릿 생성 (Twitter 및 LinkedIn 데이터를 AI 모델이 처리하도록 설정)
    summary_prompt_template = PromptTemplate(
        input_variables=["information", "twitter_posts"],  # 프롬프트에서 사용할 입력 변수
        template=summary_template,  # 위에서 정의한 템플릿 사용
        partial_variables={
            "format_instructions": summary_parser.get_format_instructions()  # 출력 형식을 지정하는 부분
        },
    )

    # 🔹 GPT-4o-mini 모델 설정 (온도 값 0으로 설정하여 응답의 일관성을 높임)
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")

    # 🔹 AI 체인 생성 (프롬프트 -> GPT 모델 실행 -> 결과 파싱)
    chain = summary_prompt_template | llm | summary_parser

    # 🔹 AI 실행 및 결과 생성
    res = chain.invoke(input={"information": linkedin_data, "twitter_posts": tweets})

    # 🔹 결과 출력
    print(res)


# 📌 실행하는 경우 (환경 변수 로드 후 특정 인물에 대해 Ice Breaker 생성)
if __name__ == "__main__":
    load_dotenv()  # .env 파일에서 환경 변수 불러오기

    print("Ice Breaker Enter")  # 실행 시작 메시지
    ice_break_with(name="Harrison Chase")  # "Harrison Chase"라는 인물에 대해 Ice Breaker 생성
