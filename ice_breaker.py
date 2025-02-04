# 🔹 OS 모듈과 dotenv 로드
import os  # 운영체제(OS) 관련 기능을 다룰 수 있는 모듈
from dotenv import load_dotenv  # .env 파일에서 환경변수를 불러오는 라이브러리

# 🔹 LangChain 관련 모듈
from langchain_core.prompts import PromptTemplate  # 프롬프트 템플릿을 만드는 모듈
# from langchain_openai import ChatOpenAI  # OpenAI의 LLM(Language Model)을 사용하는 모듈
# 🔹 Ollama 모델을 사용하도록 변경
# 기존 OpenAI 기반 ChatOpenAI 대신 langchain_ollama의 ChatOllama 사용

# 🔹 Output을 문자열(String)로 변환하는 파서 추가
from langchain_ollama import ChatOllama # swap to ollama instead
from langchain_core.output_parsers import StrOutputParser

# 🔹 .env 파일 로드 (환경 변수 불러오기)
load_dotenv()  # .env 파일에 저장된 API 키를 불러와서 환경 변수로 설정

# 📌 요약할 대상의 정보 (Elon Musk에 대한 텍스트)
information="""
Elon Reeve Musk (/ˈiːlɒn mʌsk/; born June 28, 1971) is a businessman and United States federal special government employee known for his key roles in the automotive company Tesla, Inc. and the space company SpaceX. He is also known for his ownership of the technology company X Corp. and his role in the founding of the Boring Company, xAI, Neuralink, and OpenAI. Musk is the wealthiest individual in the world; as of January 2025, Forbes estimates his net worth to be US$426 billion.

A member of the wealthy South African Musk family, Musk was born in Pretoria and briefly attended the University of Pretoria. At the age of 18 he immigrated to Canada, acquiring its citizenship through his Canadian-born mother, Maye. Two years later, he matriculated at Queen's University in Canada. Musk later transferred to the University of Pennsylvania and received bachelor's degrees in economics and physics. He moved to California in 1995 to attend Stanford University but never enrolled in classes, and with his brother Kimbal co-founded the online city guide software company Zip2. The startup was acquired by Compaq for $307 million in 1999. That same year, Musk co-founded X.com, a direct bank. X.com merged with Confinity in 2000 to form PayPal. In 2002, Musk acquired United States citizenship, and that October eBay acquired PayPal for $1.5 billion. Using $100 million of the money he made from the sale of PayPal, Musk founded SpaceX, a spaceflight services company, in 2002.

In 2004, Musk was an early investor in electric vehicle manufacturer Tesla Motors, Inc. (later Tesla, Inc.), providing most of the initial financing and assuming the position of the company's chairman. He later became the product architect and, in 2008, the CEO. In 2006, Musk helped create SolarCity, a solar energy company that was acquired by Tesla in 2016 and became Tesla Energy. In 2013, he proposed a hyperloop high-speed vactrain transportation system. In 2015, he co-founded OpenAI, a nonprofit artificial intelligence research company. The following year Musk co-founded Neuralink, a neurotechnology company developing brain–computer interfaces, and the Boring Company, a tunnel construction company. In 2018, the U.S. Securities and Exchange Commission (SEC) sued Musk, alleging he falsely announced that he had secured funding for a private takeover of Tesla. To settle the case, Musk stepped down as the chairman of Tesla and paid a $20 million fine. In 2022, he acquired Twitter for $44 billion, merged the company into his newly-created X Corp., and rebranded the service as X the following year. In 2023, Musk founded xAI, an artificial intelligence company.

Musk's actions and expressed views have made him a polarizing figure. He has been criticized for making unscientific and misleading statements, including COVID-19 misinformation, affirming antisemitic and transphobic comments, and promoting conspiracy theories. His acquisition of Twitter was controversial due to large employee layoffs, an increase in hate speech, the spread of misinformation and disinformation on the service, and changes to various service features including verification. Musk has engaged in political activities in several countries, including as a vocal and financial supporter of U.S. president Donald Trump, becoming the largest donor in the 2024 United States presidential election. In January 2025, Musk was appointed head of the Department of Government Efficiency, while at Trump's inauguration he made a controversial gesture that received widespread criticism
"""

if __name__ == "__main__":
    # 1️⃣ 요약 프롬프트 템플릿 만들기
    # 🔹 `{information}`: 중괄호 `{}` 안에 변수 값을 넣을 수 있도록 만든 템플릿
    summary_template = """
        given the information {information} about a person from I want you to provide : 
        1. a short summary
        2. two interesting facts about them
    """

    # 2️⃣ PromptTemplate 객체 생성
    summary_prompt_template = PromptTemplate(input_variables=['information'], template=summary_template)

    # 3️⃣ 모델 사용 설정
    # Chat GPT 모델 사용
    # llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    # 🔹 temperature=0: 답변을 더 **일관되게** 출력하도록 설정 (값이 클수록 랜덤성이 높아짐)
    #  model_name="gpt-4o-mini": 사용 모델 선택
    # 🔹 Ollmam class 사용 설정
    llm = ChatOllama(model="llama3.2")

    # 5️⃣ 체인 실행 
    # chain = summary_prompt_template | llm
    chain = summary_prompt_template | llm | StrOutputParser()
    res = chain.invoke(input={"information": information})

    # 6️⃣ 결과 출력
    print(res)