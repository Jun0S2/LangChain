import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
# .env 파일 로드
load_dotenv()

if __name__ == "__main__":
    print("Hello Langchain!")
    print(os.environ.get("OPENAI_API_KEY", "환경 변수 없음"))  # 안전한 방식
