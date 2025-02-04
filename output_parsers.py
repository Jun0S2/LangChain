# 필수 라이브러리 임포트
from typing import List, Dict, Any  # 타입 힌팅을 위한 모듈
from langchain_core.output_parsers import PydanticOutputParser  # LangChain의 Pydantic 기반 출력 파서
from pydantic import BaseModel, Field  # 데이터 모델 검증을 위한 Pydantic 라이브러리

# 📌 요약 결과를 저장할 데이터 모델 정의 (Pydantic 사용)
class Summary(BaseModel):
    """
    - AI 모델이 생성한 요약 결과를 구조화하기 위한 데이터 모델.
    - `BaseModel`을 상속하여 Pydantic을 기반으로 필드 검증 수행.
    """

    summary: str = Field(description="summary")  # 논문의 요약 내용을 저장하는 필드
    facts: List[str] = Field(description="interesting facts about them")  # 관련된 흥미로운 사실 리스트

    def to_dict(self) -> Dict[str, Any]:
        """
        - Pydantic 객체를 딕셔너리 형태로 변환하는 메서드.
        - JSON 직렬화 시 유용하게 사용 가능.
        """
        return {"summary": self.summary, "facts": self.facts}


# 📌 Pydantic 기반의 LangChain 출력 파서 생성
summary_parser = PydanticOutputParser(pydantic_object=Summary)


# Examples
# parsed_output = summary_parser.parse('{"summary": "Elon Musk is a visionary entrepreneur...", "facts": ["Founded SpaceX", "CEO of Tesla"]}')
# print(parsed_output.summary)  # "Elon Musk is a visionary entrepreneur..."
# print(parsed_output.facts)  # ["Founded SpaceX", "CEO of Tesla"]

