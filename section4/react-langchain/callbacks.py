from typing import Dict, Any, List
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult

# BaseCallbackHandler를 상속한 사용자 정의 콜백 핸들러 클래스 정의
class AgentCallbackHandler(BaseCallbackHandler):
    
    # LLM이 시작될 때 호출되는 메서드
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """
        LLM이 실행을 시작할 때 호출됩니다.

        :param serialized: 직렬화된 데이터로, LLM의 설정 정보나 메타데이터가 포함될 수 있습니다.
        :param prompts: LLM에 입력된 프롬프트(질문) 리스트입니다.
        :param kwargs: 추가적인 선택적 인자들로, 필요에 따라 다양한 데이터를 전달할 수 있습니다.
        """
        # LLM 실행 시 사용된 프롬프트 목록을 출력
        print(f"LLM is starting with prompts: {prompts}")

    # LLM이 작업을 마치고 결과를 반환할 때 호출되는 메서드
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """
        LLM이 실행을 완료하고 결과를 반환할 때 호출됩니다.

        :param response: LLM의 실행 결과가 포함된 LLMResult 객체입니다.
        :param kwargs: 추가적인 선택적 인자들로, 필요에 따라 다양한 데이터를 전달할 수 있습니다.
        """
        # LLM의 출력 결과를 출력
        print(f"LLM output: {response.text}")
