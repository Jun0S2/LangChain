# Linkedin 프로필 정보 scrapping
import os  # 환경 변수 사용을 위한 OS 모듈
import requests  # HTTP 요청을 보낼 때 사용하는 라이브러리
from dotenv import load_dotenv  # 환경 변수(.env 파일) 로드

# 🔹 환경 변수 로드 (.env에서 API 키를 불러오기 위함)
load_dotenv()

# 📌 LinkedIn 프로필 정보를 스크래핑하는 함수
def scrape_linkedin_profile(linkedin_profile_url: str, mock: bool = False):
    """
    LinkedIn 프로필 페이지에서 정보를 수집하는 함수.
    - mock=True: 실제 API 호출 대신 미리 저장된 JSON 파일을 불러옴 (테스트 용도)
    - mock=False: Proxycurl API를 사용하여 LinkedIn 데이터를 가져옴
    """

    if mock:
        # 🔹 테스트용 JSON 데이터 사용
        linkedin_profile_url = "https://gist.githubusercontent.com/emarco177/.../eden-marco.json"
        response = requests.get(linkedin_profile_url, timeout=10)
    else:
        # 🔹 실제 Proxycurl API를 사용하여 LinkedIn 프로필 데이터 요청
        api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"
        header_dic = {"Authorization": f'Bearer {os.environ.get("PROXYCURL_API_KEY")}'}
        response = requests.get(
            api_endpoint,
            headers=header_dic,
            params={"url": linkedin_profile_url},
            timeout=10,
        )

    # 🔹 API 응답을 JSON 형식으로 변환
    data = response.json()

    # 🔹 불필요한 항목 제거 (빈 값 및 특정 필드)
    data = {
        k: v for k, v in data.items()
        if v not in ([], "", None) and k not in ["people_also_viewed", "certifications"]
    }

    # 🔹 그룹 정보에서 프로필 이미지 URL 제거
    if data.get("groups"):
        for group_dict in data.get("groups"):
            group_dict.pop("profile_pic_url")

    return data


# 🔹 테스트 실행
if __name__ == "__main__":
    linkedin_profile_url = "https://www.linkedin.com/in/eden-marco/"
    scrape_linkedin_profile(linkedin_profile_url)
