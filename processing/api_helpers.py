# -*- coding: utf-8 -*-
import os
import logging
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

def get_pyannote_token():
    """ .env 파일에서 Pyannote 인증 토큰을 가져옵니다. """
    token = os.getenv("PYANNOTE_TOKEN")
    if not token:
        logging.warning("PYANNOTE_TOKEN이 .env 파일에 설정되지 않았습니다. 화자 분리 기능이 제한될 수 있습니다.")
    return token

def check_api_keys(llm_choice: str):
    """필요한 API 키가 .env 파일에 모두 설정되었는지 확인합니다."""
    missing_keys = []
    if not os.getenv("PYANNOTE_TOKEN"):
        missing_keys.append("PYANNOTE_TOKEN")
    if not os.getenv("OPENAI_API_KEY"):
        # Whisper STT와 GPT-4 모두 이 키를 사용하므로 항상 필요합니다.
        missing_keys.append("OPENAI_API_KEY")
    
    if llm_choice == "gemini-pro" and not os.getenv("GOOGLE_API_KEY"):
        missing_keys.append("GOOGLE_API_KEY")
        
    if missing_keys:
        # 중복 제거 (예: OpenAI 키가 두 번 필요할 경우)
        unique_missing_keys = sorted(list(set(missing_keys)))
        return f"API 키가 .env 파일에 설정되지 않았습니다: {', '.join(unique_missing_keys)}"
    
    return None
