# -*- coding: utf-8 -*-

# --- 경로 이름 ---
AUDIO_INPUT_DIR_NAME = "data"
RESULTS_DIR_NAME = "results"
TEMP_DIR_NAME = "temp"

# --- 모델 및 처리 설정 ---
# STT 모델
STT_MODEL = "whisper-1"

# 사용 가능한 LLM 모델
AVAILABLE_LLMS = ["gpt-4o", "gemini-2.5-pro"]

# 기본 회의 주제 및 키워드 (UI에서 오버라이드 가능)
DEFAULT_MEETING_TOPIC = "회의"
DEFAULT_KEYWORDS = ["핵심", "내용", "정리"]
