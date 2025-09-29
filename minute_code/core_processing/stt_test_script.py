# -*- coding: utf-8 -*-
import os
import logging
from openai import OpenAI
from pydub import AudioSegment # For segmenting audio if needed, or just use full file
import time

# Project imports
from core_processing.stt_diarization import transcribe_segment
from utils.prompts import STT_PROMPT_TEMPLATE
from utils.config import STT_MODEL as DEFAULT_STT_MODEL, TEMP_DIR_NAME
from utils.api_keys import get_openai_client # To get the client

# --- Configuration for Testing ---
# Configure logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Path to your test audio file (relative to project root)
# You might need to adjust this path or provide a full path
# Example: Place a test_audio.wav in the 'data' folder at the project root
TEST_AUDIO_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "test_audio.wav")

TEST_TOPIC = "STT 테스트 회의"
TEST_KEYWORDS = ["테스트", "음성인식", "성능"]
TEST_STT_MODEL = DEFAULT_STT_MODEL # Use default from config, or override here (e.g., "whisper-1")

# --- Main Test Function ---
def run_stt_test():
    logging.info(f"--- STT 테스트 시작 ---")
    logging.info(f"테스트 오디오 파일: {TEST_AUDIO_FILE_PATH}")
    logging.info(f"사용 모델: {TEST_STT_MODEL}")
    logging.info(f"테스트 주제: {TEST_TOPIC}")
    logging.info(f"테스트 키워드: {', '.join(TEST_KEYWORDS)}")

    if not os.path.exists(TEST_AUDIO_FILE_PATH):
        logging.error(f"오류: 테스트 오디오 파일을 찾을 수 없습니다: {TEST_AUDIO_FILE_PATH}")
        logging.info("테스트를 위해 'data' 폴더에 'test_audio.wav' 파일을 넣어주시거나, TEST_AUDIO_FILE_PATH를 수정해주세요.")
        return

    client = get_openai_client()
    if not client:
        logging.error("OpenAI 클라이언트 초기화 실패. .env 파일의 OPENAI_API_KEY를 확인하세요.")
        return

    # Construct the STT prompt using the template
    stt_prompt = STT_PROMPT_TEMPLATE.format(topic=TEST_TOPIC, keywords=', '.join(TEST_KEYWORDS))
    logging.info(f"사용 STT 프롬프트: {stt_prompt}")

    # For testing, we'll treat the whole audio file as one segment for simplicity
    # In a real scenario, you'd segment it first (e.g., using diarization)
    try:
        audio = AudioSegment.from_wav(TEST_AUDIO_FILE_PATH)
        
        # Create a temporary file for the segment
        temp_dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), TEMP_DIR_NAME)
        os.makedirs(temp_dir_path, exist_ok=True)
        temp_segment_path = os.path.join(temp_dir_path, f"test_segment_{int(time.time())}.wav")
        
        logging.info("음성 인식 시작...")
        start_time = time.time()
        
        # Call the transcribe_segment function directly
        # Note: transcribe_segment expects an AudioSegment object and a path to save it temporarily
        transcribed_text = transcribe_segment(client, audio, temp_segment_path, stt_prompt, TEST_STT_MODEL)
        
        end_time = time.time()
        logging.info(f"음성 인식 완료 (소요 시간: {end_time - start_time:.2f}초)")
        
        logging.info("\n--- STT 결과 ---")
        logging.info(transcribed_text)
        logging.info("----------------\n")

    except Exception as e:
        logging.error(f"STT 테스트 중 오류 발생: {e}")
    finally:
        if os.path.exists(temp_segment_path):
            os.remove(temp_segment_path)
            logging.info(f"임시 파일 삭제: {temp_segment_path}")

if __name__ == "__main__":
    run_stt_test()
