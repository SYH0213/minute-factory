# -*- coding: utf-8 -*-
import os
import time
import logging
import concurrent.futures
from pydub import AudioSegment
import re
import uuid
from slugify import slugify

# 현재 패키지의 모듈들 임포트
from utils import config
from core_processing.llm_tasks import correct_text, summarize_text, get_openai_client
from core_processing.stt_diarization import diarize_audio, transcribe_segment
from utils.file_manager import save_results
from utils.api_keys import check_api_keys
from chatbot.crag_logic import update_vector_store # 챗봇 모듈 임포트
from utils.prompts import STT_PROMPT_TEMPLATE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_pipeline(audio_path: str, llm_choice: str, topic: str, keywords: list):
    """
    메인 처리 파이프라인.
    오디오 파일을 입력받아 화자분리, STT, 교정, 요약 과정을 거쳐 결과를 저장합니다.
    
    Args:
        audio_path (str): 처리할 오디오 파일의 전체 경로.
        llm_choice (str): 사용할 LLM 모델 (예: 'gpt-4', 'gemini-pro').
        topic (str): 회의 주제.
        keywords (list): 회의 주요 키워드 리스트.

    Returns:
        tuple: (결과 폴더 경로, 상태 메시지) 튜플.
    """
    logging.info(f"--- 새로운 처리 파이프라인 시작 ---")
    logging.info(f"입력 파일: {audio_path}")
    logging.info(f"선택된 LLM: {llm_choice}")

    # --- 0. API 키 확인 --- #
    error_message = check_api_keys(llm_choice)
    if error_message:
        logging.error(error_message)
        return None, error_message

    # --- 1. 화자 분리 --- #
    diarization = diarize_audio(audio_path)
    if not diarization:
        return None, "화자 분리에 실패했습니다. Pyannote 토큰 또는 오디오 파일을 확인하세요."

    # --- 2. 병렬 STT 처리 --- #
    try:
        audio = AudioSegment.from_wav(audio_path)
    except Exception as e:
        logging.error(f"오디오 파일 로딩 실패: {e}")
        return None, f"오디오 파일({os.path.basename(audio_path)})을 열 수 없습니다."

    stt_prompt = STT_PROMPT_TEMPLATE.format(topic=topic, keywords=', '.join(keywords))
    client = get_openai_client()
    if not client:
        return None, "OpenAI API 클라이언트 초기화에 실패했습니다. .env 파일을 확인하세요."

    tasks = []
    segments_info = []
    temp_dir = config.TEMP_DIR_NAME
    os.makedirs(temp_dir, exist_ok=True) # 임시 폴더 생성

    for i, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
        if turn.end - turn.start < 1.0:
            continue
        start_ms = turn.start * 1000
        end_ms = turn.end * 1000
        segment_audio = audio[start_ms:end_ms]
        segment_filename = os.path.join(temp_dir, f"segment_{i}_{int(time.time())}.wav")
        tasks.append((segment_audio, segment_filename))
        segments_info.append({'turn': turn, 'speaker': speaker})

    transcribed_texts = [""] * len(tasks)
    stt_start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_index = {executor.submit(transcribe_segment, client, task[0], task[1], stt_prompt, config.STT_MODEL): i for i, task in enumerate(tasks)}
        
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                transcribed_texts[index] = future.result()
            except Exception as exc:
                logging.error(f"STT 작업 중 오류 발생 (인덱스 {index}): {exc}")

    stt_end_time = time.time()
    logging.info(f"음성 인식 완료. (총 처리 시간: {stt_end_time - stt_start_time:.2f}초)")

    original_transcript = []
    for i, text in enumerate(transcribed_texts):
        if text:
            info = segments_info[i]
            original_transcript.append({
                "start": info['turn'].start,
                "end": info['turn'].end,
                "speaker": info['speaker'],
                "text": text
            })

    if not original_transcript:
        return None, "음성 인식 결과가 없습니다."

    # --- 3. LLM 텍스트 교정 --- #
    full_text_for_correction = "\n".join(f"{seg['speaker']}: {seg['text']}" for seg in original_transcript)
    corrected_full_text = correct_text(llm_choice, full_text_for_correction, topic, keywords)
    
    corrected_lines = corrected_full_text.strip().split('\n')
    corrected_transcript = []
    for i, segment in enumerate(original_transcript):
        new_text = segment['text']
        if i < len(corrected_lines):
            parts = corrected_lines[i].split(':', 1)
            if len(parts) > 1:
                new_text = parts[1].strip()
        corrected_transcript.append({
            "start": segment['start'], "end": segment['end'],
            "speaker": segment['speaker'], "text": new_text
        })

    # --- 4. LLM 텍스트 요약 --- #
    text_for_summary = "\n".join(seg['text'] for seg in corrected_transcript)
    summary = summarize_text(llm_choice, text_for_summary, topic, keywords)

    # --- 5. 결과 저장 --- #
    results_path = save_results(
        base_results_dir=config.RESULTS_DIR_NAME,
        original_filename=audio_path,
        meeting_topic=topic,
        keywords=keywords,
        original_transcript=original_transcript,
        corrected_transcript=corrected_transcript,
        summary=summary
    )
    
    # --- 6. 벡터 저장소 업데이트 --- #
    if results_path:
        base_filename = os.path.splitext(os.path.basename(audio_path))[0]
        corrected_txt_path = os.path.join(results_path, f"corrected_{base_filename}.txt")
        
        # 1. Transliterate base_filename to an ASCII slug
        slugified_name = slugify(base_filename, separator='_', lowercase=True, replacements=[['.', '_']])
        # 2. Apply final sanitization (ChromaDB specific)
        collection_name = re.sub(r'[^a-zA-Z0-9._-]', '_', slugified_name)

        # 3. Ensure collection_name meets ChromaDB's rules (min 3 chars, starts/ends with alphanumeric)
        # Remove leading/trailing underscores that might violate start/end rule
        collection_name = collection_name.strip('_')
        
        # If it became empty after stripping, or is too short, use a fallback
        if not collection_name or len(collection_name) < 3:
            # Fallback to a unique identifier if the slugified name is too short
            collection_name = "meeting_" + str(uuid.uuid4())[:8].replace('-', '_') # Ensure it's always valid and long enough
        
        try:
            update_vector_store(corrected_txt_path, collection_name)
            logging.info(f"벡터 저장소 업데이트 완료: {collection_name}")
        except Exception as e:
            logging.error(f"벡터 저장소 업데이트 중 오류 발생: {e}")

    logging.info(f"--- 파이프라인 종료 --- 결과가 '{results_path}'에 저장되었습니다.")
    return results_path, "모든 처리가 완료되었습니다."
