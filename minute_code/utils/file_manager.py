# -*- coding: utf-8 -*-
import os
import json
import logging
from datetime import datetime

def save_results(base_results_dir, original_filename, meeting_topic, keywords, original_transcript, corrected_transcript, summary):
    """
    처리된 모든 결과를 파일로 저장합니다.
    결과 폴더를 '원본파일명_YYYYMMDDHHMM' 형식으로 생성하고, 충돌 시 숫자를 붙입니다.
    """
    # 'results' 폴더가 없으면 생성
    os.makedirs(base_results_dir, exist_ok=True)

    # 년월일시분 형식의 타임스탬프 생성
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    base_filename = os.path.splitext(os.path.basename(original_filename))[0]
    
    # 새 결과 폴더명 설정
    new_folder_name = f"{base_filename}_{timestamp}"
    results_dir = os.path.join(base_results_dir, new_folder_name)

    # 폴더명 충돌 시 숫자 추가
    counter = 1
    temp_results_dir = results_dir
    while os.path.exists(temp_results_dir):
        temp_results_dir = f"{results_dir}_{counter}"
        counter += 1
    results_dir = temp_results_dir
    os.makedirs(results_dir)

    logging.info(f"결과를 '{results_dir}' 폴더에 저장합니다.")

    # 파일 경로 정의
    stt_txt_path = os.path.join(results_dir, f"stt_{base_filename}.txt")
    corrected_txt_path = os.path.join(results_dir, f"corrected_{base_filename}.txt")
    summary_md_path = os.path.join(results_dir, f"summary_{base_filename}.md")
    results_json_path = os.path.join(results_dir, f"diarization_{base_filename}.json")

    # 1. 원본 STT 결과 (TXT)
    try:
        with open(stt_txt_path, "w", encoding="utf-8") as f:
            for segment in original_transcript:
                f.write(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['speaker']}: {segment['text']}\n")
        logging.info(f"원본 STT 결과를 '{stt_txt_path}'에 저장했습니다.")
    except IOError as e:
        logging.error(f"파일 저장 중 오류 발생 ({stt_txt_path}): {e}")

    # 2. LLM 교정 결과 (TXT)
    try:
        with open(corrected_txt_path, "w", encoding="utf-8") as f:
            for segment in corrected_transcript:
                f.write(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['speaker']}: {segment['text']}\n")
        logging.info(f"LLM 교정 결과를 '{corrected_txt_path}'에 저장했습니다.")
    except IOError as e:
        logging.error(f"파일 저장 중 오류 발생 ({corrected_txt_path}): {e}")

    # 3. 회의 요약 결과 (MD)
    try:
        with open(summary_md_path, "w", encoding="utf-8") as f:
            f.write(f"# 회의 요약: {meeting_topic}\n\n")
            f.write(f"## 주요 키워드\n- {', '.join(keywords)}\n\n")
            f.write("## 핵심 요약\n")
            f.write(summary + "\n\n")
        logging.info(f"회의 요약 내용을 '{summary_md_path}'에 저장했습니다.")
    except IOError as e:
        logging.error(f"파일 저장 중 오류 발생 ({summary_md_path}): {e}")

    # 4. 모든 결과 (JSON)
    combined_results = {
        "meeting_topic": meeting_topic,
        "keywords": keywords,
        "original_transcript": original_transcript,
        "corrected_transcript": corrected_transcript,
        "summary": summary
    }
    try:
        with open(results_json_path, "w", encoding="utf-8") as f:
            json.dump(combined_results, f, ensure_ascii=False, indent=4)
        logging.info(f"모든 결과를 '{results_json_path}'에 저장했습니다.")
    except IOError as e:
        logging.error(f"파일 저장 중 오류 발생 ({results_json_path}): {e}")
        
    return results_dir
