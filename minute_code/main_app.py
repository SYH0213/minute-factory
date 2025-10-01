# -*- coding: utf-8 -*-
import gradio as gr
import os
import logging

# 프로젝트 모듈 임포트
from ui.app_layout import create_ui
from utils.config import (
    AUDIO_INPUT_DIR_NAME,
    RESULTS_DIR_NAME,
    TEMP_DIR_NAME
)

# --- 기본 설정 및 디렉터리 생성 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_directories():
    """프로젝트에 필요한 모든 디렉터리를 생성합니다."""
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # config.py에서 정의된 디렉토리 이름들을 사용하여 경로 구성
    dir_names = [AUDIO_INPUT_DIR_NAME, RESULTS_DIR_NAME, TEMP_DIR_NAME]
    
    for dir_name in dir_names:
        path = os.path.join(ROOT_DIR, dir_name)
        os.makedirs(path, exist_ok=True)
        logging.info(f"Directory '{path}' is ready.")

# --- 애플리케이션 실행 ---
if __name__ == "__main__":
    # 1. 필요 디렉터리 생성
    setup_directories()
    
    # 2. UI 생성 및 실행
    app = create_ui()
    app.launch()