# -*- coding: utf-8 -*-
import gradio as gr
import os
import logging
from datetime import datetime

# 통합된 처리 파이프라인 및 설정 임포트
from processing.pipeline import run_pipeline
from processing.config import (
    AUDIO_INPUT_DIR_NAME,
    RESULTS_DIR_NAME,
    TEMP_DIR_NAME,
    AVAILABLE_LLMS,
    DEFAULT_MEETING_TOPIC,
    DEFAULT_KEYWORDS
)
# 오디오 입력 UI 모듈 임포트
from processing.audio_input import (
    get_audio_files_for_df,
    get_audio_files_for_dropdown,
    refresh_audio_dropdown,
    upload_file,
    save_recording
)

# --- 기본 설정 및 디렉터리 생성 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, AUDIO_INPUT_DIR_NAME)
RESULTS_DIR = os.path.join(ROOT_DIR, RESULTS_DIR_NAME)
TEMP_DIR = os.path.join(ROOT_DIR, TEMP_DIR_NAME)

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# --- UI 헬퍼 함수 ---

def create_zoom_link(url):
    """Creates a clickable markdown link if the URL is a valid Zoom link."""
    if url and "zoom.us" in url:
        return gr.Markdown(f"➡️ <a href='{url}' target='_blank' style='color: blue; text-decoration: underline;'>클릭하여 Zoom 회의 열기</a>")
    elif not url:
        return gr.Markdown("")
    return gr.Markdown("<span style='color: red;'>유효한 Zoom 회의 링크를 입력해주세요.</span>")

# --- Gradio 콜백 함수 ---

def run_processing_and_update_ui(audio_filename, llm_choice, topic, keywords_str, progress=gr.Progress(track_tqdm=True)):
    """처리 파이프라인을 실행하고 UI를 업데이트합니다."""
    if not audio_filename:
        return "처리할 오디오 파일을 먼저 선택해주세요.", "", ""

    progress(0, desc="준비 중...")
    audio_path = os.path.join(DATA_DIR, audio_filename)
    keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]

    # 파이프라인 실행
    progress(0.1, desc="파이프라인 실행 시작...")
    results_path, message = run_pipeline(audio_path, llm_choice, topic, keywords)
    progress(0.9, desc="결과 파일 로딩 중...")

    if not results_path:
        return f"**처리 실패:** {message}", "", ""

    # 결과 파일 읽기
    summary = "요약 파일을 찾을 수 없습니다."
    corrected_text = "교정된 텍스트 파일을 찾을 수 없습니다."
    try:
        base_filename = os.path.splitext(audio_filename)[0]
        summary_path = os.path.join(results_path, f"summary_{base_filename}.md")
        corrected_path = os.path.join(results_path, f"corrected_{base_filename}.txt")
        
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = f.read()
        with open(corrected_path, 'r', encoding='utf-8') as f:
            corrected_text = f.read()
    except Exception as e:
        logging.error(f"결과 파일 읽기 오류: {e}")
        message += f"\n결과 파일 로딩 중 오류 발생: {e}"

    progress(1, desc="완료")
    return f"**{message}** 결과는 '{results_path}' 폴더에 저장되었습니다.", summary, corrected_text

# --- Gradio UI 빌드 ---
with gr.Blocks(theme=gr.themes.Soft(), title="AI 회의록 정리") as demo:
    gr.Markdown("<h1><center>Minute Factory: AI 회의록 정리</center></h1>")

    with gr.Tabs() as tabs:
        # --- Tab 1: File Management ---
        with gr.TabItem("음성 파일 관리"):
            gr.Markdown("음성/영상 파일을 업로드하거나 서버의 파일을 관리합니다. (mp4, m4a 등은 wav로 자동 변환)")
            with gr.Row():
                audio_list_df = gr.Dataframe(
                    headers=["파일명", "크기", "수정일"],
                    value=get_audio_files_for_df(DATA_DIR),
                    interactive=False
                )
            with gr.Row():
                with gr.Column():
                    file_uploader = gr.File(label="음성/영상 파일 업로드 (WAV 자동 변환)")
                    upload_status = gr.Markdown("")
            
        # --- Tab 2: Voice Recording ---
        with gr.TabItem("음성 녹음"):
            gr.Markdown("마이크를 사용하여 새 음성을 녹음하고 서버에 저장합니다.")
            record_status = gr.Markdown()
            mic_audio = gr.Audio(sources=["microphone"], type="filepath", label="음성 녹음")
            save_filename_box = gr.Textbox(label="저장할 파일명 (확장자 제외)", placeholder="예: 주간회의_240927")
            save_button = gr.Button("녹음 저장하기")

        # --- Tab 3: Zoom Meeting Guide ---
        with gr.TabItem("Zoom 회의"):
            gr.Markdown("## Zoom 회의 참여 및 녹화 안내")
            gr.Markdown(
                "**회의 참여:** 아래에 Zoom 회의 링크를 입력하면 참여할 수 있는 링크가 생성됩니다.\n"
                "**회의 녹화:** 이 앱은 Zoom 회의를 직접 녹화할 수 없습니다. 대신, Zoom의 자체 녹화 기능을 사용하세요."
            )
            with gr.Row():
                zoom_url_input = gr.Textbox(label="Zoom 회의 링크", placeholder="https://zoom.us/j/1234567890")
            zoom_link_output = gr.Markdown("")
            gr.Markdown(
                "### 녹화 파일을 업로드하는 방법\n"
                "1. Zoom 회의 중 '기록' (Record) 버튼을 눌러 **'이 컴퓨터에 기록'**을 선택합니다.\n"
                "2. 회의가 끝나면 녹화 파일이 동영상(.mp4) 또는 오디오(.m4a) 파일로 컴퓨터에 저장됩니다.\n"
                "3. **'1. 파일 업로드 및 관리'** 탭으로 이동하여 저장된 파일을 업로드하세요.\n"
                "4. 업로드된 파일은 자동으로 음성(.wav) 파일로 변환되어 목록에 추가됩니다."
            )

        # --- Tab 4: Processing & Summary ---
        with gr.TabItem("처리 & 요약"):
            gr.Markdown("오디오 파일을 선택하고 처리 및 요약을 실행합니다.")
            with gr.Row():
                audio_dropdown = gr.Dropdown(label="처리할 오디오 파일", choices=get_audio_files_for_dropdown(DATA_DIR), allow_custom_value=True)
                refresh_button = gr.Button("파일 목록 새로고침")
            
            gr.Markdown("회의 정보를 입력하세요.")
            with gr.Row():
                topic_input = gr.Textbox(label="회의 주제", value=DEFAULT_MEETING_TOPIC)
                keywords_input = gr.Textbox(label="주요 키워드 (쉼표로 구분)", value=", ".join(DEFAULT_KEYWORDS))
            
            llm_dropdown = gr.Radio(label="사용할 LLM", choices=AVAILABLE_LLMS, value=AVAILABLE_LLMS[0])
            
            start_button = gr.Button("처리 시작", variant="primary")
            process_status = gr.Markdown()

            with gr.Accordion("처리 결과 보기", open=True):
                summary_output = gr.Markdown(label="회의 요약")
                corrected_output = gr.Textbox(label="교정된 대화록", lines=15, interactive=False)

    # --- 이벤트 핸들러 연결 ---
    
    # 1. 파일 업로드
    file_uploader.upload(
        fn=lambda file, progress: upload_file(file, DATA_DIR, progress),
        inputs=[file_uploader],
        outputs=[upload_status, audio_list_df, audio_dropdown]
    )

    # 2. 녹음 저장
    save_button.click(
        fn=lambda temp_file, fname: save_recording(temp_file, fname, DATA_DIR),
        inputs=[mic_audio, save_filename_box],
        outputs=[record_status, audio_list_df, audio_dropdown]
    )

    # 3. Zoom 링크 생성
    zoom_url_input.change(
        fn=create_zoom_link,
        inputs=[zoom_url_input],
        outputs=[zoom_link_output]
    )

    # 4. 처리 시작
    start_button.click(
        fn=run_processing_and_update_ui,
        inputs=[audio_dropdown, llm_dropdown, topic_input, keywords_input],
        outputs=[process_status, summary_output, corrected_output]
    )

    # 새로고침 버튼
    refresh_button.click(fn=lambda: refresh_audio_dropdown(DATA_DIR), outputs=[audio_dropdown])

if __name__ == "__main__":
    demo.launch()
