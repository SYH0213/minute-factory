# -*- coding: utf-8 -*-
import gradio as gr
import os
import shutil
from datetime import datetime
import logging

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

def get_audio_files_for_dropdown():
    """'data' 폴더에 있는 오디오 파일 목록을 드롭다운용으로 반환합니다."""
    try:
        return sorted([f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.wav', '.mp3', '.m4a'))])
    except FileNotFoundError:
        return []

def get_audio_files_for_df():
    """'data' 폴더의 파일 목록을 데이터프레임용으로 상세 정보와 함께 반환합니다."""
    files_with_details = []
    try:
        for f in sorted(os.listdir(DATA_DIR)):
            path = os.path.join(DATA_DIR, f)
            if os.path.isfile(path) and f.lower().endswith(('.wav', '.mp3', '.m4a')):
                size_kb = f"{os.path.getsize(path) / 1024:.2f} KB"
                mod_time = datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y-%m-%d %H:%M')
                files_with_details.append([f, size_kb, mod_time])
    except Exception as e:
        logging.error(f"파일 목록(DF) 로딩 중 오류: {e}")
    return files_with_details

def refresh_audio_dropdown():
    """오디오 파일 드롭다운을 최신 상태로 업데이트합니다."""
    return gr.Dropdown(choices=get_audio_files_for_dropdown())

def refresh_audio_df():
    """오디오 파일 데이터프레임을 최신 상태로 업데이트합니다."""
    return gr.Dataframe(value=get_audio_files_for_df())

def create_zoom_link(url):
    """Creates a clickable markdown link if the URL is a valid Zoom link."""
    if url and "zoom.us" in url:
        return gr.Markdown(f"➡️ <a href='{url}' target='_blank' style='color: blue; text-decoration: underline;'>클릭하여 Zoom 회의 열기</a>")
    elif not url:
        return gr.Markdown("")
    return gr.Markdown("<span style='color: red;'>유효한 Zoom 회의 링크를 입력해주세요.</span>")

# --- Gradio 콜백 함수 ---

def upload_file(file_obj, progress=gr.Progress(track_tqdm=True)):
    """파일을 업로드하고 WAV로 변환합니다."""
    if file_obj is None:
        return gr.Markdown("파일이 선택되지 않았습니다."), refresh_audio_df(), refresh_audio_dropdown()

    try:
        from pydub import AudioSegment
    except ImportError:
        return gr.Markdown("오류: pydub 라이브러리가 필요합니다. (pip install pydub)"), refresh_audio_df(), refresh_audio_dropdown()

    original_path = file_obj.name
    filename = os.path.basename(original_path)
    filename_base, ext = os.path.splitext(filename)
    wav_path = os.path.join(DATA_DIR, f"{filename_base}.wav")

    try:
        progress(0, desc="파일 변환 중...")
        audio = AudioSegment.from_file(original_path)
        audio.export(wav_path, format="wav")
        status = f"'{filename}'이(가) '{os.path.basename(wav_path)}'(으)로 변환되어 저장되었습니다."
    except Exception as e:
        status = f"파일 변환 중 오류 발생: {e}"
        logging.error(status)

    return gr.Markdown(status), gr.Dataframe(value=get_audio_files_for_df()), gr.Dropdown(choices=get_audio_files_for_dropdown(), value=os.path.basename(wav_path))

def save_recording(temp_filepath, filename):
    """녹음 파일을 저장합니다."""
    if temp_filepath is None:
        return gr.Markdown("녹음된 파일이 없습니다."), refresh_audio_df(), refresh_audio_dropdown()
    
    filename = filename or f"record_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    filename = filename if filename.lower().endswith('.wav') else f"{filename}.wav"
    destination_path = os.path.join(DATA_DIR, filename)
    shutil.move(temp_filepath, destination_path)
    
    status = f"'{filename}'(으)로 녹음을 저장했습니다."
    return gr.Markdown(status), gr.Dataframe(value=get_audio_files_for_df()), gr.Dropdown(choices=get_audio_files_for_dropdown(), value=filename)

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
                    value=get_audio_files_for_df(),
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
                zoom_url_input = gr.Textbox(label="Zoom 회의 링크", placeholder="https://zoom.us/j/ப்புகளை")
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
                audio_dropdown = gr.Dropdown(label="처리할 오디오 파일", choices=get_audio_files_for_dropdown(), allow_custom_value=True)
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
        fn=upload_file,
        inputs=[file_uploader],
        outputs=[upload_status, audio_list_df, audio_dropdown]
    )

    # 2. 녹음 저장
    save_button.click(
        fn=save_recording,
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
    refresh_button.click(fn=refresh_audio_dropdown, outputs=[audio_dropdown])

if __name__ == "__main__":
    demo.launch()
