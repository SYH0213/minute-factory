# -*- coding: utf-8 -*-
import gradio as gr
import os
import logging
import re
import uuid
from slugify import slugify

# 프로젝트 모듈 임포트
from core_processing.main_pipeline import run_pipeline
from utils.config import (
    AUDIO_INPUT_DIR_NAME,
    RESULTS_DIR_NAME,
    AVAILABLE_LLMS,
    DEFAULT_MEETING_TOPIC,
    DEFAULT_KEYWORDS
)
from ui.input_handlers import (
    get_audio_files_for_df,
    get_audio_files_for_dropdown,
    refresh_audio_dropdown,
    upload_file,
    save_recording
)
from chatbot.crag_logic import run_query

# --- 기본 설정 ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT_DIR, AUDIO_INPUT_DIR_NAME)
RESULTS_DIR = os.path.join(ROOT_DIR, RESULTS_DIR_NAME)

# --- UI 헬퍼 함수 ---

def create_zoom_link(url):
    """Creates a clickable markdown link if the URL is a valid Zoom link."""
    if url and "zoom.us" in url:
        return gr.Markdown(f">➡️ <a href='{url}' target='_blank' style='color: blue; text-decoration: underline;'>클릭하여 Zoom 회의 열기</a>")
    elif not url:
        return gr.Markdown("")
    return gr.Markdown("<span style='color: red;'>유효한 Zoom 회의 링크를 입력해주세요.</span>")

def get_processed_meetings():
    """처리된 회의록 목록을 스캔하여 드롭다운용으로 반환합니다."""
    processed_meetings = []
    if not os.path.exists(RESULTS_DIR):
        return processed_meetings

    for folder_name in os.listdir(RESULTS_DIR):
        folder_path = os.path.join(RESULTS_DIR, folder_name)
        if os.path.isdir(folder_path):
            base_filename = '_'.join(folder_name.split('_')[:-1])
            corrected_file = f"corrected_{base_filename}.txt"
            if os.path.exists(os.path.join(folder_path, corrected_file)):
                # 1. Transliterate base_filename to an ASCII slug
                slugified_name = slugify(base_filename, separator='_', lowercase=True, replacements=[['.', '_']])
                # 2. Apply final sanitization (ChromaDB specific) - remove any remaining non-allowed chars
                collection_name = re.sub(r'[^a-zA-Z0-9._-]', '_', slugified_name)

                # 3. Ensure collection_name meets ChromaDB's rules (min 3 chars, starts/ends with alphanumeric)
                # Remove leading/trailing underscores that might violate start/end rule
                collection_name = collection_name.strip('_')
                
                # If it became empty after stripping, or is too short, use a fallback
                if not collection_name or len(collection_name) < 3:
                    # Fallback to a sanitized version of the full folder_name if base_filename is too short/empty
                    temp_name = slugify(folder_name, separator='_', lowercase=True, replacements=[['.', '_']])
                    temp_name = re.sub(r'[^a-zA-Z0-9._-]', '_', temp_name).strip('_')
                    if len(temp_name) >= 3:
                        collection_name = temp_name
                    else:
                        # Last resort: generate a unique, valid name
                        collection_name = "meeting_" + str(uuid.uuid4())[:8].replace('-', '_') # Ensure it's always valid and long enough
                processed_meetings.append((folder_name, collection_name))
    
    return sorted(processed_meetings, key=lambda x: x[0], reverse=True)

# --- Gradio 콜백 래퍼 함수 ---

def upload_wrapper(file, progress=gr.Progress(track_tqdm=True)):
    """upload_file에 DATA_DIR을 전달하고 Gradio 프로그레스 바를 활성화하는 래퍼 함수"""
    return upload_file(file, DATA_DIR, progress)

def save_recording_wrapper(temp_file, fname):
    """save_recording에 DATA_DIR을 전달하기 위한 래퍼 함수"""
    return save_recording(temp_file, fname, DATA_DIR)

# --- Gradio 콜백 함수 ---

def run_processing_and_update_ui(audio_filename, llm_choice, topic, keywords_str, progress=gr.Progress(track_tqdm=True)):
    """처리 파이프라인을 실행하고 UI를 업데이트합니다."""
    if not audio_filename:
        return "처리할 오디오 파일을 먼저 선택해주세요.", "", "", gr.Dropdown(choices=[name for name, _ in get_processed_meetings()]), {}

    progress(0, desc="준비 중...")
    audio_path = os.path.join(DATA_DIR, audio_filename)
    keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]

    results_path, message = run_pipeline(audio_path, llm_choice, topic, keywords)
    progress(0.9, desc="결과 파일 로딩 중...")

    if not results_path:
        return f"**처리 실패:** {message}", "", "", gr.Dropdown(choices=[name for name, _ in get_processed_meetings()]), {}

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
    new_meetings = get_processed_meetings()
    return f"**{message}** 결과는 '{results_path}' 폴더에 저장되었습니다.", summary, corrected_text, gr.Dropdown(choices=[name for name, _ in new_meetings]), dict(new_meetings)

def handle_chat_message(user_question, history, collection_name):
    """챗봇 메시지를 처리하고 답변을 생성합니다. (messages 포맷)"""
    if not collection_name:
        history.append({"role": "user", "content": user_question})
        history.append({"role": "assistant", "content": "먼저 좌측 상단에서 대화할 회의록을 선택해주세요."})
        return history, ""
        
    history.append({"role": "user", "content": user_question})
    yield history, ""

    response = run_query(user_question, collection_name)
    history.append({"role": "assistant", "content": response})
    yield history, ""

# --- Q&A 탭 콜백 함수 ---

# 화자별 아이콘 리스트 (이모지)
SPEAKER_ICONS = ["😀", "😎", "😊", "🧑", "👩", "🤔", "🤓", "🤖"]

def load_meeting_data(selection, state):
    """드롭다운에서 회의록 선택 시 요약과 대화록을 로드합니다."""
    if not selection:
        return gr.update(value=None), gr.update(value=None), None

    # 1. 파일 경로 찾기
    base_filename = '_'.join(selection.split('_')[:-1])
    results_folder_path = os.path.join(RESULTS_DIR, selection)
    summary_path = os.path.join(results_folder_path, f"summary_{base_filename}.md")
    corrected_path = os.path.join(results_folder_path, f"corrected_{base_filename}.txt")

    # 2. 요약 파일 읽기
    summary_content = "요약 파일을 찾을 수 없습니다."
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary_content = f.read()
    except FileNotFoundError:
        logging.warning(f"요약 파일 없음: {summary_path}")
    except Exception as e:
        logging.error(f"요약 파일 읽기 오류: {e}")

    # 3. 대화록 파일 읽고 파싱하기 (chat.png UI 처럼)
    transcript_chat_history = []
    speaker_icon_map = {}
    icon_index = 0
    try:
        with open(corrected_path, 'r', encoding='utf-8') as f:
            full_text = f.read() # 파일 전체를 하나의 문자열로 읽기
        
        lines = full_text.split('\n') # 문자열을 줄바꿈 기준으로 나누어 리스트 생성

        for line in lines:
            if not line.strip(): # 빈 줄은 건너뛰기
                continue

            match = re.search(r'\[.*?s - .*?s\]\s*(.*?):\s*(.*)', line)
            if match:
                speaker, text = match.groups()
                speaker = speaker.strip()
                
                # 새로운 화자일 경우, 아이콘 리스트에서 아이콘 할당
                if speaker not in speaker_icon_map:
                    speaker_icon_map[speaker] = SPEAKER_ICONS[icon_index % len(SPEAKER_ICONS)]
                    icon_index += 1
                
                icon = speaker_icon_map[speaker]
                
                # 아이콘, 화자 이름, 대화 내용을 포함하는 왼쪽 정렬 말풍선 생성
                chat_message = f"{icon} **{speaker}**\n{text.strip()}"
                transcript_chat_history.append((chat_message, None))
            else:
                transcript_chat_history.append((line.strip(), None))
    except FileNotFoundError:
        logging.warning(f"대화록 파일 없음: {corrected_path}")
        transcript_chat_history = [("대화록 파일을 찾을 수 없습니다.", None)]
    except Exception as e:
        logging.error(f"대화록 파일 읽기 오류: {e}")
        transcript_chat_history = [("대화록 파일 로딩 중 오류 발생: " + str(e), None)]

    # 4. Collection 이름 가져오기
    collection_name = state.get(selection)

    return summary_content, transcript_chat_history, collection_name



# --- Gradio UI 빌드 ---
def create_ui():
    with gr.Blocks(theme=gr.themes.Soft(), title="AI 회의록 정리") as demo:
        gr.Markdown("<h1><center>Minute Factory: AI 회의록 정리</center></h1>")

        with gr.Tabs():
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
                
            with gr.TabItem("음성 녹음"):
                gr.Markdown("마이크를 사용하여 새 음성을 녹음하고 서버에 저장합니다.")
                record_status = gr.Markdown()
                mic_audio = gr.Audio(sources=["microphone"], type="filepath", label="음성 녹음")
                save_filename_box = gr.Textbox(label="저장할 파일명 (확장자 제외)", placeholder="예: 주간회의_240927")
                save_button = gr.Button("녹음 저장하기")

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

            with gr.TabItem("회의록 검색 Q&A"):
                with gr.Column():
                    with gr.Row():
                        chatbot_meeting_selector = gr.Dropdown(
                            label="대화할 회의록 선택", 
                            choices=[name for name, _ in get_processed_meetings()],
                            value=None
                        )
                        chatbot_refresh_button = gr.Button("회의록 목록 새로고침")
                    
                    with gr.Tabs():
                        with gr.TabItem("📜 대화록"):
                            transcript_output = gr.Chatbot(label="전체 대화 내용", height=500)
                        with gr.TabItem("📝 요약"):
                            summary_output_qa = gr.Markdown(label="회의 요약 내용")
                        with gr.TabItem("❓ 질문하기"):
                            chatbot_history = gr.Chatbot(label="대화 내용", height=500, type="messages")
                            chatbot_question = gr.Textbox(label="질문 입력", placeholder="회의록 내용을 기반으로 질문을 입력하세요...")
                            chatbot_submit_button = gr.Button("전송", variant="primary")

                available_meetings_state = gr.State(dict(get_processed_meetings()))
                selected_collection_state = gr.State()

        # --- 이벤트 핸들러 연결 ---
        
        file_uploader.upload(
            fn=upload_wrapper,
            inputs=[file_uploader],
            outputs=[upload_status, audio_list_df, audio_dropdown]
        )

        save_button.click(
            fn=save_recording_wrapper,
            inputs=[mic_audio, save_filename_box],
            outputs=[record_status, audio_list_df, audio_dropdown]
        )

        zoom_url_input.change(
            fn=create_zoom_link,
            inputs=[zoom_url_input],
            outputs=[zoom_link_output]
        )

        refresh_button.click(fn=lambda: refresh_audio_dropdown(DATA_DIR), outputs=[audio_dropdown])

        # Q&A 탭 이벤트 핸들러
        def refresh_chatbot_dropdown():
            new_meetings = get_processed_meetings()
            return gr.Dropdown(choices=[name for name, _ in new_meetings]), dict(new_meetings)

        chatbot_refresh_button.click(
            fn=refresh_chatbot_dropdown,
            outputs=[chatbot_meeting_selector, available_meetings_state]
        )
        
        # 처리 & 요약 탭에서 처리가 완료되면 Q&A 탭의 드롭다운과 상태를 함께 업데이트
        start_button.click(
            fn=run_processing_and_update_ui,
            inputs=[audio_dropdown, llm_dropdown, topic_input, keywords_input],
            outputs=[process_status, summary_output, corrected_output, chatbot_meeting_selector, available_meetings_state]
        )

        # 회의록 선택 시 데이터 로드
        chatbot_meeting_selector.change(
            fn=load_meeting_data,
            inputs=[chatbot_meeting_selector, available_meetings_state],
            outputs=[summary_output_qa, transcript_output, selected_collection_state]
        )

        # 챗봇 질문/답변
        chatbot_submit_button.click(
            fn=handle_chat_message,
            inputs=[chatbot_question, chatbot_history, selected_collection_state],
            outputs=[chatbot_history, chatbot_question]
        )
        chatbot_question.submit(
            fn=handle_chat_message,
            inputs=[chatbot_question, chatbot_history, selected_collection_state],
            outputs=[chatbot_history, chatbot_question]
        )
        
        return demo
