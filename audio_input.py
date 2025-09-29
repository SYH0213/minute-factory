import gradio as gr
import os
import shutil
from datetime import datetime

# --- Configuration ---
STORAGE_DIR = "data"
os.makedirs(STORAGE_DIR, exist_ok=True)

# --- Helper Functions ---
def get_file_list_for_df():
    """Returns a list of lists for Gradio Dataframe, including file size and mod time."""
    files_with_details = []
    try:
        for f in sorted(os.listdir(STORAGE_DIR)):
            path = os.path.join(STORAGE_DIR, f)
            if os.path.isfile(path):
                size_kb = f"{os.path.getsize(path) / 1024:.2f} KB"
                mod_time = datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y-%m-%d %H:%M')
                files_with_details.append([f, size_kb, mod_time])
    except Exception as e:
        print(f"Error getting file list: {e}")
    return files_with_details

# --- Core Logic Functions ---
def upload_and_convert(file_obj, progress=gr.Progress(track_tqdm=True)):
    """Handles file upload, converts to WAV if necessary, and updates file list."""
    if file_obj is None:
        return get_file_list_for_df(), "파일이 선택되지 않았습니다."

    try:
        from pydub import AudioSegment
    except ImportError:
        return get_file_list_for_df(), "오류: pydub 라이브러리가 설치되지 않았습니다. (pip install pydub)"

    original_path = file_obj.name
    original_filename = os.path.basename(original_path)
    filename_base, original_ext = os.path.splitext(original_filename)
    
    final_wav_path = os.path.join(STORAGE_DIR, f"{filename_base}.wav")

    status_message = ""
    
    try:
        progress(0, desc="변환 준비 중...")
        if original_ext.lower() not in ['.wav', '.mp4', '.m4a']: # Handle video/audio formats
            status_message = f"'{original_filename}'을(를) WAV로 변환 중..."
            audio = AudioSegment.from_file(original_path)
            audio.export(final_wav_path, format="wav")
            status_message = f"'{original_filename}'을(를) WAV로 변환하여 저장했습니다."
        elif original_ext.lower() in ['.mp4', '.m4a']:
            status_message = f"'{original_filename}'에서 오디오를 추출하여 WAV로 변환 중..."
            audio = AudioSegment.from_file(original_path, format=original_ext.lower().replace('.', ''))
            audio.export(final_wav_path, format="wav")
            status_message = f"'{original_filename}'의 오디오를 WAV로 변환하여 저장했습니다."
        else: # Is a .wav file
            status_message = f"WAV 파일 '{original_filename}'을(를) 저장 중..."
            shutil.copy(original_path, final_wav_path)
            status_message = f"'{original_filename}'을(를) 저장했습니다."
            
    except FileNotFoundError:
        return get_file_list_for_df(), "치명적 오류: FFmpeg가 설치되지 않았거나 시스템 경로에 추가되지 않았습니다. mp4, m4a 등의 파일을 변환하려면 FFmpeg 설치가 필요합니다."
    except Exception as e:
        # For other errors that might mention ffmpeg
        if "ffmpeg" in str(e).lower():
            return get_file_list_for_df(), "치명적 오류: FFmpeg 관련 오류가 발생했습니다. FFmpeg가 올바르게 설치되었는지 확인해주세요."
        return get_file_list_for_df(), f"오류 발생: {e}"

    return get_file_list_for_df(), status_message

def save_recording(temp_filepath, filename):
    """Saves a new recording from the microphone."""
    if temp_filepath is None:
        return get_file_list_for_df(), "녹음된 파일이 없습니다."
    
    if not filename:
        filename = f"record_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if not filename.lower().endswith('.wav'):
        filename += ".wav"
        
    destination_path = os.path.join(STORAGE_DIR, filename)
    shutil.move(temp_filepath, destination_path)
    
    return get_file_list_for_df(), f"'{filename}'(으)로 녹음을 저장했습니다."

def create_zoom_link(url):
    """Creates a clickable markdown link if the URL is a valid Zoom link."""
    if url and "zoom.us" in url:
        return gr.Markdown(f"➡️ <a href='{url}' target='_blank' style='color: blue; text-decoration: underline;'>클릭하여 Zoom 회의 열기</a>")
    elif not url:
        return gr.Markdown("")
    return gr.Markdown("<span style='color: red;'>유효한 Zoom 회의 링크를 입력해주세요.</span>")

# --- Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft(), title="오디오 관리") as demo:
    gr.Markdown("# 모바일용 오디오 관리 프로토타입")

    with gr.Tabs():
        # --- Tab 1: File Management ---
        with gr.TabItem("음성 파일 관리"):
            gr.Markdown("서버에 저장된 음성 파일을 관리하고 새 파일을 업로드합니다. (mp4, m4a 파일에서 오디오 자동 추출)")
            with gr.Row():
                # This component will be updated by other tabs
                file_list_df = gr.Dataframe(
                    headers=["파일명", "크기", "수정일"],
                    value=get_file_list_for_df(),
                    interactive=False,
                    elem_id="file_list_df" # Add an element ID
                )
            with gr.Row():
                with gr.Column():
                    upload_file_obj = gr.File(label="음성/영상 파일 업로드 (WAV 자동 변환)")
                    upload_status = gr.Markdown("")
            
            upload_file_obj.upload(
                upload_and_convert,
                inputs=[upload_file_obj],
                outputs=[file_list_df, upload_status]
            )

        # --- Tab 2: Voice Recording ---
        with gr.TabItem("음성 녹음"):
            gr.Markdown("마이크를 사용하여 새 음성을 녹음하고 서버에 저장합니다.")
            
            mic_audio = gr.Audio(sources=["microphone"], type="filepath", label="음성 녹음")
            
            gr.Markdown("저장할 파일명을 입력하세요 (확장자 제외). 입력하지 않으면 자동으로 생성됩니다.")
            save_filename_box = gr.Textbox(label="파일명", placeholder="예: 회의록_240925")
            
            save_button = gr.Button("녹음 저장하기")
            record_status = gr.Markdown("")

            save_button.click(
                save_recording,
                inputs=[mic_audio, save_filename_box],
                outputs=[file_list_df, record_status]
            )

        # --- Tab 3: Zoom Meeting ---
        with gr.TabItem("Zoom 회의"):
            gr.Markdown("## Zoom 회의 참여 및 녹화 안내")
            gr.Markdown(
                "**회의 참여:** 아래에 Zoom 회의 링크를 입력하면 참여할 수 있는 링크가 생성됩니다.\n"
                "**회의 녹화:** 이 앱은 Zoom 회의를 직접 녹화할 수 없습니다. 대신, Zoom의 자체 녹화 기능을 사용하세요."
            )

            with gr.Row():
                zoom_url_input = gr.Textbox(label="Zoom 회의 링크", placeholder="https://zoom.us/j/...")
            
            zoom_link_output = gr.Markdown("")

            gr.Markdown(
                "### 녹화 파일을 업로드하는 방법\n"
                "1. Zoom 회의 중 '기록' (Record) 버튼을 눌러 **'이 컴퓨터에 기록'**을 선택합니다.\n"
                "2. 회의가 끝나면 녹화 파일이 동영상(.mp4) 또는 오디오(.m4a) 파일로 컴퓨터에 저장됩니다.\n"
                "3. 상단의 **'음성 파일 관리'** 탭으로 이동하여 저장된 파일을 업로드하세요.\n"
                "4. 업로드된 파일은 자동으로 음성(.wav) 파일로 변환되어 목록에 추가됩니다."
            )

            zoom_url_input.change(
                create_zoom_link,
                inputs=[zoom_url_input],
                outputs=[zoom_link_output]
            )

if __name__ == "__main__":
    demo.launch()