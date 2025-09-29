# -*- coding: utf-8 -*-
import gradio as gr
import os
import shutil
from datetime import datetime
import logging

def get_audio_files_for_dropdown(data_dir):
    """'data' 폴더에 있는 오디오 파일 목록을 드롭다운용으로 반환합니다."""
    try:
        return sorted([f for f in os.listdir(data_dir) if f.lower().endswith(('.wav', '.mp3', '.m4a'))])
    except FileNotFoundError:
        return []

def get_audio_files_for_df(data_dir):
    """'data' 폴더의 파일 목록을 데이터프레임용으로 상세 정보와 함께 반환합니다."""
    files_with_details = []
    try:
        for f in sorted(os.listdir(data_dir)):
            path = os.path.join(data_dir, f)
            if os.path.isfile(path) and f.lower().endswith(('.wav', '.mp3', '.m4a')):
                size_kb = f"{os.path.getsize(path) / 1024:.2f} KB"
                mod_time = datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y-%m-%d %H:%M')
                files_with_details.append([f, size_kb, mod_time])
    except Exception as e:
        logging.error(f"파일 목록(DF) 로딩 중 오류: {e}")
    return files_with_details

def refresh_audio_dropdown(data_dir):
    """오디오 파일 드롭다운을 최신 상태로 업데이트합니다."""
    return gr.Dropdown(choices=get_audio_files_for_dropdown(data_dir))

def refresh_audio_df(data_dir):
    """오디오 파일 데이터프레임을 최신 상태로 업데이트합니다."""
    return gr.Dataframe(value=get_audio_files_for_df(data_dir))

def upload_file(file_obj, data_dir, progress=gr.Progress(track_tqdm=True)):
    """파일을 업로드하고 WAV로 변환합니다."""
    if file_obj is None:
        return gr.Markdown("파일이 선택되지 않았습니다."), refresh_audio_df(data_dir), gr.Dropdown(choices=get_audio_files_for_dropdown(data_dir))

    try:
        from pydub import AudioSegment
    except ImportError:
        return gr.Markdown("오류: pydub 라이브러리가 필요합니다. (pip install pydub)"), refresh_audio_df(data_dir), gr.Dropdown(choices=get_audio_files_for_dropdown(data_dir))

    original_path = file_obj.name
    filename = os.path.basename(original_path)
    filename_base, ext = os.path.splitext(filename)
    wav_path = os.path.join(data_dir, f"{filename_base}.wav")

    try:
        progress(0, desc="파일 변환 중...")
        audio = AudioSegment.from_file(original_path)
        audio.export(wav_path, format="wav")
        status = f"'{filename}'이(가) '{os.path.basename(wav_path)}'(으)로 변환되어 저장되었습니다."
    except Exception as e:
        status = f"파일 변환 중 오류 발생: {e}"
        logging.error(status)

    return gr.Markdown(status), gr.Dataframe(value=get_audio_files_for_df(data_dir)), gr.Dropdown(choices=get_audio_files_for_dropdown(data_dir), value=os.path.basename(wav_path))

def save_recording(temp_filepath, filename, data_dir):
    """녹음 파일을 저장합니다."""
    if temp_filepath is None:
        return gr.Markdown("녹음된 파일이 없습니다."), refresh_audio_df(data_dir), gr.Dropdown(choices=get_audio_files_for_dropdown(data_dir))
    
    filename = filename or f"record_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    filename = filename if filename.lower().endswith('.wav') else f"{filename}.wav"
    destination_path = os.path.join(data_dir, filename)
    shutil.move(temp_filepath, destination_path)
    
    status = f"'{filename}'(으)로 녹음을 저장했습니다."
    return gr.Markdown(status), gr.Dataframe(value=get_audio_files_for_df(data_dir)), gr.Dropdown(choices=get_audio_files_for_dropdown(data_dir), value=filename)
