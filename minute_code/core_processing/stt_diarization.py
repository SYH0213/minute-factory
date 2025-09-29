# -*- coding: utf-8 -*-
import os
import logging
import torch
from pyannote.audio import Pipeline
from utils.api_keys import get_pyannote_token

def diarize_audio(audio_path):
    """
    pyannote.audio를 사용하여 오디오 파일의 화자를 분리합니다.
    """
    token = get_pyannote_token()
    if not token:
        logging.error("Pyannote 토큰이 없어 화자 분리를 진행할 수 없습니다.")
        return None
    if not os.path.exists(audio_path):
        logging.error(f"오디오 파일을 찾을 수 없습니다: {audio_path}")
        return None
    
    logging.info(f"오디오 파일({audio_path})에 대한 화자 분리를 시작합니다...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Pyannote: 사용 장치를 '{device}'(으)로 설정합니다.")

        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
        pipeline.to(device)

        diarization = pipeline(audio_path)
        logging.info("화자 분리 완료.")
        return diarization
    except Exception as e:
        logging.error(f"화자 분리 중 오류 발생: {e}")
        return None

def transcribe_segment(client, audio_segment, segment_path, prompt, model):
    """
    Whisper API를 사용하여 오디오 세그먼트를 텍스트로 변환합니다.
    """
    try:
        audio_segment.export(segment_path, format="wav")
        with open(segment_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                prompt=prompt
            )
        return transcript.text
    except Exception as e:
        logging.error(f"Whisper API 호출 중 오류 발생: {e}")
        return ""
    finally:
        if os.path.exists(segment_path):
            os.remove(segment_path)
