import os
import uuid 
from fastapi import UploadFile
from pydub import AudioSegment

AUDIO_DIR = "uploaded_audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

async def save_and_convert_audio(file: UploadFile):
    file_id = str(uuid.uuid4())
    input_path = os.path.join(AUDIO_DIR, f"{file_id}_{file.filename}")
    wav_output_path = os.path.join(AUDIO_DIR, f"{file_id}.wav")

    with open(input_path, "wb") as f:
        f.write(await file.read())

    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(44100).set_channels(1)
    audio.export(wav_output_path, format = "wav")

    return wav_output_path

def get_audio_file_path(track_id: str) -> str:
    wav_path = os.path.join(AUDIO_DIR, f"{track_id}.wav")
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"Audio file not found: {wav_path}")
    return wav_path