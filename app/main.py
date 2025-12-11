from fastapi import FastAPI, UploadFile, File 
from fastapi.responses import JSONResponse

from app.audio_io import save_and_convert_audio
from app.feature_extractor import extract_features

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/tracks/")
async def upload_track(file: UploadFile = File(...)):
    wav_path = await save_and_convert_audio(file)

    features = extract_features(wav_path)

    return JSONResponse({
        "filename": file.filename,
        "wav_path": wav_path,
        "features": features
    })