from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import uuid

from app.audio_io import save_and_convert_audio, get_audio_file_path
from app.feature_extractor import extract_features
from app.variation_engine import apply_variation_chain, FFmpegError
from app.prompt_interpreter import interpret
from app.models import VariationRequest, VariationResponse

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/tracks/")
async def upload_track(file: UploadFile = File(...)):
    wav_path = await save_and_convert_audio(file)

    features = extract_features(wav_path)
    track_id = str(uuid.uuid4())
    track_path = f"uploaded_audio/{track_id}.wav"
    os.rename(wav_path, track_path)

    return JSONResponse({
        "filename": file.filename,
        "wav_path": track_path,
        "features": features,
        "track_id": track_id
    })

@app.post("/tracks/{track_id}/variations/", response_model = VariationResponse)
async def create_variation(track_id: str, body: VariationRequest):
    try:
        wav_path = get_audio_file_path(track_id)
    except FileNotFoundError:
        raise HTTPException(status_code = 404, detail = "Track not found")
    
    # Interpret prompt to get default params
    try:
        interpreted_params = interpret(body.prompt)
    except Exception as e:
        raise HTTPException(status_code = 422, detail = f"Prompt interpretation failed: {str(e)}")
    
    # Merge provided params with interpreted params (provided params take precedence)
    params = {
        "tempo_factor": body.tempo_factor if body.tempo_factor is not None else interpreted_params.get("tempo_factor", 1.0),
        "brightness_db": body.brightness_db if body.brightness_db is not None else interpreted_params.get("brightness_db", 0.0),
        "bass_db": body.bass_db if body.bass_db is not None else interpreted_params.get("bass_db", 0.0),
        "reverb": body.reverb if body.reverb is not None else interpreted_params.get("reverb", 0.0),
        "compression": body.compression if body.compression is not None else interpreted_params.get("compression", 0.0),
    }
    
    try:
        output_path = apply_variation_chain(wav_path, params)
    except FFmpegError as e:
        raise HTTPException(status_code = 500, detail = f"Variation engine failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code = 500, detail = f"Unexpected processing error: {str(e)}")

    variation_id = os.path.splitext(os.path.basename(output_path))[0]

    return VariationResponse(
        track_id=track_id,
        variation_id=variation_id,
        params={k: float(v) for k, v in params.items()},
        output_path=output_path,
        status="success"
    )