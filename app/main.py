from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import os
import uuid

from app.audio_io import save_and_convert_audio, get_audio_file_path
from app.feature_extractor import extract_features
from app.variation_engine import apply_variation_chain, FFmpegError
from app.prompt_interpreter import interpret
from app.models import (
    VariationRequest, VariationResponse,
    FeedbackRequest, FeedbackResponse, FeedbackStatsResponse,
    RecommendationsResponse, RecommendationItem
)
from app.database import store_variation
from app.feedback_service import submit_feedback, get_variation_feedback_stats

app = FastAPI(
    title="MoodMap API",
    description="Audio variation and ML-powered recommendation API",
    version="1.0.0"
)

# CORS middleware - allow frontend to make requests
# In production, set CORS_ORIGINS env var (comma-separated URLs)
# For local dev, allow localhost
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:3001").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Load ML models on startup."""
    try:
        from app.ml.model_loader import load_models
        load_models()
    except Exception as e:
        print(f"Note: ML models not loaded (will use keyword system): {e}")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/audio/{filename}")
def get_audio(filename: str):
    """
    Serve WAV files from uploaded_audio/ so the browser/frontend can play/download them.
    """
    path = Path("uploaded_audio") / filename

    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(path, media_type="audio/wav", filename=filename)


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
    
    # Extract features from original track for ML training
    try:
        original_features = extract_features(wav_path)
    except Exception:
        original_features = None  # Don't fail if feature extraction fails
    
    # Store variation in database for ML training
    try:
        store_variation(
            variation_id=variation_id,
            track_id=track_id,
            prompt=body.prompt,
            params=params,
            audio_features=original_features
        )
    except Exception:
        # Don't fail the request if database storage fails
        pass

    return VariationResponse(
        track_id=track_id,
        variation_id=variation_id,
        params={k: float(v) for k, v in params.items()},
        output_path=output_path,
        status="success"
    )


@app.post("/variations/{variation_id}/feedback", response_model=FeedbackResponse)
async def create_feedback(variation_id: str, body: FeedbackRequest):
    """
    Submit feedback (rating) for a variation.
    Rating scale: 1 (poor) to 5 (excellent)
    """
    try:
        result = submit_feedback(variation_id, body.rating, body.comment)
        return FeedbackResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")


@app.get("/variations/{variation_id}/feedback/stats", response_model=FeedbackStatsResponse)
async def get_feedback_stats(variation_id: str):
    """Get feedback statistics for a variation."""
    try:
        stats = get_variation_feedback_stats(variation_id)
        return FeedbackStatsResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get feedback stats: {str(e)}")


@app.get("/tracks/{track_id}/recommendations", response_model=RecommendationsResponse)
async def get_recommendations(track_id: str, prompt: str, num_recommendations: int = 5):
    """
    Get ML-powered recommendations for audio parameters based on prompt.
    
    Generates multiple candidate parameter sets and scores them using feedback learning.
    Returns top recommendations sorted by predicted rating.
    
    Args:
        track_id: ID of the track
        prompt: Text prompt describing desired transformation
        num_recommendations: Number of recommendations to return (default: 5, max: 10)
    
    Returns:
        RecommendationsResponse with top parameter sets and predicted ratings
    """
    # Validate track exists
    try:
        get_audio_file_path(track_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Track not found")
    
    # Validate num_recommendations
    num_recommendations = max(1, min(10, num_recommendations))
    
    try:
        from app.ml.recommender import get_recommender
        
        recommender = get_recommender()
        result = recommender.get_recommendations_with_explanation(
            track_id=track_id,
            prompt=prompt,
            num_recommendations=num_recommendations
        )
        
        # Convert to response model
        recommendations = [
            RecommendationItem(**rec) for rec in result["recommendations"]
        ]
        
        return RecommendationsResponse(
            track_id=result["track_id"],
            prompt=result["prompt"],
            num_recommendations=result["num_recommendations"],
            recommendations=recommendations,
            model_trained=result["model_trained"],
            explanation=result["explanation"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate recommendations: {str(e)}"
        )