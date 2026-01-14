from pydantic import BaseModel, Field
from typing import Dict, Optional, List

class VariationRequest(BaseModel):
    prompt: str = Field(..., description = "The prompt for the audio transformation")
    tempo_factor: Optional[float] = Field(None, description = "The tempo factor for the variation (optional, will be interpreted from prompt if not provided)")
    brightness_db: Optional[float] = Field(None, description = "The brightness factor for the variation (optional, will be interpreted from prompt if not provided)")
    bass_db: Optional[float] = Field(None, description = "The bass factor for the variation (optional, will be interpreted from prompt if not provided)")
    reverb: Optional[float] = Field(None, description = "The reverb factor for the variation (optional, will be interpreted from prompt if not provided)")
    compression: Optional[float] = Field(None, description = "The compression factor for the variation (optional, will be interpreted from prompt if not provided)")

class VariationResponse(BaseModel):
    track_id: str = Field(..., description = "The track ID of the original audio")
    variation_id: str = Field(..., description = "The variation ID (UUID from output filename)")
    params: Dict[str, float] = Field(..., description = "Final parameters used for the variation")
    output_path: str = Field(..., description = "The path to the generated variation WAV file")
    status: str = Field(..., description = "Status of the variation creation")

class FeedbackRequest(BaseModel):
    rating: int = Field(..., ge=1, le=5, description = "Rating from 1 (poor) to 5 (excellent)")
    comment: Optional[str] = Field(None, description = "Optional comment about the variation")

class FeedbackResponse(BaseModel):
    feedback_id: str = Field(..., description = "The feedback ID")
    variation_id: str = Field(..., description = "The variation ID")
    rating: int = Field(..., description = "The rating provided")
    comment: Optional[str] = Field(None, description = "Optional comment")
    status: str = Field(..., description = "Status of feedback submission")

class FeedbackStatsResponse(BaseModel):
    variation_id: str = Field(..., description = "The variation ID")
    average_rating: Optional[float] = Field(None, description = "Average rating (1-5)")
    count: int = Field(..., description = "Number of ratings")
    min_rating: Optional[int] = Field(None, description = "Minimum rating")
    max_rating: Optional[int] = Field(None, description = "Maximum rating")

class RecommendationItem(BaseModel):
    params: Dict[str, float] = Field(..., description = "Recommended audio parameters")
    predicted_rating: float = Field(..., description = "Predicted rating (1-5)")
    strategy: str = Field(..., description = "Strategy used to generate this recommendation")
    confidence: str = Field(..., description = "Confidence level: high, medium, or low")

class RecommendationsResponse(BaseModel):
    track_id: str = Field(..., description = "The track ID")
    prompt: str = Field(..., description = "The prompt used")
    num_recommendations: int = Field(..., description = "Number of recommendations returned")
    recommendations: List[RecommendationItem] = Field(..., description = "List of recommended parameter sets, sorted by predicted rating")
    model_trained: bool = Field(..., description = "Whether the feedback learning model is trained")
    explanation: str = Field(..., description = "Explanation of how recommendations were generated")