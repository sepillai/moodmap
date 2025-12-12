from pydantic import BaseModel, Field
from typing import Dict, Optional

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