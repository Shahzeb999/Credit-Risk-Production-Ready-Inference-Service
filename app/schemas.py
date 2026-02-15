# define request/response shaped and validation with Pydantic

from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    features: list[float] = Field(..., min_length=23, max_length=23, description="The features to predict")

class PredictResponse(BaseModel):
    risk_score: float = Field(..., description="The risk score of the prediction")
    label: int = Field(..., description="The label of the prediction")


