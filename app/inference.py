from app.model_loader import load_model
from app.schemas import PredictRequest, PredictResponse
import torch

def predict(request: PredictRequest, model) -> PredictResponse:
    features = request.features
    features = torch.tensor(features, dtype=torch.float32)
    features = features.unsqueeze(0)
    with torch.no_grad():
        output = model(features)
        probs = torch.softmax(output, dim=1)
        risk_score = probs[0,1].item()
        label = torch.argmax(output, dim=1).item()
    return PredictResponse(risk_score=risk_score, label=label)

if __name__ == "__main__":
    request = PredictRequest(features=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0])
    model = load_model()
    response = predict(request, model)
    print(response)