# Production-Ready ML Inference Service

A real-time **Credit Card Default Prediction** API: train a PyTorch model on the UCI Credit Card Default dataset, save artifacts, and serve predictions via FastAPI with input validation and logging.

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  data/data.xls  │────▶│  training/       │────▶│  artifacts/     │────▶│  app/           │
│  (UCI dataset)  │     │  dataset.py      │     │  model.pth      │     │  model_loader   │
│                 │     │  train.py        │     │  config.json    │     │  inference      │
└─────────────────┘     └──────────────────┘     └─────────────────┘     │  schemas        │
                                     │                        │           │  main (FastAPI) │
                                     │                        │           └────────┬────────┘
                                     ▼                        │                    │
                              PyTorch MLP                     │                    ▼
                              (23 → 12 → 2)                   └──────────▶  POST /predict
                                                                           GET  /
```

**Flow:** Data → Dataset + Train → Save model & config → API loads model at startup → Client sends 23 features → API returns `risk_score` and `label`.

---

## Tech Stack

- **Python 3.10+**
- **PyTorch** – model training and inference
- **FastAPI** – API server
- **Pydantic** – request/response validation
- **Docker** – containerized service (optional)
- **UCI Credit Card Default** dataset

---

## Repository Structure

```
├── app/
│   ├── main.py          # FastAPI app, / and /predict endpoints
│   ├── schemas.py        # PredictRequest, PredictResponse (Pydantic)
│   ├── inference.py      # predict(request, model) → response
│   └── model_loader.py   # load_model() from artifacts
├── training/
│   ├── dataset.py       # CreditCardDataset (PyTorch Dataset)
│   └── train.py         # Train MLP, save model + config to artifacts/
├── artifacts/
│   ├── model.pth        # Saved model state_dict
│   └── config.json     # input_dim, hidden_dim, num_classes, model_path
├── data/
│   └── data.xls        # UCI Credit Card Default dataset
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## How to Run Locally

### 1. Clone and set up

```bash
cd /path/to/project
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. (Optional) Train the model

If `artifacts/model.pth` and `artifacts/config.json` are not present, train first:

```bash
# From project root; training expects data at ./data/data.xls
python3 -m training.train
```

This writes `artifacts/model.pth` and `artifacts/config.json`.

### 3. Run the API

```bash
# From project root
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- API: **http://localhost:8000**
- Interactive docs: **http://localhost:8000/docs**

---

## Example API Request / Response

**GET /** (health/welcome)

```bash
curl http://localhost:8000/
```

Response:

```json
{"message": "Welcome to the Credit Card Default Prediction API"}
```

---

**POST /predict** (risk prediction)

Request body: exactly **23 floats** in the same order as the training features (LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0–PAY_6, BILL_AMT1–6, PAY_AMT1–6).

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [20000, 2, 2, 1, 24, 2, 2, -1, -1, -2, -2, 3913, 3102, 689, 0, 0, 0, 0, 689, 0, 0, 0, 0]}'
```

Response:

```json
{"risk_score": 0.7623466849327087, "label": 1}
```

- **risk_score**: probability of default (0–1).
- **label**: predicted class (0 = no default, 1 = default).

---

## Run with Docker

From the project root:

```bash
docker build -t credit-card-api .
docker run -p 8000:8000 credit-card-api
```

Then call **http://localhost:8000/** and **http://localhost:8000/predict** as above.

**Note:** The Dockerfile must copy both `app/` and `artifacts/` so the container has the model and config. If your Dockerfile only copies `app/`, add `COPY artifacts/ ./artifacts/` and use `COPY app/ ./app/` so the `app` package and artifacts are available at runtime.

---

## License

MIT (or your choice).
