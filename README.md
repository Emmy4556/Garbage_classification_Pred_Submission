# Garbage Classification Predictor

A from-scratch CNN (PyTorch) trained to classify images into 10 waste
categories: cardboard, paper, plastic, metal, glass, biological, battery,
trash, shoes, clothes. Served behind a FastAPI endpoint with a simple
web UI, packaged in Docker.

## Run locally

```bash
docker build -t garbage-classifier .
docker run -p 8000:8000 garbage-classifier
```

Then open http://localhost:8000 in your browser, or call the API directly:

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@sample.jpg"
```

## Endpoints

- `GET /` — web UI
- `GET /health` — health check
- `GET /classes` — list of class labels
- `POST /predict` — accepts an image file, returns predicted class + probabilities
