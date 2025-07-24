import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
import pandas as pd
import joblib

from src import  model

client = None


def setup_module(module):
    X = pd.DataFrame({
        "objective_len": [10, 20],
        "title_len": [5, 7],
        "remuneracao": [2000.0, 3000.0],
        "job_title_len": [7, 8],
        "job_level": [1, 2],
        "job_english": [1, 2],
        "job_area_len": [4, 5],
    })
    y = [0, 1]
    X_feat = model.add_features(X)
    clf = LogisticRegression().fit(X_feat, y)
    joblib.dump(clf, model.MODEL_PATH)
    global client
    from src.app import app
    client = TestClient(app)


def test_predict_endpoint(tmp_path, monkeypatch):
    payload = {
        "objective_len": 15,
        "title_len": 6,
        "remuneracao": 2500.0,
        "job_title_len": 7,
        "job_level": 1,
        "job_english": 1,
        "job_area_len": 4,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "match" in data and "probability" in data
