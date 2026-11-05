import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

modelo  = joblib.load("modelo_final.pkl")
scaler  = joblib.load("scaler.pkl")
encoder = joblib.load("label_encoder.pkl")

N_FEATURES = scaler.n_features_in_
FEATURE_NAMES = list(getattr(scaler, "feature_names_in_",
                     ["latitude","longitude","mes","trimestre"][:N_FEATURES]))

print(f"✓ Modelo carregado. Features esperadas ({N_FEATURES}): {FEATURE_NAMES}")
print(f"✓ Classes: {list(encoder.classes_)}")

app = FastAPI(
    title="Retrospect — Predição de Espécies",
    description="CRISP-DM Fase 6 — iNaturalist Brasil",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class EntradaPredicao(BaseModel):
    latitude:   float
    longitude:  float
    mes:        int
    trimestre:  int

class RespostaPredicao(BaseModel):
    grupo_previsto: str
    confianca:      float
    top3:           list[dict]
    features_usadas: list[str]

@app.get("/")
def raiz():
    return {
        "status": "ok",
        "features_esperadas": N_FEATURES,
        "feature_names": FEATURE_NAMES,
        "classes": list(encoder.classes_),
    }

@app.post("/api/prever-especie", response_model=RespostaPredicao)
def prever(entrada: EntradaPredicao):
    valores_base = {
        "latitude":   entrada.latitude,
        "longitude":  entrada.longitude,
        "mes":        entrada.mes,
        "trimestre":  entrada.trimestre,
    }
    vetor = [valores_base.get(f, 0.0) for f in FEATURE_NAMES]

    try:
        X_norm = scaler.transform([vetor])
        proba  = modelo.predict_proba(X_norm)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    top3_idx = proba.argsort()[-3:][::-1]
    top3 = [
        {"grupo": encoder.classes_[i], "confianca": round(float(proba[i]), 4)}
        for i in top3_idx
    ]

    return RespostaPredicao(
        grupo_previsto=top3[0]["grupo"],
        confianca=top3[0]["confianca"],
        top3=top3,
        features_usadas=FEATURE_NAMES,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)