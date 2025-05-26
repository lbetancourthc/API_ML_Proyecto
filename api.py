# api.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import numpy as np

# Cargar modelo, vectorizador, binarizador y stop words
model = joblib.load("modelo_entrenado.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
mlb = joblib.load("multilabel_binarizer.pkl")
stop_words = joblib.load("stopwords_list.pkl")

# Crear app
app = FastAPI()

# Definir el esquema de entrada
# Modelo de entrada
class MoviePlot(BaseModel):
    plot: str

# Función de limpieza de texto
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# Ruta principal para predicción
@app.post("/predict")
def predict_genres(data: MoviePlot):
    # Limpiar texto
    clean_plot = clean_text(data.plot)
    X_input = tfidf.transform([clean_plot])

    # Obtener probabilidades
    probs = model.predict_proba(X_input)
    probs_dict = {f"p_{genre}": float(prob) for genre, prob in zip(mlb.classes_, probs[0])}

    # Devolver los top 5 géneros con mayor probabilidad
    top_genres = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "top_genres": top_genres,
        "all_probabilities": probs_dict
    }
