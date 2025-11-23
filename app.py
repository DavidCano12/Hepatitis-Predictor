from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os

app = Flask(__name__)

# === Rutas de los archivos del modelo y scaler ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modelo_regresion_logistica (2).pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# === Cargar modelo de Regresión Logística y scaler al iniciar la app ===
with open(MODEL_PATH, "rb") as f:
    modelo = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)


# === Construir el vector de features en el MISMO orden del entrenamiento ===
# ['Age', 'Sex', 'Estado_Civil', 'Ciudad', 'Steroid', 'Antivirals',
#  'Fatigue', 'Malaise', 'Anorexia', 'Liver_Big', 'Liver_Firm',
#  'Spleen_Palpable', 'Spiders', 'Ascites', 'Varices',
#  'Bilirubin', 'Alk_Phosphate', 'Sgot', 'Albumin', 'Protime', 'Histology']

def construir_vector_features(data: dict) -> np.ndarray:
    # 1. Demográficos
    age = float(data.get('age', 0))
    sex = 1 if data.get('sex') == 'Masculino' else 0

    estado_civil_map = {
        'Soltero': 0,
        'Casado': 1,
        'Divorciado': 2,
        'Viudo': 3
    }
    estado_civil = estado_civil_map.get(data.get('estado_civil', 'Soltero'), 0)

    ciudad = float(data.get('ciudad_code', 1))

    # 2. Medicación
    steroid = 1 if data.get('steroid') == 'Si' else 0
    antivirals = 1 if data.get('antivirals') == 'Si' else 0

    # 3. Síntomas
    fatigue = 1 if data.get('fatigue') == 'Si' else 0
    malaise = 1 if data.get('malaise') == 'Si' else 0
    anorexia = 1 if data.get('anorexia') == 'Si' else 0

    # 4. Hallazgos físicos
    liver_big = 1 if data.get('liver_big') == 'Si' else 0
    liver_firm = 1 if data.get('liver_firm') == 'Si' else 0
    spleen_palpable = 1 if data.get('spleen_palpable') == 'Si' else 0
    spiders = 1 if data.get('spiders') == 'Si' else 0
    ascites = 1 if data.get('ascites') == 'Si' else 0
    varices = 1 if data.get('varices') == 'Si' else 0

    # 5. Laboratorio
    bilirubin = float(data.get('bilirubin', 0.5))
    alk_phosphate = float(data.get('alk_phosphate', 50))
    sgot = float(data.get('sgot', 25))
    albumin = float(data.get('albumin', 4))
    protime = float(data.get('protime', 12))

    # 6. Histología
    histology = 1 if data.get('histology') == 'Si' else 0

    # Vector final (1 fila x 21 columnas)
    X = np.array([[
        age,            # Age
        sex,            # Sex
        estado_civil,   # Estado_Civil
        ciudad,         # Ciudad
        steroid,        # Steroid
        antivirals,     # Antivirals
        fatigue,        # Fatigue
        malaise,        # Malaise
        anorexia,       # Anorexia
        liver_big,      # Liver_Big
        liver_firm,     # Liver_Firm
        spleen_palpable,# Spleen_Palpable
        spiders,        # Spiders
        ascites,        # Ascites
        varices,        # Varices
        bilirubin,      # Bilirubin
        alk_phosphate,  # Alk_Phosphate
        sgot,           # Sgot
        albumin,        # Albumin
        protime,        # Protime
        histology       # Histology
    ]], dtype=float)

    return X


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Soporta JSON (fetch) y form-data (submit normal)
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()

        # 1) Vector de características
        X = construir_vector_features(data)

        # 2) Escalar igual que en el entrenamiento
        X_scaled = scaler.transform(X)

        # 3) Probabilidades con la Regresión Logística
        proba = modelo.predict_proba(X_scaled)[0]   # [p_clase0, p_clase1]
        clases = list(modelo.classes_)

        # Suponemos que la clase "1" = positivo
        if 1 in clases:
            idx_pos = clases.index(1)
            idx_neg = 1 - idx_pos
        else:
            idx_pos = 1
            idx_neg = 0

        probability_positive = float(proba[idx_pos])
        probability_negative = float(proba[idx_neg])

        # 4) Predicción final (0/1)
        pred = int(modelo.predict(X_scaled)[0])
        texto = 'Positivo para Hepatitis' if pred == 1 else 'Negativo para Hepatitis'

        return jsonify({
            'prediction': pred,
            'probability_negative': probability_negative,
            'probability_positive': probability_positive,
            'result': texto
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
