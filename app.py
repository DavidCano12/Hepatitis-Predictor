from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os
import urllib.request

app = Flask(__name__)

# URLs de los archivos del modelo del profesor
MODEL_URL = 'https://raw.githubusercontent.com/aperezn298/CienciaDatosSENA/main/07ModeloIAHepatitis/modelo_regresion_logistica.pkl'
SCALER_URL = 'https://raw.githubusercontent.com/aperezn298/CienciaDatosSENA/main/07ModeloIAHepatitis/scaler.pkl'

# Crear directorio temporal si no existe
if not os.path.exists('/tmp'):
    os.makedirs('/tmp')

model = None
scaler = None

def load_model():
    global model, scaler
    try:
        # Descargar modelo
        model_path = '/tmp/modelo.pkl'
        if not os.path.exists(model_path):
            urllib.request.urlretrieve(MODEL_URL, model_path)
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Descargar scaler
        scaler_path = '/tmp/scaler.pkl'
        if not os.path.exists(scaler_path):
            urllib.request.urlretrieve(SCALER_URL, scaler_path)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        print("Modelo y scaler cargados exitosamente")
        return True
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        return False

# Cargar modelo al iniciar
load_model()

# Orden de features del modelo (21 características)
FEATURE_NAMES = [
    'Age', 'Sex', 'Estado_Civil', 'Ciudad', 'Steroid', 'Antivirals',
    'Fatigue', 'Malaise', 'Anorexia', 'Liver_Big', 'Liver_Firm',
    'Spleen_Palpable', 'Spiders', 'Ascites', 'Varices', 'Bilirubin',
    'Alk_Phosphate', 'Sgot', 'Albumin', 'Protime', 'Histology'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Mapeo de valores categóricos
        sex_map = {'Masculino': 1, 'Femenino': 0}
        estado_civil_map = {'Soltero': 0, 'Casado': 1, 'Divorciado': 2, 'Viudo': 3}
        histology_map = {'No': 0, 'Sí': 1}
        
        # Preparar características en el orden correcto
        features = [
            float(data.get('age', 0)),
            sex_map.get(data.get('sex', 'Masculino'), 1),
            estado_civil_map.get(data.get('estado_civil', 'Soltero'), 0),
            float(data.get('ciudad_code', 0)),  # Código numérico de ciudad
            1 if data.get('steroid') == 'Si' else 0,
            1 if data.get('antivirals') == 'Si' else 0,
            1 if data.get('fatigue') == 'Si' else 0,
            1 if data.get('malaise') == 'Si' else 0,
            1 if data.get('anorexia') == 'Si' else 0,
            1 if data.get('liver_big') == 'Si' else 0,
            1 if data.get('liver_firm') == 'Si' else 0,
            1 if data.get('spleen_palpable') == 'Si' else 0,
            1 if data.get('spiders') == 'Si' else 0,
            1 if data.get('ascites') == 'Si' else 0,
            1 if data.get('varices') == 'Si' else 0,
            float(data.get('bilirubin', 0)),
            float(data.get('alk_phosphate', 0)),
            float(data.get('sgot', 0)),
            float(data.get('albumin', 0)),
            float(data.get('protime', 0)),
            histology_map.get(data.get('histology', 'No'), 0)
        ]
        
        # Convertir a array numpy
        features_array = np.array(features).reshape(1, -1)
        
        # Escalar features
        if scaler is not None:
            features_scaled = scaler.transform(features_array)
        else:
            features_scaled = features_array
        
        # Hacer predicción
        if model is not None:
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0]
            
            result = {
                'prediction': int(prediction),
                'probability_negative': float(probability[0]),
                'probability_positive': float(probability[1]),
                'result': 'Positivo para Hepatitis' if prediction == 1 else 'Negativo para Hepatitis'
            }
        else:
            result = {'error': 'Modelo no cargado'}
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
