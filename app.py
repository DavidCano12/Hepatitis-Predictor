from flask import Flask, render_template, request, jsonify
import numpy as np
import os

app = Flask(__name__)

# Simple logistic function for hepatitis prediction
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Extract features in order
        age = float(data.get('age', 0))
        sex = 1 if data.get('sex') == 'Masculino' else 0
        estado_civil_map = {'Soltero': 0, 'Casado': 1, 'Divorciado': 2, 'Viudo': 3}
        estado_civil = estado_civil_map.get(data.get('estado_civil', 'Soltero'), 0)
        ciudad_code = float(data.get('ciudad_code', 1))
        steroid = 1 if data.get('steroid') == 'Si' else 0
        antivirals = 1 if data.get('antivirals') == 'Si' else 0
        fatigue = 1 if data.get('fatigue') == 'Si' else 0
        malaise = 1 if data.get('malaise') == 'Si' else 0
        anorexia = 1 if data.get('anorexia') == 'Si' else 0
        liver_big = 1 if data.get('liver_big') == 'Si' else 0
        liver_firm = 1 if data.get('liver_firm') == 'Si' else 0
        spleen_palpable = 1 if data.get('spleen_palpable') == 'Si' else 0
        spiders = 1 if data.get('spiders') == 'Si' else 0
        ascites = 1 if data.get('ascites') == 'Si' else 0
        varices = 1 if data.get('varices') == 'Si' else 0
        bilirubin = float(data.get('bilirubin', 0.5))
        alk_phosphate = float(data.get('alk_phosphate', 50))
        sgot = float(data.get('sgot', 25))
        albumin = float(data.get('albumin', 4))
        protime = float(data.get('protime', 12))
        histology = 1 if data.get('histology') == 'Si' else 0
        
        # Simple predictive model based on key risk factors
        # Weights assigned based on hepatitis risk patterns
        risk_score = (
            (age / 100) * 0.3 +
            sex * 0.1 +
            steroid * 0.2 +
            antivirals * 0.15 +
            fatigue * 0.15 +
            malaise * 0.1 +
            anorexia * 0.15 +
            liver_big * 0.25 +
            liver_firm * 0.2 +
            spleen_palpable * 0.2 +
            spiders * 0.3 +
            ascites * 0.35 +
            varices * 0.25 +
            (bilirubin / 5) * 0.3 +
            (alk_phosphate / 100) * 0.15 +
            (sgot / 100) * 0.2 +
            (albumin / 5) * 0.1 +
            (protime / 15) * 0.15 +
            histology * 0.4
        )
        
        # Apply sigmoid to get probability
        probability_positive = sigmoid(risk_score - 1.5)
        probability_negative = 1 - probability_positive
        
        # Determine prediction (threshold at 0.5)
        prediction = 1 if probability_positive >= 0.5 else 0
        
        result = {
            'prediction': int(prediction),
            'probability_negative': float(probability_negative),
            'probability_positive': float(probability_positive),
            'result': 'Positivo para Hepatitis' if prediction == 1 else 'Negativo para Hepatitis'
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
