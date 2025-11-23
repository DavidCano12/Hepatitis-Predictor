# 🏥 Hepatitis-Predictor

Aplicación web de Machine Learning para predicción de hepatitis usando modelos de regresión logística. Este proyecto implementa un clasificador entrenado con un dataset de hepatitis que utiliza 21 características clínicas, demográficas y de laboratorio.

## 📋 Descripción

**Hepatitis-Predictor** es una aplicación Flask desplegada en Render que proporciona predicciones en tiempo real sobre la probabilidad de que un paciente tenga hepatitis basándose en sus datos clínicos y de laboratorio. La aplicación utiliza un modelo de Regresión Logística entrenado en un dataset de hepatitis con excelente desempeño (Accuracy: 1.0).

## 🚀 Características Principales

- **Modelo ML entrenado**: Regresión Logística con 21 features
- **Predicción en tiempo real**: Interfaz web interactiva
- **Escalado de features**: Utiliza StandardScaler para normalización de datos
- **API REST**: Endpoint `/predict` para integraciones
- **Interfaz web amigable**: Formulario HTML5 con validación
- **Despliegue en producción**: Alojado en Render

## 📊 Características del Modelo (21 Variables)

### Información Demográfica
- **Age**: Edad del paciente (años)
- **Sex**: Sexo (Masculino/Femenino)
- **Estado_Civil**: Estado civil (Soltero/Casado/Divorciado/Viudo)
- **Ciudad**: Ciudad de residencia (código 1-50)

### Síntomas Clínicos
- **Fatigue**: Fatiga
- **Malaise**: Malestar general
- **Anorexia**: Falta de apetito

### Medicamentos/Tratamientos
- **Steroid**: Uso de esteroides
- **Antivirals**: Uso de antivirales

### Hallazgos Físicos
- **Liver_Big**: Hígado aumentado de tamaño
- **Liver_Firm**: Hígado firme
- **Spleen_Palpable**: Bazo palpable
- **Spiders**: Arañas vasculares
- **Ascites**: Acumulación de líquido en abdomen
- **Varices**: Varices esofágicas
- **Histology**: Histología positiva

### Pruebas de Laboratorio
- **Bilirubin**: Bilirrubina (mg/dL)
- **Alk_Phosphate**: Fosfatasa alcalina (U/L)
- **Sgot**: Transaminasa SGOT (U/L)
- **Albumin**: Albúmina (g/dL)
- **Protime**: Tiempo de protrombina (segundos)

## 📈 Desempeño del Modelo

```
Entrenamiento:
- Accuracy: 1.0 (100%)
- Precision: 1.0
- Recall: 1.0
- F1-Score: 1.0

Test:
- Accuracy: 1.0 (100%)
- Precision: 1.0
- Recall: 1.0
- F1-Score: 1.0
```

## 🌐 Despliegue

**URL en vivo**: https://hepatitis-predictor.onrender.com

La aplicación está desplegada en Render usando:
- Python 3.x
- Flask
- scikit-learn
- NumPy y Pandas

## 📦 Instalación Local

```bash
# Clonar el repositorio
git clone https://github.com/DavidCano12/Hepatitis-Predictor.git
cd Hepatitis-Predictor

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la aplicación
python app.py
```

Luego accede a: `http://localhost:5000`

## 🛠️ Estructura del Proyecto

```
Hepatitis-Predictor/
├── app.py                 # Aplicación Flask principal
├── requirements.txt       # Dependencias Python
├── Procfile              # Configuración para Render
├── templates/
│   └── index.html        # Interfaz web
└── README.md             # Este archivo
```

## 🔧 Dependencias

```
Flask==2.3.2
Werkzeug==2.3.6
scikit-learn==1.3.0
numpy==1.24.3
pandas==2.0.3
```

## 📡 API Endpoint

### POST `/predict`

**Request JSON**:
```json
{
  "age": 45,
  "sex": "Masculino",
  "estado_civil": "Casado",
  "ciudad_code": 10,
  "steroid": "No",
  "antivirals": "Si",
  "fatigue": "Si",
  "malaise": "No",
  "anorexia": "No",
  "liver_big": "Si",
  "liver_firm": "No",
  "spleen_palpable": "No",
  "spiders": "No",
  "ascites": "No",
  "varices": "No",
  "bilirubin": 0.8,
  "alk_phosphate": 55.0,
  "sgot": 35.0,
  "albumin": 3.5,
  "protime": 12.5,
  "histology": "No"
}
```

**Response**:
```json
{
  "prediction": 0,
  "probability_negative": 0.92,
  "probability_positive": 0.08,
  "result": "Negativo para Hepatitis"
}
```

## 🎯 Cómo Usar

1. Accede a: https://hepatitis-predictor.onrender.com
2. Completa el formulario con los datos del paciente
3. Haz clic en "Realizar Predicción"
4. Observa el resultado mostrando:
   - Predicción (Positivo/Negativo)
   - Probabilidad de ser negativo
   - Probabilidad de ser positivo

## 🔐 Nota de Seguridad

Esta aplicación es para fines educativos. No debe usarse para diagnóstico clínico real sin validación profesional.

## 📚 Fuentes de Datos

Modelo original y dataset del Prof. Álvaro Pérez Niño - SENA
Repositorio de referencia: https://github.com/aperezn298/CienciaDatosSENA

## 👨‍💻 Autor

**David Cano**  
Estudiante de Ciencia de Datos - SENA

## 📄 Licencia

Este proyecto está disponible para uso educativo.

## 🤝 Contribuciones

Para sugerencias o mejoras, por favor abre un issue en el repositorio.
