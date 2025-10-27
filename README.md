# 🎧 Audio Deepfake Detection System

An advanced **machine learning–driven solution** for detecting audio deepfakes through **acoustic feature analysis** and **random forest classification**.

---

## 🧠 Overview

This project introduces a **robust and intelligent audio deepfake detection framework** that leverages **53-dimensional acoustic features** to distinguish between authentic and synthetically generated speech.  
Using a **Random Forest classifier**, the system delivers **high accuracy**, **low false positives**, and **real-time detection capabilities**.

---

## ✨ Key Features

- Extraction of **53-dimensional acoustic features**  
- **Random Forest–based classification** for deepfake detection  
- **Real-time audio analysis** with multi-format support  
- **Interactive visualizations** of prediction results  
- **Language detection** integrated into the analysis pipeline  
- **High detection accuracy** with optimized model performance  

---

## 🧰 Technology Stack

| Category | Technologies |
|-----------|---------------|
| **Machine Learning Frameworks** | scikit-learn, XGBoost |
| **Audio Processing** | librosa, soundfile, praat-parselmouth |
| **Deep Learning** | PyTorch |
| **Speech Recognition** | OpenAI Whisper |
| **Data Analysis** | NumPy, Pandas |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **User Interface** | Streamlit |

---

## ⚙️ Installation Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Sakshi-21-Hub/Deepfake.git
cd deepfake-release
---
```
### 2. Create and activate a virtual environment:
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. Install dependencies:
```powershell
pip install -r requirements.txt
```

## Usage

### Running the Application

To start the Streamlit application:

```powershell
streamlit run main.py
```

### Training the Model

To train the model on your dataset:

```powershell
python train_real_dataset.py
```

### Running Predictions

Use the trained model for predictions:

```python
from models.deepfake_detector import DeepfakeDetector

# Initialize detector
detector = DeepfakeDetector()

# Make predictions
result = detector.predict_from_features(features)
print(f"Confidence: {result['confidence']}%")
print(f"Is Deepfake: {result['is_deepfake']}")
```

## Project Structure

```
├── main.py                 # Main application entry point
├── train_real_dataset.py   # Model training script
├── models/
│   ├── deepfake_detector.py    # Core detector implementation
│   ├── feature_extractor.py    # Feature extraction
│   └── language_detector.py    # Language detection
├── utils/
│   ├── audio_utils.py      # Audio processing utilities
│   ├── model_loader.py     # Model loading utilities
│   └── visualization.py    # Visualization tools
└── trained_models/         # Saved model artifacts
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Developed by Neha Gupta

## Acknowledgments
