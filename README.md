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
**2. Create and Activate a Virtual Environment**
python -m venv .venv
.\.venv\Scripts\activate
**### 2. Create and Activate a Virtual Environmen**
```bash
python -m venv .venv
.\.venv\Scripts\activate

**### 3. Install Dependencies**
```bash
pip install -r requirements.txt

**### 🚀 Usage Instructions
### ▶️ Run the Streamlit Application**

Launch the user interface for real-time analysis:
```bash
streamlit run main.py

**### 🧩 Train the Model**

Train the Random Forest model using your dataset:
```bash
python train_real_dataset.py

**### 🔍 Run Predictions in Code**
```bash
from models.deepfake_detector import DeepfakeDetector

**### Initialize the detector**
```bash
detector = DeepfakeDetector()

**### Run predictions**
```bash
result = detector.predict_from_features(features)
print(f"Confidence: {result['confidence']}%")
print(f"Is Deepfake: {result['is_deepfake']}")

📂 Project Structure
├── main.py                  # Streamlit application entry point
├── train_real_dataset.py    # Model training script
├── models/
│   ├── deepfake_detector.py     # Core detection logic
│   ├── feature_extractor.py     # Acoustic feature extraction
│   └── language_detector.py     # Language identification module
├── utils/
│   ├── audio_utils.py           # Audio processing utilities
│   ├── model_loader.py          # Model loading helpers
│   └── visualization.py         # Visualization and plotting tools
└── trained_models/              # Trained model artifacts

**## 🤝 Contributing**

Contributions and suggestions are highly appreciated!
Feel free to fork the repository and submit a pull request with improvements or new features.

**## 📜 License**

This project is licensed under the MIT License — see the LICENSE file for more information.

**##👩‍💻 Author**

Created by Sakshi Khanvilkar

**##🙏 Acknowledgments**
Special thanks to all contributors, researchers, and open-source developers whose work inspired and supported the development of this project.
