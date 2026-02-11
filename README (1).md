# Hand-Sign-Language
# 🧠 Sign Language Alphabet Recognizer (with Groq + Offline Word Prediction)

## 📋 Description
This project uses a **TensorFlow/Keras-based sign language recognition model** with **real-time webcam inference**.  
It combines **offline word prediction** (via `wordfreq` and `symspellpy`) with **online next-word prediction** using the **Groq API**.  
The interface displays top predictions, live letter detection, and predictive text suggestions — and can **speak the sentence aloud** using text-to-speech.

---

## ⚙️ Requirements

- **Python 3.11.0**
- The following libraries (install via `requirements.txt`):

```bash
pip install -r requirements.txt
```

### `requirements.txt`
```
numpy>=1.24.0
opencv-python>=4.7.0
tensorflow==2.18.0
keras==3.5.0
groq>=0.1.0
wordfreq>=3.0.0
symspellpy>=6.7.0
pyttsx3>=2.90
tqdm>=4.64.0
```

---

## 🧩 Setup Instructions

### 1. Create and activate a virtual environment
#### Windows (PowerShell)
```powershell
python -m venv venv
.env\Scripts\Activate.ps1
```

#### Linux/macOS
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Set up Groq API key
Create a free Groq account and get an API key.  
Then set it as an environment variable:

#### Windows (PowerShell)
```powershell
$env:GROQ_API_KEY = "your_groq_api_key_here"
```

#### Linux/macOS
```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

---

## 🎥 Running the Program
```bash
python webcam_infer_ui_hybrid_final_adjusted.py
```

---

## ⌨️ Keyboard Controls

| Key | Action |
|-----|---------|
| `a` | Speak the current sentence (Text-to-Speech) |
| `n` | Clear the sentence and start a new one (“New sentence”) |
| `← / →` | Navigate between word predictions |
| `Enter` | Accept the selected prediction |
| `1–5` | Choose a prediction directly |
| `x` or `Backspace` | Delete the last character |
| `ESC` | Exit the program |

---

## 🧠 Notes
- Works best with **Python 3.11.x** and **TensorFlow 2.18.0**.  
- The webcam feed uses a defined ROI box for gesture detection.
- Predictions are shown at the bottom; the model’s live letter output appears at the top right of the box.
- The TTS runs asynchronously — you can speak multiple times without freezing the UI.
